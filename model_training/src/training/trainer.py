import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import time
import os
import json
from typing import Dict, Optional, Tuple
from tqdm import tqdm

# 可选导入wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..models import TranslationModel
from ..data import TranslationDataset
from .metrics import compute_bleu, compute_loss
from .utils import save_checkpoint, load_checkpoint, setup_device


class Trainer:
    """
    翻译模型训练器
    支持混合精度训练、梯度累积、模型保存等功能
    """

    def __init__(self, model: TranslationModel,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: Dict = None):
        """
        初始化训练器
        Args:
            model: 翻译模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}

        # 设置设备
        self.device = setup_device()
        self.model.to(self.device)

        # 训练参数
        self.learning_rate = self.config.get('learning_rate', 5e-4)
        self.weight_decay = self.config.get('weight_decay', 1e-4)
        self.gradient_accumulation_steps = self.config.get(
            'gradient_accumulation_steps', 4)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)

        # 优化器和调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * self.config.get('num_epochs', 10)
        )

        # 混合精度训练（仅在CUDA设备上启用）
        self.use_amp = self.config.get(
            'use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_bleu = 0.0

        # 日志记录
        self.log_interval = self.config.get('log_interval', 100)
        self.save_interval = self.config.get('save_interval', 1000)
        self.eval_interval = self.config.get('eval_interval', 500)

        # 输出目录
        self.output_dir = self.config.get('output_dir', 'checkpoints')
        os.makedirs(self.output_dir, exist_ok=True)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略pad token

        # 初始化wandb（如果启用）
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get('wandb_project', 'translation-model'),
                config=self.config
            )

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        Returns:
            训练统计信息
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            src_mask = batch['src_mask'].to(self.device)
            tgt_mask = batch['tgt_mask'].to(self.device)

            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(src_ids, tgt_ids, src_mask, tgt_mask)
                    # 计算损失（目标序列偏移一位）
                    loss = self.criterion(
                        outputs.transpose(1, 2),
                        tgt_ids[:, 1:]
                    )
            else:
                outputs = self.model(src_ids, tgt_ids, src_mask, tgt_mask)
                loss = self.criterion(
                    outputs.transpose(1, 2),
                    tgt_ids[:, 1:]
                )

            # 梯度累积
            loss = loss / self.gradient_accumulation_steps

            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积步骤
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # 优化器步骤
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # 更新统计信息
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })

            # 记录日志
            if self.global_step % self.log_interval == 0:
                self._log_training_step(
                    loss.item() * self.gradient_accumulation_steps)

            # 保存检查点
            if self.global_step % self.save_interval == 0:
                self._save_checkpoint()

            # 验证
            if self.val_loader and self.global_step % self.eval_interval == 0:
                val_metrics = self.evaluate()
                self._log_validation(val_metrics)

        # 计算平均损失
        avg_loss = total_loss / num_batches

        return {
            'train_loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估模型
        Returns:
            验证指标
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                # 将数据移到设备
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)
                tgt_mask = batch['tgt_mask'].to(self.device)

                # 前向传播
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            src_ids, tgt_ids, src_mask, tgt_mask)
                        loss = self.criterion(
                            outputs.transpose(1, 2),
                            tgt_ids[:, 1:]
                        )
                else:
                    outputs = self.model(src_ids, tgt_ids, src_mask, tgt_mask)
                    loss = self.criterion(
                        outputs.transpose(1, 2),
                        tgt_ids[:, 1:]
                    )

                total_loss += loss.item()

                # 生成翻译结果用于BLEU计算
                predictions = self.model.generate(src_ids, max_len=50)

                # 收集预测和目标
                for pred, target in zip(predictions, tgt_ids):
                    # 解码预测和目标
                    pred_text = self._decode_sequence(pred)
                    target_text = self._decode_sequence(target)

                    all_predictions.append(pred_text)
                    all_targets.append([target_text])  # BLEU需要列表格式

        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        bleu_score = compute_bleu(all_predictions, all_targets)

        return {
            'val_loss': avg_loss,
            'val_bleu': bleu_score
        }

    def _decode_sequence(self, sequence: torch.Tensor) -> str:
        """解码序列为文本"""
        # 这里需要根据实际的分词器实现
        # 暂时返回简单的字符串表示
        return " ".join([str(id.item()) for id in sequence if id.item() not in [0, 1, 2, 3]])

    def _log_training_step(self, loss: float):
        """记录训练步骤日志"""
        log_data = {
            'train_loss': loss,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'epoch': self.current_epoch,
            'global_step': self.global_step
        }

        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.log(log_data)

        print(
            f"Step {self.global_step}: Loss = {loss:.4f}, LR = {log_data['learning_rate']:.6f}")

    def _log_validation(self, metrics: Dict[str, float]):
        """记录验证日志"""
        log_data = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            **metrics
        }

        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.log(log_data)

        print(
            f"Validation - Loss: {metrics.get('val_loss', 0):.4f}, BLEU: {metrics.get('val_bleu', 0):.4f}")

        # 更新最佳模型
        val_loss = metrics.get('val_loss', float('inf'))
        val_bleu = metrics.get('val_bleu', 0.0)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint('best_loss.pth')

        if val_bleu > self.best_val_bleu:
            self.best_val_bleu = val_bleu
            self._save_checkpoint('best_bleu.pth')

    def _save_checkpoint(self, filename: str = None):
        """保存检查点"""
        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pth"

        checkpoint_path = os.path.join(self.output_dir, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_val_bleu': self.best_val_bleu,
            'config': self.config
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_bleu = checkpoint.get('best_val_bleu', 0.0)

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"检查点已加载: {checkpoint_path}")

    def train(self, num_epochs: int):
        """
        开始训练
        Args:
            num_epochs: 训练轮数
        """
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"模型参数数量: {self.model.count_parameters():,}")
        print(f"设备: {self.device}")
        print(f"混合精度训练: {self.use_amp}")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_metrics = self.train_epoch()

            # 验证
            if self.val_loader:
                val_metrics = self.evaluate()
                self._log_validation(val_metrics)

            # 保存epoch检查点
            self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

            # 打印epoch总结
            print(f"Epoch {epoch + 1} 完成:")
            print(f"  训练损失: {train_metrics['train_loss']:.4f}")
            if self.val_loader:
                print(f"  验证损失: {val_metrics.get('val_loss', 0):.4f}")
                print(f"  验证BLEU: {val_metrics.get('val_bleu', 0):.4f}")

        # 训练完成
        total_time = time.time() - start_time
        print(f"训练完成！总用时: {total_time / 3600:.2f} 小时")

        # 保存最终模型
        self._save_checkpoint('final_model.pth')

        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.finish()
