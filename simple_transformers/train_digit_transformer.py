import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from simple_transformer import DigitTransformer
import matplotlib.pyplot as plt

def get_mnist_data():
    """加载MNIST数据，只选择数字1和2"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载训练集
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    # 只选择数字1和2
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in [1, 2]]
    train_dataset = Subset(train_dataset, train_indices)
    
    # 加载测试集
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    # 只选择数字1和2
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in [1, 2]]
    test_dataset = Subset(test_dataset, test_indices)
    
    return train_dataset, test_dataset

def train_model(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 将标签转换为0和1（原来是1和2）
        target = target - 1
        
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'训练批次: {batch_idx}/{len(train_loader)}, '
                  f'损失: {loss.item():.4f}, '
                  f'准确率: {100.*correct/total:.2f}%')
    
    return total_loss / len(train_loader), 100.*correct/total

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target - 1  # 将标签转换为0和1
            
            output = model(data.view(-1, 784))
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return 100.*correct/total

def plot_training_history(train_losses, train_accs, test_accs):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(test_accs, label='测试准确率')
    plt.title('准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    train_dataset, test_dataset = get_mnist_data()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # 创建模型
    model = DigitTransformer(d_model=64, num_heads=4).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    num_epochs = 10
    train_losses = []
    train_accs = []
    test_accs = []
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # 训练
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 评估
        test_acc = evaluate_model(model, test_loader, device)
        test_accs.append(test_acc)
        
        print(f'训练损失: {train_loss:.4f}')
        print(f'训练准确率: {train_acc:.2f}%')
        print(f'测试准确率: {test_acc:.2f}%')
    
    # 绘制训练历史
    plot_training_history(train_losses, train_accs, test_accs)
    
    # 保存模型
    torch.save(model.state_dict(), 'digit_transformer.pth')
    print('模型已保存到 digit_transformer.pth')

if __name__ == '__main__':
    main() 