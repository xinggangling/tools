import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from simple_transformer import DigitTransformer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['STHeiti',
                                          'PingFang SC', 'Arial Unicode MS']  # 适配macOS常见中文字体
matplotlib.rcParams['axes.unicode_minus'] = False


def get_mnist_data():
    """加载MNIST数据，只选择数字1和2"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载训练集
    train_dataset = datasets.MNIST(
        'data', train=True, download=True, transform=transform)
    # 只选择数字1和2
    train_indices = [i for i, (_, label) in enumerate(
        train_dataset) if label in [1, 2]]
    train_dataset = Subset(train_dataset, train_indices)

    # 加载测试集
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    # 只选择数字1和2
    test_indices = [i for i, (_, label) in enumerate(
        test_dataset) if label in [1, 2]]
    test_dataset = Subset(test_dataset, test_indices)

    return train_dataset, test_dataset


def get_my_data():
    """加载自采集的手写数据"""
    import cv2
    import numpy as np
    from PIL import Image

    def mnist_style(img):
        # 灰度
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 自适应阈值二值化
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        # 裁剪数字区域
        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        digit = img[y:y+h, x:x+w]
        # 缩放到20x20
        digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
        # 四周补白到28x28
        padded = np.pad(digit, ((4, 4), (4, 4)), 'constant', constant_values=0)
        return padded

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        # 使用ImageFolder加载原始图片
        raw_dataset = datasets.ImageFolder('my_digits', transform=None)

        # 创建处理后的数据集
        processed_images = []
        processed_labels = []

        for img, label in raw_dataset:
            # 转换为numpy数组
            img_np = np.array(img)
            # 应用MNIST风格的预处理
            processed = mnist_style(img_np)
            # 转换为PIL Image
            processed = Image.fromarray(processed)
            # 应用标准化
            processed = transform(processed)
            processed_images.append(processed)
            processed_labels.append(label)

        # 创建自定义数据集
        class ProcessedDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]

        dataset = ProcessedDataset(processed_images, processed_labels)

        # 划分训练集和测试集
        n = len(dataset)
        train_size = int(0.8 * n)
        test_size = n - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        print(f"成功加载自采集数据: 训练集 {train_size}张, 测试集 {test_size}张")
        return train_dataset, test_dataset
    except Exception as e:
        print(f"加载自采集数据失败: {e}")
        return None, None


def get_my_data1():
    """加载自采集的手写数据"""
    import cv2
    import numpy as np
    from PIL import Image

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        dataset = datasets.ImageFolder('my_digits', transform=transform)
        # 划分训练集和测试集
        n = len(dataset)
        train_size = int(0.8 * n)
        test_size = n - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        print(f"成功加载自采集数据: 训练集 {train_size}张, 测试集 {test_size}张")
        return train_dataset, test_dataset
    except Exception as e:
        print(f"加载自采集数据失败: {e}")
        return None, None


def get_combined_data():
    """合并MNIST和自采集数据"""
    # 加载MNIST数据
    mnist_train, mnist_test = get_mnist_data()
    print(f"加载MNIST数据: 训练集 {len(mnist_train)}张, 测试集 {len(mnist_test)}张")

    # 加载自采集数据
    my_train, my_test = get_my_data1()

    if my_train is not None and my_test is not None:
        # 合并数据集
        train_dataset = ConcatDataset([mnist_train, my_train])
        test_dataset = ConcatDataset([mnist_test, my_test])
        print(f"合并后数据: 训练集 {len(train_dataset)}张, 测试集 {len(test_dataset)}张")
    else:
        print("使用纯MNIST数据训练")
        train_dataset = mnist_train
        test_dataset = mnist_test

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

    # 加载合并后的数据
    train_dataset, test_dataset = get_combined_data()

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
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, device)
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
