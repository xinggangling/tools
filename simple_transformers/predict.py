import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from simple_transformer import DigitTransformer


def load_model(model_path, device):
    """加载训练好的模型"""
    # 创建模型实例
    model = DigitTransformer(d_model=64, num_heads=4)
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model


def preprocess_image1(image_path):
    """预处理图像"""
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Grayscale(1),  # 转换为灰度图
        transforms.Resize((28, 28)),  # 调整大小为28x28
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])

    # 加载并转换图像
    image = Image.open(image_path)
    image = transform(image)
    return image


def preprocess_image(image_path):
    """预处理图像，去除背景并标准化为MNIST风格"""
    from PIL import Image, ImageOps
    import cv2
    import numpy as np

    # 1. 打开图片并转为灰度
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)

    # 2. 自适应阈值去背景（二值化）
    # 先高斯模糊去噪
    image_np = cv2.GaussianBlur(image_np, (5, 5), 0)
    # 自适应阈值
    image_np = cv2.adaptiveThreshold(
        image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 3. 找到数字的外接矩形并居中裁剪
    coords = cv2.findNonZero(image_np)
    x, y, w, h = cv2.boundingRect(coords)
    digit = image_np[y: y + h, x: x + w]

    # 4. 缩放到20x20，四周补白到28x28
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit, ((4, 4), (4, 4)), "constant", constant_values=0)

    # 5. 转为PIL Image
    final_img = Image.fromarray(padded)
    final_img.save("gray.png")

    # 6. 标准化为MNIST风格
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    return transform(final_img)


def predict_digit(model, image, device):
    """预测数字"""
    with torch.no_grad():
        # 添加批次维度
        image = image.unsqueeze(0).to(device)
        # 展平图像
        image = image.view(-1, 784)
        # 进行预测
        output = model(image)
        # 获取预测结果
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        # 将预测结果转换回原始标签（0->1, 1->2）
        prediction += 1
        confidence = probabilities[0][prediction - 1].item()

    return prediction, confidence


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model_path = "digit_transformer.pth"
    model = load_model(model_path, device)

    # 预测示例
    image_path = "1.png"  # 替换为你的测试图像路径
    try:
        # 预处理图像
        image = preprocess_image(image_path)

        print(f"image_path: {image_path}")
        print(f"image.shape: {image.shape}")

        # 进行预测
        prediction, confidence = predict_digit(model, image, device)

        print(f"预测结果: 数字 {prediction}")
        print(f"置信度: {confidence:.2%}")

    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()
