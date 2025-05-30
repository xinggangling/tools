import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang SC', 'Arial Unicode MS']  # 适配macOS常见中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from simple_transformer import DigitTransformer

def generate_test_image():
    # 创建一个28x28的空白图像
    image = np.zeros((28, 28), dtype=np.float32)  # 确保使用float32类型
    
    # 绘制数字2
    # 上部分
    image[5:7, 8:20] = 1
    # 右上到左下的斜线
    for i in range(8):
        image[7+i, 19-i] = 1
    # 下部分
    image[15:17, 8:20] = 1
    # 底部
    image[17:19, 8:20] = 1
    
    # 保存图像
    plt.imsave('test_image.png', image, cmap='gray')
    print("测试图片已保存为 test_image.png")
    
    return image

def predict_digit(image):
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitTransformer(d_model=64, num_heads=4)
    model.load_state_dict(torch.load('digit_transformer.pth'))
    model.to(device)
    model.eval()
    
    # 转换图像为张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 预处理图像
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # 确保数据类型匹配
    image_tensor = image_tensor.float()  # 确保使用float类型
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(output, dim=1)
        
    return predicted.item(), probabilities[0].cpu().numpy()

if __name__ == "__main__":
    # 生成测试图片
    test_image = generate_test_image()
    
    # 预测
    predicted_digit, probabilities = predict_digit(test_image)
    
    # 显示结果
    print(f"\n预测结果：")
    print(f"预测的数字是：{predicted_digit}")
    print(f"预测概率：")
    print(f"数字1的概率：{probabilities[0]:.4f}")
    print(f"数字2的概率：{probabilities[1]:.4f}")
    
    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(test_image, cmap='gray')
    plt.title(f'预测结果: {predicted_digit}')
    plt.axis('off')
    plt.show() 