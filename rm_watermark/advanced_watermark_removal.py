import cv2
import numpy as np
import os


class WatermarkRemover:
    def __init__(self):
        self.window_name = "Watermark Removal"
        self.image = None
        self.result = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.rectangles = []  # 存储所有框选的区域

    def interactive_removal(self, image_path):
        """交互式水印去除"""
        # 读取图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        self.result = self.image.copy()

        # 创建窗口
        cv2.namedWindow(self.window_name)

        # 设置鼠标回调
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # 显示原始图像
        cv2.imshow(self.window_name, self.image)

        print("\n=== 水印去除工具使用说明 ===")
        print("1. 框选操作：")
        print("   - 点击并拖动：框选水印区域")
        print("   - 松开鼠标：确认选择")
        print("2. 键盘操作：")
        print("   - 'r' 键：重置所有选择")
        print("   - 's' 键：保存结果")
        print("   - 'ESC' 键：退出程序")
        print("=======================\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('s'):  # 保存
                self._save_result(image_path)
            elif key == ord('r'):  # 重置
                self.rectangles = []
                self.result = self.image.copy()
                cv2.imshow(self.window_name, self.image)

        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 创建临时图像用于显示
                temp_image = self.image.copy()
                # 绘制所有已确认的矩形
                for rect in self.rectangles:
                    cv2.rectangle(temp_image, rect[0], rect[1], (0, 0, 255), 2)
                # 绘制当前正在框选的矩形
                cv2.rectangle(temp_image, self.start_point,
                              (x, y), (0, 0, 255), 2)
                cv2.imshow(self.window_name, temp_image)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)

            # 确保起点和终点形成有效的矩形
            if self.start_point != self.end_point:
                # 添加新的矩形到列表
                self.rectangles.append((self.start_point, self.end_point))

                # 创建掩码
                mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                # 填充所有矩形区域
                for rect in self.rectangles:
                    cv2.rectangle(mask, rect[0], rect[1], 255, -1)

                # 应用修复
                self.result = cv2.inpaint(
                    self.image, mask, 3, cv2.INPAINT_TELEA)

                # 显示结果
                cv2.imshow(self.window_name, self.result)

    def _save_result(self, image_path):
        """保存结果"""
        output_path = os.path.splitext(image_path)[0] + '_removed.png'
        cv2.imwrite(output_path, self.result)
        print(f"\n结果已保存到: {output_path}")


def main():
    remover = WatermarkRemover()
    image_path = "rm_watermark/dolphin.png"
    remover.interactive_removal(image_path)


if __name__ == "__main__":
    main()
