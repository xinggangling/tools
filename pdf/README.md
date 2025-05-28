# PPT原位翻译工具

这是一个使用本地开源大模型的PPT原位翻译工具，可以将PPT文件中的文本内容翻译成目标语言。

## 功能特点

- 支持翻译PPT中的文本框内容
- 支持翻译PPT中的表格内容
- 使用本地Ollama运行的开源大模型进行翻译
- 保留原PPT的格式和样式
- 显示翻译进度条

## 前置要求

1. 安装Ollama
   - 访问 [Ollama官网](https://ollama.ai/) 下载并安装
   - 拉取需要的模型，例如：
     ```bash
     ollama pull mistral
     ```

2. 安装Python依赖
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 确保Ollama服务已启动并运行在默认端口(11434)

2. 运行程序：
   ```bash
   python ppt_translator.py
   ```

3. 按照提示输入：
   - 要翻译的PPT文件路径
   - 保存翻译后文件的路径
   - 目标语言（默认为中文）
   - 要使用的模型名称（默认为mistral）

## 支持的模型

- mistral
- llama2
- codellama
- 其他Ollama支持的模型

## 注意事项

- 请确保Ollama服务已正确启动
- 确保已下载要使用的模型
- 翻译质量取决于所选模型的能力
- 如果翻译内容较多，可能需要等待一段时间
- 建议在翻译前备份原始PPT文件

## 支持的语言代码

- zh-cn：简体中文
- en：英语
- ja：日语
- ko：韩语
- fr：法语
- de：德语
- es：西班牙语
- ru：俄语
- 更多语言代码请参考Google翻译支持的语言列表 