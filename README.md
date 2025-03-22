# 🪄 Magic-Prompt

一个基于AI的提示词（Prompt）生成工具，帮助用户创建高质量的AI提示模板。

## 功能特点

- 交互式提示词生成界面
- 引导式问答，获取用户需求
- 基于用户需求自动生成优质提示词模板
- 简洁美观的Gradio用户界面

## 技术栈

- Python 3.9+
- LangChain
- LangGraph
- OpenAI API
- Gradio

## 安装

1. 克隆代码库：
```bash
git clone https://github.com/yourusername/Magic-Prompt.git
cd Magic-Prompt
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 创建环境变量文件 `.env`：
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1 # 可选，如使用非默认端点
OPENAI_MODEL=gpt-3.5-turbo # 或其他支持的模型
```

## 启动应用

```bash
python app.py
```

启动后，访问浏览器显示的地址（通常为 http://127.0.0.1:7860）即可使用。

## 使用方法

1. 在输入框中描述你想要创建的提示词类型和目的
2. 与AI进行对话，明确提示词的目标、变量、约束和需求
3. 获取生成的提示词模板
4. 点击「New Chat」开始新的会话

## 示例

用户：我需要一个可以总结文章的提示词

AI：(会引导用户明确细节，如变量、约束等)

最终生成：(一个结构化的提示词模板)