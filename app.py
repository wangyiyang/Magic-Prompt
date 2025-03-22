import os
import uuid
import time
import logging
from typing import List, Generator, Tuple, Any, Dict, Optional

import gradio as gr
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("magic-prompt")
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import Annotated
from typing_extensions import TypedDict, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    temperature=0,
)

# Define templates
template = """你的身份是一个提示词专家，你的任务是从用户那里获取有关他们想要创建的提示词模板类型的信息。

你应该获取以下信息：

- 提示词的目标是什么
- 哪些变量将传递到提示词模板中
- 输出不应该做什么的任何约束
- 输出必须遵守的任何要求

如果你无法辨别这些信息，请要求用户澄清！不要尝试随意猜测。

在你能够辨别所有信息后，调用相关工具。"""

prompt_system = """根据以下需求，编写一个优质的提示词模板：

{reqs}"""

# Define PromptInstructions class
class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""
    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]

# 绑定工具到LLM
llm_with_tool = llm.bind_tools([PromptInstructions])

# 定义消息处理函数
def get_messages_info(messages):
    # 添加系统提示并保留历史消息
    return [SystemMessage(content=template)] + messages

def get_prompt_messages(messages: list):
    # 提取工具调用参数和相关消息
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            # 提取工具调用的参数信息
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            # 跳过工具消息
            continue
        elif tool_call is not None:
            # 收集工具调用后的消息
            other_msgs.append(m)
    
    # 使用提取的参数构建提示词系统指令
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs

# 定义工作流链函数
def info_chain(state):
    # 收集用户需求并引导用户提供必要信息
    messages = get_messages_info(state["messages"])
    # 调用带工具的LLM生成回应或工具调用
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

def prompt_gen_chain(state):
    # 生成实际的提示词模板
    messages = get_prompt_messages(state["messages"])
    # 调用LLM生成提示词模板
    response = llm.invoke(messages)
    
    # 格式化提示词模板并增加使用说明
    formatted_content = response.content.strip()
    enhanced_content = f"### 🪄 生成的提示词模板\n\n```\n{formatted_content}\n```\n\n您可以复制上面的提示词并根据需要进行调整。\n\n如果您对这个提示词模板有任何疑问或需要进一步改进，请告诉我！"
    
    # 创建带有增强格式的新响应消息
    enhanced_response = AIMessage(content=enhanced_content)
    return {"messages": [enhanced_response]}

# Define state handler
def get_state(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"

# Define State type
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create workflow
memory = MemorySaver()
workflow = StateGraph(State)

# Add nodes to workflow
workflow.add_node("info", info_chain)
workflow.add_node("prompt", prompt_gen_chain)

@workflow.add_node
def add_tool_message(state: State):
    # 提取工具调用信息
    tool_call = state["messages"][-1].tool_calls[0]
    tool_id = tool_call["id"]
    tool_args = tool_call["args"]
    
    # 从工具调用中提取需求内容用于展示
    if isinstance(tool_args, dict):
        objective = tool_args.get("objective", "")
        variables = tool_args.get("variables", [])
        constraints = tool_args.get("constraints", [])
        requirements = tool_args.get("requirements", [])
        
        # 构建需求概览文本
        reqs_summary = f"### 📋 需求概览\n\n"
        reqs_summary += f"**目标**: {objective}\n\n"
        
        if variables:
            reqs_summary += f"**变量**:\n"
            for var in variables:
                reqs_summary += f"- {var}\n"
            reqs_summary += "\n"
            
        if constraints:
            reqs_summary += f"**约束**:\n"
            for constraint in constraints:
                reqs_summary += f"- {constraint}\n"
            reqs_summary += "\n"
            
        if requirements:
            reqs_summary += f"**要求**:\n"
            for req in requirements:
                reqs_summary += f"- {req}\n"
                
        message_content = f"{reqs_summary}\n\n⏳ 正在根据您的需求生成提示词模板..."
    else:
        # 备用消息，如果没有正确提取到参数
        message_content = "⏳ 正在生成提示词模板..."  
        
    return {
        "messages": [
            ToolMessage(
                content=message_content,
                tool_call_id=tool_id,
            )
        ]
    }

# Add edges to workflow
workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")

# Compile workflow
graph = workflow.compile(checkpointer=memory)

# Message history and thread management
class ChatManager:
    def __init__(self):
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id, "verbose": True}}
        self.message_history = []
        
    def reset(self):
        """Reset the chat history and create a new thread."""
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id, "verbose": True}}
        self.message_history = []

# Initialize the chat manager
chat_manager = ChatManager()

# 处理用户消息并实现流式输出
def process_message(message: str, history: List[List[str]]) -> Generator[str, None, None]:
    """
    处理用户消息并流式输出响应。
    
    参数:
        message: 用户消息
        history: Gradio格式的聊天历史
        
    返回:
        生成器，流式输出响应
    """
    logger.info(f"Processing user message: {message[:50]}...")
    
    try:
        # 添加用户消息到内部历史记录
        chat_manager.message_history.append(HumanMessage(content=message))
        
        # 准备输入数据
        input_data = {"messages": chat_manager.message_history}
        
        # 初始化输出变量
        output = ""
        waiting_messages = []  # 等待消息列表，用于显示工具调用状态
        
        # 确保至少生成一个初始响应，避免生成器过早结束
        yield "正在思考..."
        
        # 日志记录开始拿取流式输出
        logger.info("Start streaming LangGraph response")
        
        # 使用一般模式获取返回内容，更加简单稳定
        for chunk in graph.stream(input_data, config=chat_manager.config):
            # 获取消息列表
            messages = next(iter(chunk.values())).get("messages", [])
            if not messages:
                logger.warning("Received empty messages from graph")
                continue
                
            logger.info(f"Received {len(messages)} messages from graph")
            
            # 获取最新消息
            latest_message = messages[-1]
            
            # 处理AI消息
            if isinstance(latest_message, AIMessage):
                if latest_message.content:
                    # 将消息添加到历史记录
                    if latest_message not in chat_manager.message_history:
                        chat_manager.message_history.append(latest_message)
                    
                    # 检测是否有工具调用
                    if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                        logger.info("Detected tool calls in AI message")
                        # 显示工具调用状态
                        output = f"{latest_message.content}\n\n正在分析您的需求..."
                    else:
                        # 更新响应
                        output = latest_message.content
                    
                    # 流式输出当前响应
                    logger.info(f"Yielding AI response: {output[:50]}...")
                    yield output
            
            # 处理工具消息
            elif isinstance(latest_message, ToolMessage):
                # 将消息添加到历史记录
                if latest_message not in chat_manager.message_history:
                    chat_manager.message_history.append(latest_message)
                
                # 显示工具消息状态
                logger.info("Received tool message (prompt template)")
                if "\n\n正在分析您的需求..." in output:
                    # 如果之前有分析状态，去掉它
                    base_content = output.split("\n\n正在分析您的需求...")[0]
                    output = f"{base_content}\n\n提示词模板生成完成!\n\n{latest_message.content}"
                else:
                    # 直接设置工具消息内容
                    output = latest_message.content
                
                logger.info(f"Yielding tool response: {output[:50]}...")
                yield output
        
        # 确保返回最终响应
        logger.info("Streaming completed, returning final response")
        return output
        
    except Exception as e:
        # 例外处理
        error_msg = f"非常抱歉，处理您的消息时出现错误: {str(e)}"
        logger.exception("Error in process_message:")
        yield error_msg
        return error_msg

# 重置聊天历史的函数
def reset_chat():
    chat_manager.reset()
    return None

# 使用ChatInterface定义Gradio应用
demo = gr.ChatInterface(
    fn=process_message,
    title="🪄 Magic-Prompt 提示词生成器",
    description="AI驱动的高质量提示词模板生成工具。描述您想要创建的提示词类型，我将指导您完成创建过程。",
    examples=[
        "我需要一个用于创意写作的提示词", 
        "帮我创建一个RAG系统的提示词", 
        "设计一个图像生成的提示词",
        "写一个能总结文章的提示词模板"
    ],
    theme="soft",
    type="messages",
    fill_height=True,
    chatbot=gr.Chatbot(
        height=650, 
        type="messages",
        show_copy_button=True
    ),
    save_history=True
)

# 启动应用
if __name__ == "__main__":
    # 显示启动信息
    print("✨ 正在启动 Magic-Prompt 提示词生成器...")
    print("💡 使用方法: 描述您想要创建的提示词，按照AI的引导提供信息")
    print("⚙️ 技术栈: LangChain + LangGraph + Gradio + OpenAI API")
    
    # 启动Gradio应用
    demo.launch(
        show_api=False,
        share=False,
        inbrowser=True,
        favicon_path="🪄"
    )
