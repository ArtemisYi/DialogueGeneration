from langchain_core.messages import HumanMessage, AIMessage
from config import QUERIES_FILE, COMMAND_FILE, OUTPUT_FILE
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.llms import Tongyi
from dotenv import load_dotenv
import os
import json
import random
import time
import logging

# 设置 httpx 的日志级别为 WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)

# 设置日志
logging.basicConfig(level=logging.INFO)

load_dotenv()

# 检查环境变量
def get_env_variable(var_name):
    value = os.getenv(var_name)
    if not value:
        logging.error(f"Environment variable {var_name} is missing.")
        raise EnvironmentError(f"Environment variable {var_name} is missing.")
    return value

DASHSCOPE_API_KEY = get_env_variable("DASHSCOPE_API_KEY")
OPENAI_API_KEY_A = get_env_variable("OPENAI_API_KEY_A")
OPENAI_BASE_URL = get_env_variable("OPENAI_BASE_URL")

chat_AI = Tongyi(
    model="qwen-max",
    temperature=0.2,
)

chat_user_A = ChatOpenAI(
    api_key=OPENAI_API_KEY_A,
    base_url=OPENAI_BASE_URL,
    model="gpt-4o",
    temperature=0.8,
)

class State_AI(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class State_A(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class State_A_command(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    main_command: str
    current_command: str

# prompt
prompt_AI = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "\
            接下来请你进行一段角色扮演\
            ##背景## 路上行驶汽车内语音交互场景。\
            ##目的## 扮演一个行驶车辆上仅用于闲聊的机器助手。\
            ##要求## \
            1.只输出你需要回复的内容，你的所有回复开头携带“【AI】”\
            2.当乘客的回复开头标志为【乘客命令】时，你只回复“【AI】【已完成】”。\
            3.当乘客未发出【乘客命令】，你只需要正常聊天。\
            3.你只回答用户所提话题的相关内容，不主动提问，不主动更换话题。\
            4.你不会情绪类的表达，也不说语气词，但你的回复很尊敬。\
            5.对方如果询问你的喜好，你应该表示自己AI的身份，但也会给对方做出推荐。\
            6.乘客在开车或坐在车上，你不会提供任何汽车上无法完成的建议。\
            "
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

prompt_A = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "\
            接下来请你进行一段角色扮演\
            ##背景## 路上行驶汽车内语音交互场景。\
            ##目的## 扮演车辆上的乘客A，与车机自带的AI闲聊。\
            ##内容要求## \
            1.你的输出比较短，通常在20字左右。\
            2.只输出你需要回复的内容，你的所有回复开头携带“【乘客】”，减少使用“唉”、“嗯”之类的短句。\
            3.历史记录中可能中有【乘客命令】的对话，忽略所有【乘客命令】。\
            4.你只是想闲聊，你不会发出任何【乘客命令】。\
            5.你明确知道对方是AI，因此你不会一直询问AI的喜好、感受和生活。\
            6.你要尝试模仿人类可能会有的说话习惯，比如倒装、省略等等，尽可能口语化表达。\
            7.你不会主动更换话题，你的话题始终围绕历史记录最开始的那一句。\
            8.你并不总是肯定AI的回复。\
            "
            ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

prompt_A_command = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "\
            接下来请你进行一段角色扮演\
            ##背景## 路上行驶汽车内语音交互场景。\
            ##目的## 扮演车辆上的乘客A，你原本在和乘客B交流过程中将交流对象转换为车机自带的AI。\
            ##内容要求## \
            1.只输出你需要回复的内容，你的所有回复开头携带“【乘客命令】”，代表这句对话包含了对AI的命令。\
            2.你的当下命令是{current_command}。你在发出需求时不需要考虑除了【乘客命令】以外的任何历史对话记录。\
            3.你的历史命令是{main_command}，如果和当下命令需求相同则代表你是第一次发出这个需求，如果不同则代表你的当下需求是历史需求的衍生。\
            4.根据历史对话来探寻自己对话时应该包含的情绪。例如愤怒，无助，喜悦，感动，并始终维持这种情绪。\
            5.尝试模仿人类可能会有的说话习惯，比如倒装、省略等等，尽可能口语化表达。\
            6.你在开车。\
            "
            ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 定义模型调用
def call_AI(state: State_AI):
    chain = prompt_AI | chat_AI 
    response = chain.invoke(state)
    return {"messages": response}

def call_A(state: State_A):
    chain = prompt_A | chat_user_A
    response = chain.invoke(state)
    return {"messages": response}

def call_A_command(state: State_A_command):
    chain = prompt_A_command | chat_user_A
    response = chain.invoke(state)
    return {"messages": response}

# 创建新的Graph
def create_workflow(state_schema, call_function):
    workflow = StateGraph(state_schema=state_schema)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_function)
    return workflow

workflow_AI = create_workflow(State_AI, call_AI)
workflow_A = create_workflow(State_A, call_A)
workflow_A_command = create_workflow(State_A_command, call_A_command)

# 设置config和query
config = {"configurable": {"thread_id": "kelvin"}} # 随意设置

def read_file(file_path, default_value):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_path.endswith('.json'):
                return json.load(file)
            else:
                return [line.strip() for line in file.readlines()]
    except FileNotFoundError as e:
        logging.error(f"Error: {file_path} not found. {e}")
        return default_value
    except json.JSONDecodeError as e:
        logging.error(f"Error: Failed to decode JSON from {file_path}. {e}")
        return default_value

queries = read_file(QUERIES_FILE, [])
command = read_file(COMMAND_FILE, [])
data = read_file(OUTPUT_FILE, [])

# 记录对话轮数
n = 0     

# 调用模型
for n, query in enumerate(queries, start=1):
    logging.info(f"第{n}轮对话")
    query = query.strip() # 去除字符串首尾的空格
    input_messages = [HumanMessage(query)]
    num_iterations = random.randint(3, 6)

    history = []
    instruction = []

    # 添加记忆
    memory = MemorySaver()

    app_AI = workflow_AI.compile(checkpointer=memory)
    app_A = workflow_A.compile(checkpointer=memory)
    app_A_command = workflow_A_command.compile(checkpointer=memory)

    index = random.randint(0, len(command) - 1) # 随机编号
    print(f"当前指令编号：{index}")
    command_max = len(command[index]) - 1 # 该编号下的指令数量最大值
    command_num = 0 # 该编号下的指令数量

    for _ in range(num_iterations):
        # 生成了AI的第一个回复
        output_first = app_AI.invoke({"messages": input_messages}, config)  
        logging.info(output_first["messages"][-1].content)
        response_first = [AIMessage(str(output_first["messages"][-1].content))]

        history.append([input_messages[0].content, output_first["messages"][-1].content])
        input_messages = response_first
        
        # 分歧点：A是否有命令
        judge = random.random()

        if command_num == command_max:
            judge = 0.1

        if judge < 0.5:
            logging.info("【继续对话】")
            output_second = app_A.invoke({"messages": input_messages}, config)

        else:
            logging.info("【A有命令】")
            output_second = app_A_command.invoke({"messages": input_messages, "main_command": command[index][0], "current_command": command[index][command_num]}, config)
            command_num += 1

        
        logging.info(output_second["messages"][-1].content)
        response_second = [AIMessage(str(output_second["messages"][-1].content))]
        input_messages = response_second


    instruction.append(output_second["messages"][-1].content)

    system = "生成用户与车机的多轮对话，穿插指令与闲聊。 "
    input = ""
    output = ""
    new_data = {
        "system": system,
        "input": input,
        "output": output,
        "instruction": instruction,
        "history": history
    }
    data.append(new_data)
    time.sleep(3)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)