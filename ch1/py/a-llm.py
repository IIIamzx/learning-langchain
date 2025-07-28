from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI

# 加载环境变量
load_dotenv()


# 使用标准 OpenAI API（推荐用于追踪测试）
llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key="sk-4000a9e2f25d469fb524a5174df01693",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# llm = ChatOpenAI(
#     model="kimi-k2-0711-preview",
#     openai_api_key="sk-mGrUVkP78Nbqt83f8o25s3pi2WgtvSuOMvK2hXZXDWxmfnOl",
#     openai_api_base="https://api.moonshot.cn/v1"
# )


# response = llm.invoke("The sky is")

response = llm.stream("今天天气怎么样？")
print("模型回答：")
for chunk in response:
    print(chunk.content, end="", flush=True)
