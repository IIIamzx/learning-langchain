import asyncio
from llm_factory import LLMFactory
from langchain_core.messages import HumanMessage


# llm = LLMFactory.get_llm("kimi-k2")
llm = LLMFactory.get_siliconflow_llm("deepseek-ai/DeepSeek-V3")

async def main():
    prompt = [HumanMessage("你是谁？ 你能够做什么")]
    
    # 使用异步流式输出
    async for chunk in llm.astream(prompt):
        print(chunk.content, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())


"""
astream() 返回的是一个 async_generator（异步生成器）
你不能在同步的 for 循环中直接使用异步生成器
需要在异步环境中使用 async for
"""