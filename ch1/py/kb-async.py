from langchain_core.runnables import chain
from llm_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

model = LLMFactory.get_llm("qwen-plus")


@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)


async def main():
    return await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})

if __name__ == "__main__":
    import asyncio
    print(asyncio.run(main()))
