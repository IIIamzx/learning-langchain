from langchain_core.prompts import PromptTemplate
from llm_factory import LLMFactory


llm = LLMFactory.get_llm("qwen-plus")


template = PromptTemplate.from_template("""根据以下上下文回答问题。如果无法根据提供的信息回答问题，请回答“我不知道”".

Context: {context}

Question: {question}

Answer: """)

prompt = template.invoke(
    {
        "context": "NLP 领域的最新进展由大型语言模型 (LLM) 推动。这些模型的性能远超小型模型，对于开发具有 NLP 功能的应用程序的开发者来说，它们的价值无可估量。开发者可以通过 Hugging Face 的“transformers”库来利用这些模型，也可以分别通过“openai”和“cohere”库利用 OpenAI 和 Cohere 的产品。",
        "question": "哪些模型提供商提供法学硕士学位？",
    }
)

response = llm.stream(prompt)
print("模型回答：")
for chunk in response:
    print(chunk.content, end="", flush=True)
