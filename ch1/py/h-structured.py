from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from llm_factory import LLMFactory


# llm = LLMFactory.get_llm("qwen-plus")
llm = LLMFactory.get_siliconflow_llm("deepseek-ai/DeepSeek-R1")

class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""

    answer: str
    """The answer to the user's question"""
    justification: str
    """Justification for the answer"""

structured_llm = llm.with_structured_output(AnswerWithJustification)

response = structured_llm.invoke("一磅砖头和一磅羽毛哪个更重")
print(response)