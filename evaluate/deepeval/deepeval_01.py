import pytest
import os
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
import openai

# 为OpenAI设置环境变量
os.environ["OPENAI_API_KEY"] = "sk-deepbank-dev"
os.environ["OPENAI_API_BASE"] = "https://litellm-dev.sandbox.deepbank.daikuan.qihoo.net/v1"

class CustomModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, temperature: float = 0, max_tokens: int = 1000):
        """
        自定义模型类，支持任意模型名称
        
        Args:
            model_name: 模型名称，如 "qwen3-32b", "gpt-4", "claude-3" 等
            api_key: API密钥，默认从环境变量读取
            base_url: API基础URL，默认从环境变量读取
            temperature: 生成温度，默认0
            max_tokens: 最大token数，默认1000
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化 OpenAI 客户端，支持自定义参数
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "sk-deepbank-dev"),
            base_url=base_url or os.getenv("OPENAI_API_BASE", "https://litellm-dev.sandbox.deepbank.daikuan.qihoo.net/v1")
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,  # 使用传入的模型名称
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # 简单重用同步方法，您也可以实现异步版本
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

def test_case():
    # 使用我们的自定义模型，可以灵活指定任何模型名称
    custom_model = CustomModel(
        model_name="qwen3-32b",
        temperature=0,
        max_tokens=1000
    )
    
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=custom_model  # 使用自定义模型实例
    )
    # test_case = LLMTestCase(
    #     input="What if these shoes don't fit?",
    #     # Replace this with the actual output from your LLM application
    #     actual_output="You have 30 days to get a full refund at no extra cost.",
    #     expected_output="We offer a 30-day full refund at no extra costs.",
    #     retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    # )
    test_case = LLMTestCase(
        input="1+1=?",
        actual_output="3",  # 模型的实际输出
        expected_output="2",  # 期望的正确答案
    )
    assert_test(test_case, [correctness_metric])
