import os
from deepeval import evaluate
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

def run_evaluation_and_print_results():
    """
    运行评测并打印结果的函数
    """
    # 1. 创建自定义模型
    custom_model = CustomModel(
        model_name="qwen3-32b",
        temperature=0,
        max_tokens=1000
    )
    
    # 2. 定义评测指标
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'. Provide detailed reasoning for the score.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=custom_model
    )
    
    # 3. 创建测试用例 - 可以在这里设置断点查看测试数据
    test_cases = [
        LLMTestCase(
            input="1+1=?",
            actual_output="3",  # 错误答案
            expected_output="2",
        ),
        LLMTestCase(
            input="1+1=?", 
            actual_output="2",  # 正确答案
            expected_output="2",
        ),
        LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris",
            expected_output="Paris",
        ),
        LLMTestCase(
            input="What is the capital of France?",
            actual_output="London",  # 错误答案
            expected_output="Paris",
        )
    ]
    
    # 4. 运行评测 - 可以在这里设置断点查看评测过程
    print("🚀 开始运行评测...")
    test_results = evaluate(test_cases, [correctness_metric])  # 🔍 断点位置1: 评测开始
    
    # 5. 打印详细结果
    print("\n" + "="*60)
    print("📊 详细评测结果")
    print("="*60)
    
    passed_count = 0
    total_score = 0
    
    for i, test_result in enumerate(test_results):
        test_case = test_cases[i]
        metric_data = test_result.metrics_data[0]  # 🔍 断点位置2: 查看每个结果
        
        total_score += metric_data.score
        if metric_data.success:
            passed_count += 1
        
        # 🔍 断点位置3: 查看详细的评测数据
        print(f"\n📝 测试用例 {i+1}:")
        print(f"   问题: {test_case.input}")
        print(f"   实际输出: {test_case.actual_output}")
        print(f"   期望输出: {test_case.expected_output}")
        print(f"   评分: {metric_data.score:.2f}/1.0")
        print(f"   阈值: {metric_data.threshold}")
        print(f"   结果: {'✅ 通过' if metric_data.success else '❌ 失败'}")
        print(f"   评测原因: {metric_data.reason}")
        
        if metric_data.error:
            print(f"   错误信息: {metric_data.error}")
    
    # 6. 打印总结 - 🔍 断点位置4: 查看最终统计
    total_tests = len(test_results)
    average_score = total_score / total_tests
    pass_rate = passed_count / total_tests
    
    print("\n" + "="*60)
    print("📈 评测总结")
    print("="*60)
    print(f"总测试数: {total_tests}")
    print(f"通过数: {passed_count}")
    print(f"失败数: {total_tests - passed_count}")
    print(f"通过率: {pass_rate:.1%}")
    print(f"平均得分: {average_score:.2f}")
    print("="*60)

if __name__ == "__main__":
    run_evaluation_and_print_results()
    print("\n🎉 评测完成！")