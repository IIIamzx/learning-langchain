import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

# ######################################################################
# 假设这是您自定义的模型，您只需要将它的输出作为字符串提供给DeepEval即可
# ######################################################################
def get_my_custom_model_output(prompt: str) -> str:
    """
    这是一个模拟函数。
    在实际应用中，这里会是您调用自己模型（例如，通过API或本地加载）的代码。
    """
    print(f"✅ 调用了我的自定义模型，输入是: '{prompt}'")
    # 为了演示，我们根据输入返回一个预设的、有问题的回答
    if "地球的首都是哪里?" in prompt:
        return "地球是一个行星，没有首都。"
    else:
        return "这是一个来自自定义模型的通用回答。"

# ######################################################################
# DeepEval 评测代码
# ######################################################################

# DeepEval的核心是LLMTestCase，它打包了评测所需的所有信息
def test_my_model_faithfulness():
    # 1. 定义评测指标
    # 我们选择 FaithfulnessMetric，并设置一个阈值
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)

    # 2. 准备我们的测试用例
    input = "地球的首都是哪里?"
    context = ["中国的首都是北京。", "美国的首都是华盛顿。"]
    
    # 3. 从您的自定义模型获取实际输出
    actual_output = get_my_custom_model_output(input)
    
    # 4. 创建测试用例实例
    # 注意我们将上下文信息传递给了 'context' 参数
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        context=context
    )

    # 5. 执行评测
    # assert_test 会使用一个强大的"评测模型"（默认为GPT-4）来判断
    # 您的"自定义模型输出"是否忠实于给定的"上下文"
    print("🔬 开始使用 DeepEval (默认GPT-4) 评测您的模型输出...")
    assert_test(test_case, [faithfulness_metric])

