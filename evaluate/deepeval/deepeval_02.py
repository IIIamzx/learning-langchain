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
        criteria="""您是一位专业的数据标注员，负责评估模型输出的正确性。您的任务是根据以下评分标准给出评分：
                    <评分标准>
                    正确的答案应当：
                    - 提供准确且完整的信息
                    - 不包含事实性错误
                    - 回答问题的所有部分
                    - 逻辑上保持一致
                    - 使用精确和准确的术语

                    在打分时，您应该进行扣分的情况包括：
                    - 事实性错误或不准确的信息
                    - 不完整或部分的答案
                    - 具有误导性或模糊不清的陈述
                    - 错误的术语使用
                    - 逻辑不一致
                    - 缺失关键信息
                    </评分标准>

                    <指导说明>
                    - 仔细阅读输入的问题和模型的输出。
                    - 将输出与参考输出进行对比，以检查事实的准确性和完整性。
                    - 重点关注输出中所呈现信息的正确性，而非其风格或冗长程度。
                    </指导说明>

                    <提醒>
                    目标是评估回复的事实正确性和完整性。
                    </提醒>

                    <输入>
                    {{input}}
                    </输入>

                    <输出>
                    {{output}}
                    </输出>

                    <参考输出>
                    {{reference_output}}
                    </参考输出>""",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=custom_model
    )
    
    # 3. 创建测试用例
    test_cases = [
        LLMTestCase(
            name="100001",
            input="劳动合同必须包含哪些条款？",
            actual_output="缓刑，全称刑罚的暂缓执行，是指对触犯刑律，经法定程序确认已构成犯罪、应受刑罚处罚的行为人，先行宣告定罪，暂不执行所判处的刑罚。以下从适用条件、执行方式、法律后果几方面为你详细介绍：",  # 模型的实际输出
            expected_output="劳动合同需包含劳动者信息、工作内容、报酬、期限等条款。",  # 期望的正确答案
            tags=["法律知识"]
        ),
        LLMTestCase(
            name="100002",
            input="什么是个人所得税？",
            actual_output="个人所得税是国家对个人（自然人）取得的各项所得征收的一种税收，包括工资薪金、劳务报酬、财产租赁等收入。",
            expected_output="个人所得税是对个人收入征收的税种，包括工资、奖金、投资收益等各类所得。",
        ),
        LLMTestCase(
            name="100003",
            input="公司注册需要什么材料？",
            actual_output="公司注册需要提供：1.公司名称预先核准通知书 2.股东身份证明 3.注册资本证明 4.公司章程 5.注册地址证明 6.法定代表人任职文件等。",
            expected_output="公司注册需要股东身份证、公司章程、注册地址证明、名称核准书等材料。",
        ),
        LLMTestCase(
            name="100004",
            input="合同违约责任有哪些？",
            actual_output="今天天气很好，阳光明媚，适合外出游玩。建议大家多出门活动，享受美好的一天。",  # 完全不相关的答案
            expected_output="合同违约责任包括继续履行、采取补救措施、赔偿损失、支付违约金等形式。",
        ),
        LLMTestCase(
            name="100005",
            input="什么是知识产权？",
            actual_output="知识产权是指人们就其智力劳动成果所依法享有的专有权利，通常包括著作权、专利权、商标权等。它保护创新者的合法权益，鼓励发明创造。",
            expected_output="知识产权是对智力成果的专有权，包括专利权、商标权、著作权等，用于保护创新。",
        ),
        LLMTestCase(
            name="100006",
            input="离婚时财产如何分割？",
            actual_output="根据《民法典》规定，离婚时夫妻共同财产由双方协议处理；协议不成的，由人民法院根据财产的具体情况，按照照顾子女、女方和无过错方权益的原则判决。",
            expected_output="离婚财产分割遵循协议优先，协议不成由法院按照照顾子女、女方和无过错方原则判决。",
        ),
        LLMTestCase(
            name="100007",
            input="什么是刑事责任年龄？",
            actual_output="刑事责任年龄是指法律规定的能够承担刑事责任的最低年龄。我国《刑法》规定，已满16周岁的人犯罪，应当负刑事责任。已满14周岁不满16周岁的人，犯故意杀人等严重犯罪的，应当负刑事责任。",
            expected_output="刑事责任年龄是承担刑责的最低年龄，一般为16周岁，14-16周岁仅对严重犯罪承担责任。",
        ),
        LLMTestCase(
            name="100008",
            input="房屋买卖合同的效力如何认定？",
            actual_output="房屋买卖合同的效力需要满足以下条件：1.当事人具有相应的民事行为能力 2.意思表示真实 3.不违反法律法规强制性规定 4.标的物合法。满足这些条件的合同具有法律效力。",
            expected_output="房屋买卖合同效力需满足当事人有行为能力、意思表示真实、不违法、标的合法等条件。",
        ),
        LLMTestCase(
            name="100009",
            input="什么是行政处罚？",
            actual_output="行政处罚是指行政机关依法对违反行政管理秩序的公民、法人或者其他组织，以减损权益或者增加义务的方式予以惩戒的行为。包括警告、罚款、没收违法所得、责令停产停业等形式。",
            expected_output="行政处罚是行政机关对违法行为的惩戒，包括警告、罚款、没收、停业等形式。",
        ),
        LLMTestCase(
            name="100010",
            input="著作权保护期限是多久？",
            actual_output="我不知道具体的时间，但是我觉得应该很长，可能是几十年吧。不过我也不太确定，建议你去查一下相关法律条文。",  # 不准确且不专业的回答
            expected_output="著作权保护期限为作者终生及其死亡后50年，法人作品为首次发表后50年。",
        )
    ]
    
    # 4. 运行评测 - 必须传入列表
    print("🚀 开始运行评测...")
    
    # 注意：evaluate的第一个参数必须是list
    evaluation_result_object = evaluate(test_cases, [correctness_metric])


    # 从返回的对象中获取.results列表
    # 这是正确的访问方式
    results_list = evaluation_result_object.test_results
    
    for i, result in enumerate(results_list, 1):
        test_case_id = test_cases[i-1].name  # 获取对应测试用例的ID
        print(f"测试用例ID: {test_case_id} - 通过: {result.metrics_data[0].success}, 得分: {result.metrics_data[0].score}, 原因: {result.metrics_data[0].reason}")


if __name__ == "__main__":
    run_evaluation_and_print_results()
    print("\n🎉 评测完成！")