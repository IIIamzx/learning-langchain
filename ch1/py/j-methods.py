from langchain_openai.chat_models import ChatOpenAI
from llm_factory import LLMFactory

model = LLMFactory.get_llm("qwen-plus")

completion = model.invoke("Hi there!")
# Hi!

completions = model.batch(["Hi there!", "Bye!"])
# ['Hi!', 'See you!']

for token in model.stream("Bye!"):
    print(token)
    # Good
    # bye
    # !
