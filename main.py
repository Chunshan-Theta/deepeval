from deepeval.metrics import AnswerRelevancyMetric
from langchain_community.llms import Ollama
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from local_llm import LocalLLM
import configparser

# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('llmsetting.conf')
base_url = config.get('ollama', None)

# Replace these with real values
custom_model = Ollama(
    model="llama3.1:8b",
    base_url=base_url,
) 

llama = LocalLLM(model=custom_model)
metric = AnswerRelevancyMetric(model=llama)


answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.5,
    model=llama,
    async_mode=False
)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output of your LLM application
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
)
answer_relevancy_metric.measure(test_case)
print(answer_relevancy_metric.score)
print(answer_relevancy_metric.reason)