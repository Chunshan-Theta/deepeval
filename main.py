from deepeval.metrics import AnswerRelevancyMetric
from langchain_community.llms import Ollama
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from local_llm import LocalLLM
import configparser

# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('llmsetting.conf')
base_url = config.get('ollama', "base_url")
assert base_url is not None, "Please provide a valid base_url in llmsetting.conf"

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

test1 = {
    "input": "台灣的首都是哪裡",
    "retrieval_context": ["台灣的首都是台北市"],
    "expected_output": "台北市"

}

test_case = LLMTestCase(
    input=test1.get("input"),
    actual_output=llama.generate(test1.get("input")),
    expected_output=test1.get("expected_output"),
    retrieval_context=test1.get("retrieval_context")
)
answer_relevancy_metric.measure(test_case)
print(answer_relevancy_metric.score)
print(answer_relevancy_metric.reason)