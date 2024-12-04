from deepeval.metrics import AnswerRelevancyMetric, GEval
from custom_provider import AnswerAIProvide
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test
from local_llm import LocalLLM
import configparser
from deepeval.dataset import EvaluationDataset








# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('llmsetting.conf')
base_url = config.get('answerAI', "base_url")
token = config.get('answerAI', "token")


assert base_url is not None, "Please provide a valid base_url in llmsetting.conf"

# Replace these with real values
custom_model = AnswerAIProvide(
    model="claude-3-5-sonnet-20240620",
    base_url=base_url,
    headers={
        "Authorization": token
    },
    system="回應問題之前，先感謝使用者",
) 

model = LocalLLM(model=custom_model, model_name="claude-3-5-sonnet-20240620")
test_source = {
    "input": "台灣的首都是哪裡",
    "retrieval_context": ["台灣的首都是台北市"],
    "expected_output": "台北市",
    "actual_output": model.generate("台灣的首都是哪裡")

}
print(f"test_source: {test_source}")
test_case = LLMTestCase(**test_source)

# print("\n\n\n-------------------")
# answer_relevancy_metric = AnswerRelevancyMetric(
#     threshold=0.5,
#     model=model,
#     async_mode=False
# )
# answer_relevancy_metric.measure(test_case)
# print(answer_relevancy_metric.score)
# print(answer_relevancy_metric.reason)

