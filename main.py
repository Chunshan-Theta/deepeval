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



print("\n\n\n-------------------")
correctness_metric = GEval(
    name="Correctness",
    model=model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT],
    evaluation_steps=[
       'Compare the actual output directly with the expected output to verify factual accuracy.',
       'Check if all elements mentioned in the expected output are present and correctly represented in the actual output.',
       'Assess if there are any discrepancies in details, values, or information between the actual and expected outputs.'
    ],
    strict_mode=False # the sorce will be 0
)

correctness_metric.measure(test_case)
print(correctness_metric.score, correctness_metric.reason)
# test_cases = [test_case]
# dataset = EvaluationDataset(test_cases=test_cases)
# dataset.evaluate([correctness_metric])

