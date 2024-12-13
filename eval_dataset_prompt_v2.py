from deepeval.metrics import AnswerRelevancyMetric, GEval
from http_provider import AnswerAIProvide
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test
from utils import DeepEvalModelInterface
import configparser
from deepeval.dataset import EvaluationDataset
import pandas as pd
from evals import _evaluation_standard_same_meaning







# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('llmsetting.conf')
base_url = config.get('answerAI', "base_url")
token = config.get('answerAI', "token")
assert base_url is not None, "Please provide a valid base_url in llmsetting.conf"
MODEL_NAME = "gpt-4o-mini"



##
custom_model = AnswerAIProvide(
    model=MODEL_NAME,
    base_url=base_url,
    headers={
        "Authorization": token
    },
    system="最後給出的`reason`請用台灣繁體中文進行輸出，且不得包含`\"`字元",
) 

model = DeepEvalModelInterface(model=custom_model, model_name=MODEL_NAME)



# print("\n\n\n-------------------")
def get_correctness_metric_score(test_case):
    correctness_metric = GEval(
        name="Correctness",
        model=model,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT],
        evaluation_steps=_evaluation_standard_same_meaning,
        async_mode=False,
        strict_mode=False # the sorce will be 0
    )

    try:
        correctness_metric.measure(test_case)
        return correctness_metric.score, correctness_metric.reason
    except Exception as e:
        return "0", f"!!!SYSTEM ERROR!!! {str(e)}"


# load the dataset
df = pd.read_csv('error_case_with_v2_reply.csv')


## prepare the test cases

def test_source(question_content: str, retrieval_context: list, expected_output: str, actual_output: str):
    return LLMTestCase(**{
        "input": question_content,
        "retrieval_context": retrieval_context,
        "expected_output": expected_output,
        "actual_output": actual_output if "[answer]" not in actual_output else actual_output.split("[answer]")[1]
    })

df[['v3_score','v3_reason']] = df.apply(lambda x: 
   pd.Series(get_correctness_metric_score(test_source(x['question'],[], x['note'], x['answer_v3']))),
    axis=1)
df.to_csv('error_case_with_score_v2.csv', index=False)

