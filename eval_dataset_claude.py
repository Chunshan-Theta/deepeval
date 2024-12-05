from deepeval.metrics import AnswerRelevancyMetric, GEval
from custom_provider import AnswerAIProvide
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test
from local_llm import LocalLLM
import configparser
from deepeval.dataset import EvaluationDataset
import pandas as pd







# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('llmsetting.conf')
base_url = config.get('answerAI', "base_url")
token = config.get('answerAI', "token")
assert base_url is not None, "Please provide a valid base_url in llmsetting.conf"
MODEL_NAME = "claude-3-5-sonnet-20240620"



##
custom_model = AnswerAIProvide(
    model=MODEL_NAME,
    base_url=base_url,
    headers={
        "Authorization": token
    },
    system="最後給出的`reason`請用台灣繁體中文進行輸出，且不得包含`\"`字元",
) 

model = LocalLLM(model=custom_model, model_name=MODEL_NAME)



# print("\n\n\n-------------------")
def get_correctness_metric_score(test_case):
    correctness_metric = GEval(
        name="Correctness",
        model=model,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT],
        evaluation_steps=[
            'Directly compare actual output to expected output to verify factual accuracy.',
            'Check whether all elements mentioned in the expected output are present and correctly represented in the actual output.',
            'Evaluate whether there are differences in details, values, or information between actual output and expected output. ',
            'Evaluate whether all elements mentioned in the expected output are highlighted in the actual output',
            'Assess whether the actual output is of appropriate depth and readability for secondary school students'       
        ],
        strict_mode=False # the sorce will be 0
    )

    correctness_metric.measure(test_case)
    return correctness_metric.score, correctness_metric.reason


# load the dataset
df = pd.read_csv('1129_lib_error_case_with_claude_reply.csv')


## prepare the test cases

def test_source(question_content: str, retrieval_context: list, expected_output: str, actual_output: str):
    return LLMTestCase(**{
        "input": question_content,
        "retrieval_context": retrieval_context,
        "expected_output": expected_output,
        "actual_output": actual_output
    })

df[['correctness_metric_claude_score','correctness_metric_claude_reason']] = df.apply(lambda x: 
   pd.Series(get_correctness_metric_score(test_source(x['question'],[], x['note'], x['claude-3-5-sonnet-20240620_reply']))),
    axis=1)
df.to_csv('1129_lib_error_case_with_claude_score.csv', index=False)

