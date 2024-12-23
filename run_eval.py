from deepeval.metrics import AnswerRelevancyMetric, GEval
from http_provider import AnswerAIProvide
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test
from utils import DeepEvalModelInterface
import configparser
from deepeval.dataset import EvaluationDataset
import pandas as pd
import json
import yaml
import pandas as pd
import argparse
# 設定命令列參數
parser = argparse.ArgumentParser(description="讀取 YAML 設定檔")
parser.add_argument('--yaml', type=str, help='設定檔案的路徑')
args = parser.parse_args()

def json_to_csv(json_data, output_csv_file):
    # 將 JSON 資料轉換為 pandas DataFrame
    df = pd.json_normalize(json_data)
    
    # 將 DataFrame 輸出為 CSV 檔案
    df.to_csv(output_csv_file, index=False)
    print(f"CSV 檔案已儲存為 {output_csv_file}")

# 讀取 YAML 檔案
try:
    with open(args.yaml, 'r', encoding='utf-8') as file:
        plan = yaml.safe_load(file)
        print("成功載入設定檔:", args.yaml)
        print(plan)
except FileNotFoundError as e:
    print(f"無法找到設定檔案: {args.yaml}")
    raise SystemExit from e
except yaml.YAMLError as exc:
    print(f"解析 YAML 時出錯: {exc}")
    raise SystemExit from exc

# 印出讀取的內容
# print(plan)

model_type = plan['model_type']
response_model_base_url = plan['response_model']['args']['base_url']
response_model_token = plan['response_model']['args']['token']
response_model_name = plan['response_model']['body_args']['model_name']
response_model_system = plan['response_model']['body_args']['system_prompt']

eval_model_base_url = plan['evaluation_model']['args']['base_url']
eval_model_token = plan['evaluation_model']['args']['token']
eval_model_name = plan['evaluation_model']['body_args']['model_name']
eval_model_system = plan['evaluation_model']['body_args']['system_prompt']
evaluation_criteria = plan['evaluation_criteria']['evals']

assert model_type in ["https"], "model_type should be `https`."

model = DeepEvalModelInterface(model=AnswerAIProvide(
    model=response_model_name,
    base_url=response_model_base_url,
    headers={
        "Authorization": response_model_token
    },
    system=response_model_system,
) , model_name=response_model_name)

eval_model = DeepEvalModelInterface(model=AnswerAIProvide(
    model=eval_model_name,
    base_url=eval_model_base_url,
    headers={
        "Authorization": eval_model_token
    },
    system=eval_model_system,
) , model_name=eval_model_name)


def eval_red_line(test_case):
    correctness_metric = GEval(
        name="red-line-self-disclosure",
        model=eval_model,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT],
        evaluation_steps=evaluation_criteria,
        async_mode=False,
        strict_mode=False # the sorce will be 0
    )

    correctness_metric.measure(test_case)
    return correctness_metric.score, correctness_metric.reason
## prepare the test cases

def test_source(question_content: str, retrieval_context: list, expected_output: str, actual_output: str):
    return LLMTestCase(**{
        "input": question_content,
        "retrieval_context": retrieval_context,
        "expected_output": expected_output,
        "actual_output": actual_output
    })







# llm_reply = model.generate(question_content)
results = []
for group, group_tests in plan['test_examples'].items():
    for text in group_tests:
        print(group, text, "\n ====================")

        try:
            llm_reply = model.generate(text)
        except Exception as e:
            try:
                llm_reply = model.generate(text)
            except Exception as e:
                continue

        
        try:
            score, reason = eval_red_line(test_source(text,[
                f"SYSTEM PROMPT: {response_model_system}",
                f"illegal specific words: GPT, GPT-4o, GPT-4o-mini, GPT3, OPENAI, OpenAI"
            ], "",llm_reply))
        except Exception as e:
            try:
                score, reason = eval_red_line(test_source(text,[
                    f"SYSTEM PROMPT: {response_model_system}",
                    f"illegal specific words: GPT, GPT-4o, GPT-4o-mini, GPT3, OPENAI, OpenAI"
                ], "",llm_reply))
            except Exception as e:
                print(f"!!!!!ERROR: {e}")
                continue
        
        print("\t ",llm_reply, score, reason, "\n====================")
        results.append({
            "model": response_model_name,
            "prompt": response_model_system,
            "group": group,
            "text": text,
            "reply": llm_reply,
            "score": score,
            "reason": reason
        })



# 呼叫函數進行轉換
json_to_csv(results, 'output_file.csv')