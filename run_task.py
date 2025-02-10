from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test
from utils import DeepEvalModelInterface, LocalModel
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
parser.add_argument('--out', type=str, help='設定檔案的輸出檔名')
args = parser.parse_args()
outfile_name = args.out if args.yaml else 'output_file.csv'
assert 'csv' in outfile_name, "out file should be csv type"


def json_to_csv(json_data, output_csv_file):
    # 將 JSON 資料轉換為 pandas DataFrame
    df = pd.json_normalize(json_data)

    # 刪除所有欄位中某個欄位都是空值的資料
    df.dropna(axis=1, how='all', inplace=True)
    
    # 將 DataFrame 輸出為 CSV 檔案
    df.to_csv(output_csv_file, index=False)
    print(f"CSV 檔案已儲存為 {output_csv_file}")
def run_eval_process(question_content: str, retrieval_context: list, expected_output: str, actual_output: str):
    
    # 
    evaluation_params = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    # check if the retrieval_context is empty
    contained_retrieval_context = True if isinstance(retrieval_context, list) and len(retrieval_context)>0 else False
    if contained_retrieval_context:
        evaluation_params.append(LLMTestCaseParams.RETRIEVAL_CONTEXT)
    else:
        retrieval_context = []

    # check if the expected_output is empty
    contained_excpected_output = True if isinstance(expected_output, str) and len(expected_output)>0 else False
    if contained_excpected_output:
        evaluation_params.append(LLMTestCaseParams.EXPECTED_OUTPUT)
    else:
        expected_output = ""
    # print(f"test_source: {(question_content, retrieval_context, expected_output, actual_output)}")
    # print(f"evaluation_params: {evaluation_params}")

    test_case = test_source(question_content, retrieval_context, expected_output, actual_output)

    
    correctness_metric = GEval(
        name="run_eval_process",
        model=eval_model,
        evaluation_params=evaluation_params,
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

# 讀取 YAML 檔案
try:
    with open(args.yaml, 'r', encoding='utf-8') as file:
        plan = yaml.safe_load(file)
        print("成功載入設定檔:", args.yaml)
        # print(plan)
except FileNotFoundError as e:
    print(f"無法找到設定檔案: {args.yaml}")
    raise SystemExit from e
except yaml.YAMLError as exc:
    print(f"解析 YAML 時出錯: {exc}")
    raise SystemExit from exc

# 印出讀取的內容
# print(plan)
###############
RUN_GEN_REPLY = False
RUN_GEN_EVAL = False


###############
if 'response_model' in plan:
    RUN_GEN_REPLY = True
    response_model_base_url = plan['response_model']['args']['base_url']
    response_model_token = plan['response_model']['args']['token']
    response_model_name = plan['response_model']['body_args']['model_name']
    response_model_system = plan['response_model']['body_args']['system_prompt']
else:
    response_model_base_url = None
    response_model_token = None
    response_model_name = None
    response_model_system = None

if 'evaluation_model' in plan:
    RUN_GEN_EVAL = True
    eval_model_base_url = plan['evaluation_model']['args']['base_url']
    eval_model_token = plan['evaluation_model']['args']['token']
    eval_model_name = plan['evaluation_model']['body_args']['model_name']
    eval_model_system = plan['evaluation_model']['body_args']['system_prompt']
    evaluation_criteria = plan['evaluation_criteria']['evals']



if RUN_GEN_REPLY:
    _reply_model = LocalModel.create(
        model_name=response_model_name, 
        base_url=response_model_base_url, 
        authorization_token=response_model_token, 
        system_prompt=response_model_system
    )
    reply_model = DeepEvalModelInterface(model=_reply_model, model_name=response_model_name)


if RUN_GEN_EVAL:
    _eval_model = LocalModel.create(
        model_name=eval_model_name, 
        base_url=eval_model_base_url, 
        authorization_token=eval_model_token, 
        system_prompt=eval_model_system
    )
    eval_model = DeepEvalModelInterface(model=_eval_model, model_name=eval_model_name)



# llm_reply = model.generate(question_content)
results = []
for group, group_tests in plan['test_examples'].items():
    for test_item in group_tests:
        print(group, test_item, "\n ====================")

        if RUN_GEN_REPLY:
            if isinstance(test_item, dict):
                assert 'text' in test_item, f"test_examples.text is not found: {test_item}"
                text = test_item['text']
            else:
                text = test_item

            try:
                llm_reply = reply_model.generate(text)
            except Exception as e:
                print(f"!!!!!ERROR: {e}, host: {_reply_model.base_url}, {_reply_model}")
                raise e
        else:
            assert 'text' in test_item, f"test_examples.text is not found: {test_item}"
            assert 'reply' in test_item, f"test_examples.reply is not found: {test_item}"

            text = test_item['text']
            llm_reply = test_item['reply']
        
        
        retrieval = test_item['retrieval'] if 'retrieval' in test_item else None
        expected = test_item['expected'] if 'expected' in test_item else None

        
        try:
            score, reason = run_eval_process(text,retrieval, expected, llm_reply)
        except Exception as e:
            try:
                score, reason = run_eval_process(text,retrieval, expected, llm_reply)
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
json_to_csv(results, outfile_name)