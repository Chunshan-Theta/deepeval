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
MODEL_SYSTEM = """
You are a very helpful assistant, please follow user's last instructions.
The following parts provides more advanced and precise restrictions and descriptions for the output results.

## Rules:

- If the previous assistant message include [step] or [concept] tag, ignore that message.
- Your response needs to be in a strict narrative output and should not include the labels [step] and [concept].
- Mainly, follow the instructions in the user's last message.
- If there is additional context in previous message, consider it as a reference.
- Mathematical expressions need to be displayed in LaTeX format.
- Do not enclose LaTeX content with ``` symbol.
"""




# Replace these with real values
custom_model = AnswerAIProvide(
    model=MODEL_NAME,
    base_url=base_url,
    headers={
        "Authorization": token
    },
    system=MODEL_SYSTEM,
) 

model = LocalLLM(model=custom_model, model_name="claude-3-5-sonnet-20240620")
# model.generate(question_content)


print("\n\n\n-------------------")
df = pd.read_csv('1129_error_case_with_score.csv')
df['claude-3-5-sonnet-20240620_reply'] = df.apply(
    lambda x: model.generate(x['question']),
    axis=1)
df.to_csv('1129_error_case_with_claude_reply.csv', index=False)


