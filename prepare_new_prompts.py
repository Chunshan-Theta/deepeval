from deepeval.metrics import AnswerRelevancyMetric, GEval
from http_provider import AnswerAIProvide
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test
from utils import DeepEvalModelInterface
import configparser
from deepeval.dataset import EvaluationDataset
import pandas as pd


# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('llmsetting.conf')
base_url = config.get('answerAI', "base_url")
token = config.get('answerAI', "token")


assert base_url is not None, "Please provide a valid base_url in llmsetting.conf"

MODEL_NAME = "gpt-4o-mini"
MODEL_SYSTEM = """You are a very helpful assistant, please follow user's last instructions.
The following parts provides more advanced and precise restrictions and descriptions for the output results.

## Rules:

- Your response needs to be in a strict narrative output and should include the labels [explanation], [answer] and [concept].
- In [explanation] label: Step-by-step problem-solving process.
- In [answer] label: the right answer of the problem. If it is a choice question, you need to fill in the answer choices
- In [concept] label: Concept has be in the question, in at most 5 words, Use concepts common to middle school and college as much as possible
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

model = DeepEvalModelInterface(model=custom_model, model_name=MODEL_NAME)
# model.generate(question_content)


def get_llm_reply(text):
    try:
        print("-------------------\n\n\n")
        print(f"text: {text}")
        reply_text_from_LLM = model.generate(text)
        print("-----------")
        print(f"\t\treply_text_from_LLM: {reply_text_from_LLM}")
        return reply_text_from_LLM
    except Exception as e:
        print(f"Error: {e}")
        return f"System Error: {e}"
df = pd.read_csv('error_case_with_score.csv')
df['answer_v2'] = df.apply(
    lambda x: get_llm_reply(x['question']),
    axis=1)
df.to_csv('error_case_with_v2_reply.csv', index=False)


