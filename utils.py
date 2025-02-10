from deepeval.models.base_model import DeepEvalBaseLLM
from http_provider import LLMProvide
from typing import Tuple
class DeepEvalModelInterface(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        model_name="Local LLM",
    ):
        self.model = model
        self.model_name = model_name

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        text =  str(chat_model.invoke(prompt))
        return text

    async def a_generate(self, prompt: str) -> str:
        raise RuntimeError("Not implemented For Async Mode")
        # chat_model = self.load_model()
        # res = await chat_model.ainvoke(prompt)
        # return res
    

    def get_model_name(self):
        return self.model_name


class LocalModel():
    @staticmethod
    def create(model_name: str, base_url: str, authorization_token: str, system_prompt: str):
        return LLMProvide(
                model=model_name,
                base_url=base_url,
                headers={
                    "Authorization": authorization_token
                },
                system=system_prompt,
            ) 


# local_model = LocalModel.create(
#     model_name="phi4:latest", 
#     base_url="https://ollama.lazyinwork.com/api/chat/completions", 
#     authorization_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjVkY2UyMDkwLWQ4NTItNGM3Mi1iZGY3LWVjZWUxZDZiODIyNCJ9.gmLSbsu4zc0IBJauCKBzRa8sdJr-C03lMVYYyd73tS4", 
#     system_prompt="be a good agent"
# )
# agent = DeepEvalModelInterface(model=local_model, model_name="good agent")
# text = agent.generate("你是誰")
# print(text)