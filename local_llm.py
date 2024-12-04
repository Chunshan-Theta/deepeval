from deepeval.models.base_model import DeepEvalBaseLLM
from typing import Tuple

class LocalLLM(DeepEvalBaseLLM):
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
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res
    

    def get_model_name(self):
        return self.model_name


