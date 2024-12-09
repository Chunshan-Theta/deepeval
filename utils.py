from deepeval.models.base_model import DeepEvalBaseLLM
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



class OllamaInterface(DeepEvalModelInterface):
    def __init__(
        self,
        model
    ):
        self.model = model
        self.model_name = 'ollama'

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return str(chat_model.invoke(prompt))

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return str(res)

    def get_model_name(self):
        return "Local LLM (ollama)"

