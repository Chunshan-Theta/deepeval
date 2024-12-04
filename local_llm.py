from deepeval.models.base_model import DeepEvalBaseLLM

class LocalLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

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


