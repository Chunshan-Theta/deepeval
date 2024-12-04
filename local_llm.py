from deepeval.models.base_model import DeepEvalBaseLLM

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

    def generate(self, prompt: str, **kwagrs) -> str:
        chat_model = self.load_model()
        return str(chat_model.invoke(prompt,**kwagrs))

    async def a_generate(self, prompt: str,**kwagrs) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt,**kwagrs)
        return str(res)

    def get_model_name(self):
        return self.model_name


