from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    BaseCallbackManager,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.outputs import Generation, GenerationChunk, LLMResult, RunInfo
import requests
import aiohttp
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    Mapping
)
import json

class LLMEndpointNotFoundError(Exception):
    """Raised when the LLM endpoint is not found."""

def _stream_response_to_generation_chunk(
    stream_response: str,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    try:
        parsed_response = json.loads(stream_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse response: {stream_response}") from e

    return GenerationChunk(
        text=parsed_response.get("message", ""),
        generation_info=parsed_response if parsed_response.get("done") else None,
    )

class _HttpCommonModel(BaseLanguageModel):
    base_url: str = "http://localhost:11434"
    """Base url the model is hosted under."""

    model: str = "llama2"
    """Model name to use."""

    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity.
    (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: 0.1)"""

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: 5.0)"""

    num_ctx: Optional[int] = None
    """Sets the size of the context window used to generate the
    next token. (Default: 2048)	"""

    num_gpu: Optional[int] = None
    """The number of GPUs to use. On macOS it defaults to 1 to
    enable metal support, 0 to disable."""

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation.
    By default, LLM will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    num_predict: Optional[int] = None
    """Maximum number of tokens to predict when generating text.
    (Default: 128, -1 = infinite generation, -2 = fill context)"""

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"""

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
    will penalize repetitions more strongly, while a lower value (e.g., 0.9)
    will be more lenient. (Default: 1.1)"""

    temperature: Optional[float] = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: 0.8)"""

    stop: Optional[List[str]] = None
    """Sets the stop tokens to use."""

    tfs_z: Optional[float] = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., 2.0) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: 1)"""

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 40)"""

    top_p: Optional[float] = None
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 0.9)"""

    system: Optional[str] = None
    """system prompt (overrides what is defined in the Modelfile)"""

    template: Optional[str] = None
    """full prompt or prompt template (overrides what is defined in the Modelfile)"""

    format: Optional[str] = None
    """Specify the format of the output (e.g., json)"""

    timeout: Optional[int] = None
    """Timeout for the request stream"""

    keep_alive: Optional[Union[int, str]] = None
    """How long the model will stay loaded into memory.

    The parameter (Default: 5 minutes) can be set to:
    1. a duration string in Golang (such as "10m" or "24h");
    2. a number in seconds (such as 3600);
    3. any negative number which will keep the model loaded \
        in memory (e.g. -1 or "-1m");
    4. 0 which will unload the model immediately after generating a response;

    See the [Ollama documents](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately)"""

    raw: Optional[bool] = None
    """raw or not."""

    headers: Optional[dict] = None
    """Additional headers to pass to endpoint (e.g. Authorization, Referer).
    This is useful when LLM is hosted on cloud services that require
    tokens for authentication.
    """

    auth: Union[Callable, Tuple, None] = None
    """Additional auth tuple or callable to enable Basic/Digest/Custom HTTP Auth.
    Expects the same format, type and values as requests.request auth parameter."""

    keys_to_remove: List[str] = ["keep_alive", "options", "raw", "system", "template"]
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling LLM."""
        return {
            "model": self.model,
            "format": self.format,
            "options": {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "num_predict": self.num_predict,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "stop": self.stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
            "system": self.system,
            "template": self.template,
            "keep_alive": self.keep_alive,
            "raw": self.raw,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model, "format": self.format}, **self._default_params}

    
    def _extract_response(self, response: requests.Response) -> List[str]:
        """Extract the response from the requests response."""
        def _gen_text_chunk(content: str) -> str:
            return json.dumps({"message": content}, ensure_ascii=False)

        return [_gen_text_chunk(choice['message']['content']) for choice in response.json()['choices']]
    
    def _gen_requests_payload(self, payload: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        request_message = payload.get("messages", [])
        assert type(request_message) == list, "payload.messages should be a list"
        if params.get("system"):
            request_message.append({
                "role": "system",
                "content": [
                {
                    "type": "text",
                    "text": params.get("system")
                }
                ]
            })
        if payload.get("prompt"):
            request_message.append({
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": payload.get("prompt")
                }
                ]
            })
        request_payload = {
            "messages": request_message,
            "max_tokens": 2048,
            **params,
        }
        request_payload = {k: v for k, v in request_payload.items() if k not in self.keys_to_remove}
        return request_payload

    def _create_generate_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        payload = {"prompt": prompt, "images": images}
        api_url = f"{self.base_url}/v1/chat/completions" if len(self.base_url.split("/"))<3 else self.base_url
        yield from self._create_stream(
            payload=payload,
            stop=stop,
            api_url=api_url,
            **kwargs,
        )

    def _create_stream(
        self,
        api_url: str,
        payload: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop

        params = self._default_params

        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]

        if "options" in kwargs:
            params["options"] = kwargs["options"]
        else:
            params["options"] = {
                **params["options"],
                "stop": stop,
                **{k: v for k, v in kwargs.items() if k not in self._default_params},
            }
        response = requests.post(
            url=api_url,
            headers={
                "Content-Type": "application/json",
                **(self.headers if isinstance(self.headers, dict) else {}),
            },
            auth=self.auth,
            json=self._gen_requests_payload(payload, params),
            stream=True,
            timeout=self.timeout,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            if response.status_code == 404:
                raise LLMEndpointNotFoundError(
                    "API call failed with status code 404. "
                    "Maybe your model is not found "
                    f"Info: api_url: {api_url}"
                )
            else:
                raise ValueError(
                    f"LLM call failed with status code {response.status_code}."
                    f"\n\n\tDetails: {response.text}"
                )
        return self._extract_response(response)


    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from LLM stream.")

        return final_chunk

    async def _acreate_generate_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        raise NotImplementedError("Async stream not implemented for http request model.")
    async def _acreate_stream(
        self,
        api_url: str,
        payload: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        raise NotImplementedError("Async stream not implemented for http request model.")
    async def _astream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        raise NotImplementedError("Async stream not implemented for http request model.")

        
class AnswerAIProvide(BaseLLM, _HttpCommonModel):
    """locally runs large language models."""

    class Config:
        extra = "forbid"

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "Claude-AnswerAI"

    def _generate(  # type: ignore[override]
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to LLM's generate endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        """
        # TODO: add caching here.
        generations = []
        for prompt in prompts:
            final_chunk = super()._stream_with_aggregation(
                prompt,
                stop=stop,
                images=images,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)  # type: ignore[arg-type]

    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk
    async def _agenerate(  # type: ignore[override]
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        raise NotImplementedError(
            "Async generation is not supported for the HTTP request model. "
            "Please use the synchronous `generate` method instead."
        )
        
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        raise NotImplementedError(
            "Async generation is not supported for the HTTP request model. "
            "Please use the synchronous `generate` method instead."
        )