import os

import torch
from typing import Tuple, Literal, Generator
from llama_cpp import Llama, ChatCompletionRequestResponseFormat
from pydantic import BaseModel, Field
import time
import datetime as dt
import logging
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

MODELS_PATH = Path(os.path.join(os.getcwd(), "models"))
if not MODELS_PATH.exists():
    MODELS_PATH.mkdir(parents=True, exist_ok=True)


class LLMGenerator:
    """
    Deterministic RAG-style answer synthesis.
    """

    def __init__(
        self, 
        # NOTE: for any model ID not listed here there is a HuggingFace model card available
        # in this you can find a config.json which contains usefull parameters such as:
        # max_position_embeddings, vocab_size, hidden_size, num_attention_heads, etc... 
        model_name: Literal[
            "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf",
            "Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf",
            "microsoft/phi-4-gguf:phi-4-Q4_0.gguf"
        ] = "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf",
        max_new_tokens: int = 512,
        max_context_tokens: int = 8192,
        system_prompt: str = "You are a helpful assistant for students of the Thomas More Campus De Nayer. Use the provided context to answer the question. If you don't know the answer, say you don't know. Always use all the relevant information from the context to provide a complete and accurate answer."
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.model = None
        self.system_prompt = system_prompt

        try:
            repo_id = model_name.split(":")[0]
            filename = model_name.split(":")[1]
            self.model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                flash_attn=True,
                n_ctx=self.max_context_tokens,
                n_gpu_layers=-1,
                n_batch=512,
                max_tokens=self.max_new_tokens,
                seed=42,
                cache_dir=MODELS_PATH
            )
        except Exception as e:
            raise ValueError(f"Error loading model {model_name}")

        # by setting eval mode, we disable gradient calculations, dropout, etc.
        # which are not needed for inference and save memory and computation
        self.model.eval()

    def generate(
        self,
        context: str, 
        prompt: str, 
        response_format: ChatCompletionRequestResponseFormat = {"type": "text"},
        stream: bool = True,
    ) -> Generator[str, None, None] | str:
        """
            Generate answer from a prompt
            Args:
                context: str: extra information to condition the answer generation (e.g., retrieved documents, instructions, etc.)
                prompt: str: the question prompt (e.g., "What is the capital of France?")
                response_format: ChatCompletionRequestResponseFormat: specify the desired output format (e.g., text, JSON, etc.)
                stream: bool: whether to return the answer as a stream of tokens or as a single string after generation is complete
                buffer_object: str: a object in which the response will be stored for outside access from a other thread (e.g., the Discord bot's main thread)
            Returns:
                answer: str: the generated answer
        """

        full_prompt = (
            f"{context} \n\n"
            f"[Vraag]: {prompt} \n\n"
            f"[Antwoord]: "
        )
        
        if len(self.model.tokenize(full_prompt)) > self.max_context_tokens:
            raise ValueError(f"Input prompt is too long for the model's context window of {self.max_context_tokens} tokens.")

        with torch.no_grad():
            # https://huggingface.co/docs/transformers/generation_strategies
            output = self.model.create_chat_completion(
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": full_prompt}],
                max_tokens=self.max_new_tokens,
                temperature=0.0,  # deterministic output
                top_p=1.0,        # no nucleus sampling
                frequency_penalty=0.0,  # no penalty for repeating tokens
                presence_penalty=0.0,   # no penalty for introducing new tokens
                stream=stream,  # enable streaming output
                response_format=response_format # we can use this to enforce a structured output format (e.g., JSON) if needed
            )

        
        if stream:
            for chunk in output:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
        else:
            return output["choices"][0]["message"]["content"]
