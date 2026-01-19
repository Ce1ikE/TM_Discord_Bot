import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    BitsAndBytesConfig, 
    TextStreamer,
)
from transformers import (
    TextIteratorStreamer,
    StoppingCriteriaList,
    GenerationConfig 
)
from typing import Tuple, Literal

import time
import datetime as dt
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

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
            "ibm-granite/granite-4.0-h-350M", # 350M parameters
            "ibm-granite/granite-4.0-h-1b",   # 1B parameters
            "ibm-granite/granite-4.0-h-micro", # 3B parameters
            "ibm-granite/granite-4.0-h-tiny", # 7B parameters
            "ibm-granite/granite-4.0-h-small", # 32B parameters
            "Qwen/Qwen2.5-1.5B",  # 1.54B parameters
            "Qwen/Qwen2.5-1.5B-Instruct",  # 1.54B parameters
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF", # LLama only
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "microsoft/Phi-3.5-mini-instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # 1.1B parameters
        ] = "ibm-granite/granite-4.0-h-350M",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self.llm = None

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                quantization_config=quantization_config,
                device_map="auto",
                # flash attention for faster inference can be enabled if supported
                # problem is that not all models support it yet and is not very well compatible with Windows only on linux
                # if running on a EC2 instance with an A100 GPU this should work fine
                # attn_implementation="flash_attention_2"
            )
        except Exception as e:
            logger.warning(f"Failed to load model with quantization: {e}, loading without quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                # attn_implementation="flash_attention_2"
            ).to(device)
        
        # by setting eval mode, we disable gradient calculations, dropout, etc.
        # which are not needed for inference and save memory and computation
        self.model.eval()
        logger.info(f"Generator model initialized on {self.model.device}")

    def generate(
        self,
        context: str, 
        prompt: str, 
        stopping_criteria: StoppingCriteriaList | None = None,
        generation_config: GenerationConfig | None = None,
        streamer: TextIteratorStreamer | None = None
    ) -> str:
        """
            Generate answer from a prompt
            Args:
                context: str: context given to the model to give a role to the llm (e.g., "You are a helpful assistant.")
                prompt: str: the question prompt (e.g., "What is the capital of France?")
                streamer: TextIteratorStreamer | None: optional streamer for real-time generation
            Returns:
                answer: str: the generated answer
        """

        full_prompt = (
            f"{context} \n\n"
            f"[Vraag]: {prompt} \n\n"
            f"[Antwoord]: "
        )
        # tokenize the prompt such that the model can process it (raw text -> token IDs) 
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            # https://huggingface.co/docs/transformers/generation_strategies
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                max_new_tokens=self.max_new_tokens,
                # KV caching for faster generation 
                # !! limit user input size so that memory does not explode !!
                use_cache=True, 
                num_beams=2 if streamer is None else 1,
                # if using a streamer, we need to set do_sample to False
                do_sample=streamer is None,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
            )

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # output_text[0] == prompt + generated text
        answer = output_text[0].split("[Antwoord]:")[-1].strip()
        return answer
