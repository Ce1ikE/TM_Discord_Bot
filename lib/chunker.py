import datetime as dt
import ftfy
import hashlib

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import docling
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from typing import List, Tuple

from .embedder import Embedder


# chunking is quite a complex topic viewed from a practical perspective
# there are various strategies and techniques to chunk documents
# - fixed size (e.g., 512 tokens)
# - fixed size with overlap (e.g., 512 tokens with 50 tokens overlap)
# - recursive chunking (e.g., split by sections, paragraphs, sentences, etc.)
# - semantic chunking (e.g., using embeddings to find semantically coherent chunks)
# - hybrid chunking (e.g., combination of fixed size and semantic chunking)
# - document structure-based chunking (e.g., based on headings, subheadings, etc.)
# - adaptive chunking (e.g., based on content density, information density, etc.)
# 
# the problem is that if a API or any DataStore can load any document format (pdf, docx, txt, etc.)
# we first need to convert the document into a uniform representation , lucky for us, docling does that.
# however a uniform representation alone is not enough, the "garbage in, garbage out" principle still applies
# we need to do some preprocessing/cleaning of the document before chunking , but what kind of cleaning?
# e.g., removing unwanted sections (headers, footers, page numbers, etc.), normalizing text (lowercasing, removing special characters, etc.), etc.
# these techniques seem logical and straightforward in NLP tasks, but a embedding model is trained on natural text not bags of words
# so we need to be careful not to remove too much information that the embedding model needs to generate meaningful embeddings.
# Thus we need evualtaion and testing to find the right balance between cleaning and preserving information.
# once we have a clean document, we can chunk it into smaller pieces.
# but then we need to decide how to chunk the document. 
# The reason for chunking documents is that most LLM/SLM's have a maximum token limit (e.g., 512, 768, 1024, etc.)
# a token is just a piece of text (word, subword, character, etc.) that a LLM/SLM can understand/process
# if we exceed the token limit, we need to split the document into smaller chunks
# so first off every chunking strategy needs to be "token aware" meaning that it needs to take into account the tokenization scheme of the LLM/SLM
# here agian we have some problems that might arise into the future when swapping/switching to other embedding models/LLM's (but that is out of scope for now)
#
# also bigger chunks means more computation/memory usage during embedding generation
# which can lead to higher costs and latency during both ingestion and inference.
#
# we also need to consider the semantic coherence of the chunks 
# e.g., we don't want to split a sentence or paragraph into multiple chunks
# which can lead to loss of context and meaning. a technique we can use here is semantic chunking
# making use of cosine similarity of embeddings to find semantically coherent chunks
#
# There is also a other problem regarding hardware during inference. 
# When a tokenizer is used to tokenize the user query and chunks (query + retrieved chunks -> [<|sos|> query <|sep|> chunk1 <|sep|> chunk2 ... <|eos|>])
# for a LLM , the LLM has a maximum context window (e.g., 2048, 4096, 8192 tokens, etc.)
# this context window has been increasing over the last couple of years with newer models
# but most often we hit a other limitation first before exceeding the context window of the LLM
# which is the GPU memory limitation.
# the compute required to process a query and the retrieved chunks grows depending on a lot of factors
# which can increase the memory usage and latency during inference.
# on a 6GB GPU (e.g.: laptop NVIDIA RTX 4060) you can only process a limited amount of tokens
# ~500-1000 tokens depending on various factors which is not a lot when you consider that a average document chunk can be 512-768 tokens
# that means during retrieval we can only retrieve a few chunks (e.g., 2-4 chunks) before exceeding the context window of the LLM during inference.
# and having a OOM (out of memory) error on a GPU is not a good user experience nor for the service reliability.
# so chunks need to be small enough to fit into the context window of the LLM during inference and also need to be small enough to fit into the GPU memory during inference.
# thus chunking strategy needs to take this into account as well.
# in this blog post:
# https://mwzero.medium.com/running-llms-locally-how-much-vram-do-you-actually-need-15a0fae64c53
# it is explained that alongside the model's different factors we also have to take into account:
# - temporary activations (batch size x sequence length x hidden size)
# - attention caches (key and value tensors for each layer (KV cache))
# - I/O buffers
# - framework overhead (e.g., PyTorch, TensorFlow, etc.)
# and they estimate a 20% overhead on top of the model size as a good rule of thumb to estimate the total GPU memory usage during inference.
# 
# VRAM = 1.2 * (params * quantization_factor (bits)) / 8 (bits to bytes) / 1024^3 (bytes to GB)
# 
# but in reality this "overhead" increases way more and quantization is only applied to linear layers
# during inference what really dominates the VRAM usage is the attention mechanism.
# the general formula of attention is:
# 
# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
#  
# where,
# - Q = Query matrix 
#       (multiplied by the input embeddings later and learned during training)
# - K = Key matrix 
#       (multiplied by the input embeddings later and learned during training)
# - QK^T = Attention scores matrix 
#          (from this we get the weights for each token)
# - V = Value matrix 
#       (the actual token representations that get weighted and summed later to change the input embeddings 
#       think of this as changing the meaning of the embeddings based on context (surrounding tokens))
# - sqrt(d_k) = scaling factor (dimension of the key vectors mostly to stabelize gradients during training)
#
# because every token needs to be compared to every other token in a sequence (sentence, paragraph, document, etc.) 
# the compute to process attention grows quadratically with the sequence length. this is a O(N^2) memory complexity.
# techniques like Flash Attention try to optimize this process by reducing memory access patterns and improving cache efficiency
# https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention
# one of the most impactful ways to reduce KV cache requirements is by modifying the Transformer’s attention mechanism itself
# these architectural innovations primarily aim to decrease the number of distinct Key (K) and Value (V) projections 
# that need to be stored and subsequently loaded from memory:
# - MHA (Multi-Head Attention) (baseline) this is your typical attention mechanism where each attention head has its own Q K V projections 
# - GQA (Group-Query Attention) here attention heads are grouped together and each group shares the same K V projections
# - MQA (Multi-Query Attention) here all attention heads share the same K V projections, only Q projections are unique per head
# - MLA (Multi-Head Latent Attention) here the idea is to compress the K V projections into a smaller set of latent representations and thus reducing the overall size of the KV cache
# https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/
# so mostly you can run a big model or a long context window, but not both at the same time.
# so a better estimation would be:
# 
# VRAM = Constant + (Linear(N) | Quadratic(N))
# 
# where, 
# - Constant = Model Size + I/O Buffers + Framework Overhead
# - Linear(N) = Attention KV Cache (proportional to sequence length N)
# - Quadratic(N) = Attention Scores (proportional to sequence length N squared)
#
# VRAM(N) [GB] = (ModelWeightsGB + FrameworkOverheadGB + CUDAWorkspaceGB) 
#                + N * (2 * L * d/g * b_kv / 1024³)
#
# - 1024³ = bytes to GB conversion 
#
# where,
# - ModelWeightsGB = (number_of_parameters × quantization_factor)  / 8 / 1024³
#   - number_of_parameters = total parameters in the model in billions
#   - quantization_factor = bits per parameter (e.g., 16 for fp16, 8 for int8, etc.)
# - FrameworkOverheadGB = estimated overhead from the deep learning framework (e.g., PyTorch, TensorFlow)
# - CUDAWorkspaceGB = estimated overhead for CUDA workspace memory
# - N = sequence length (number of tokens in prompt + generated tokens)
# - L = number of transformer layers
# - d = hidden size (dimension of model embeddings)
# - g = GQA (group query attention) group factor (query heads / KV heads usually 1 or 2)
# - b_kv = bytes per KV cache element (2 bytes for fp16, 1 byte for int8, etc.)
class Chunker:
    """
        Token-aware chunker with cleaning, keyword, and entity extraction.
    """

    def __init__(
        self,
        embedding_model: Embedder,
        max_chunk_length: int = 200,
        max_tokens: int = None, 
    ) -> None:
        self.tokenizer_name = ""
        self.max_tokens = max_tokens
        self.max_chunk_length = max_chunk_length

        self.tokenizer_name = embedding_model.model_name
        hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        # the HybridChunker implementation uses 
        # a hybrid approach, applying tokenization-aware refinements
        # https://docling-project.github.io/docling/concepts/chunking/#hybrid-chunker
        # it uses a hierarchical chunker based on the tokenizer to do:
        # 1) one pass where it splits chunks only when needed (oversized chunks with respect to max_tokens)
        # 2) another pass where it merges chunks only when possible (undersized successive chunks with respect to max_tokens)
        if self.max_tokens:
            tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=self.max_tokens,
            )
            self.chunker = HybridChunker(
                tokenizer=tokenizer,
                max_tokens=self.max_tokens,
                merge_peers=True,
            )
        else:
            tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
            )
            self.chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=True,
            )
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning using ftfy."""
        # fix text encoding issues
        cleaned_text = ftfy.fix_text(text)
        # further cleaning steps can be added here
        return cleaned_text

    def _chunk(self, text: str) -> List[str]:
        chunks = []
        current = ""
        for paragraph in text.split("\n"):
            if len(current) + len(paragraph) > self.max_chunk_length:
                chunks.append(current.strip())
                current = paragraph
            else:
                current += "\n" + paragraph
        if current:
            chunks.append(current.strip())
        return chunks

    def process(
        self,
        texts: List[str],
    ) -> Tuple[List[str], dict]:
        """Chunk a Docling document, enrich metadata, and collect timings."""
        
        evaluation = {
            "start_time": dt.datetime.now(),
            "end_time": None,
            "total_time": None,
            "avg_cleaning_time": 0,
            "total_texts": 0,
            "total_chunks": 0,
            # applying summaries might provide some speed ups 
            # during retrieval time as less tokens need to be processed
            # but for now no summarization step will be implemented
        }

        cleaning_time = []
        chunks: List[str] = []

        for idx, text in enumerate(texts):
            evaluation["total_texts"] += 1
            # we can do some preprocessing/cleaning here
            # e.g., removing unwanted sections, normalizing text, etc.
            # https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-enrichment-phase
            start_cleaning = dt.datetime.now()
            cleaned_text = self._clean_text(text)
            end_cleaning = dt.datetime.now()
            cleaning_time.append((end_cleaning - start_cleaning).total_seconds())
            
            chunks.append(cleaned_text)

        evaluation["end_time"] = dt.datetime.now()
        evaluation["total_time"] = (evaluation["end_time"] - evaluation["start_time"]).total_seconds()
        evaluation["avg_cleaning_time"] = sum(cleaning_time) / max(len(cleaning_time), 1)
        evaluation["total_chunks"] = len(chunks)

        return chunks, evaluation