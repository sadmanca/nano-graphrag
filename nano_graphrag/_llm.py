import numpy as np
import os

# Gemini imports
try:
    import google.genai as genai
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

global_openai_async_client = None
global_azure_openai_async_client = None
global_gemini_client = None


def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client


def get_gemini_client_instance():
    global global_gemini_client
    if global_gemini_client is None:
        if not _HAS_GEMINI:
            raise ImportError("google-genai is required for Gemini support. Install it with: pip install google-genai")
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required for Gemini")
        
        global_gemini_client = genai.Client(api_key=api_key)
    return global_gemini_client


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]
    # Flatten to plain text for Gemini
    flat = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            flat.append(c)
        else:
            flat.append(str(c))
    full_prompt = "\n\n".join(flat)
    generative_model = gemini.GenerativeModel(model)
    response = generative_model.generate_content(full_prompt)
    if hasattr(response, "text") and response.text:
        text = response.text
    elif hasattr(response, "candidates") and response.candidates:
        # best-effort extraction
        first = response.candidates[0]
        try:
            text = first.content.parts[0].text
        except Exception:  # noqa: E722
            text = str(first)
    else:
        text = ""
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": text, "model": model}})
        await hashing_kv.index_done_callback()
    return text


async def gpt_4o_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    # Retained legacy function name for backward compatibility; now Gemini only
    # Use flash-lite for speed or flash for complex tasks
    model_name = os.getenv("GEMINI_MAIN_MODEL", "models/gemini-2.5-flash")  # Flash is faster than flash-lite for many tasks
    return await _gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    # Retained legacy function name for backward compatibility; now Gemini only
    # Use flash-lite for cheap/fast operations
    model_name = os.getenv("GEMINI_CHEAP_MODEL", os.getenv("GEMINI_MAIN_MODEL", "models/gemini-2.5-flash-lite"))
    return await _gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


# Gemini LLM and Embedding Functions

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def gemini_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    gemini_client = get_gemini_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    
    # Prepare the prompt for Gemini
    full_prompt = ""
    if system_prompt:
        full_prompt += f"System: {system_prompt}\n\n"
    
    # Add history messages
    for msg in history_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        full_prompt += f"{role.capitalize()}: {content}\n\n"
    
    full_prompt += f"User: {prompt}"
    
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, full_prompt)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Remove unsupported kwargs for Gemini
    gemini_kwargs = {k: v for k, v in kwargs.items() if k in ['temperature', 'max_tokens']}
    
    response = await gemini_client.aio.models.generate_content(
        model=model,
        contents=full_prompt,
        **gemini_kwargs
    )
    
    result = response.text.strip()
    
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": result,
                    "model": model,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return result


async def gemini_2_5_flash_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await gemini_complete_if_cache(
        "gemini-2.5-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gemini_1_5_pro_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await gemini_complete_if_cache(
        "gemini-1.5-pro",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=3072, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def gemini_embedding(texts: list[str]) -> np.ndarray:
    gemini_client = get_gemini_client_instance()
    
    # Use Gemini text embedding model - batch process all texts at once
    response = await gemini_client.aio.models.embed_content(
        model="gemini-embedding-001",  # Use correct model name
        contents=texts  # Pass all texts at once for efficiency
    )
    
    # Extract embedding values from response
    embeddings = []
    for embedding_obj in response.embeddings:
        embeddings.append(embedding_obj.values)
    
    return np.array(embeddings)
