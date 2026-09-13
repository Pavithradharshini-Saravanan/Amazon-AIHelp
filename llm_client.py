import os
import json
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

CACHE_FILE = Path("data/.llm_cache.json")


def _load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _get_cache_key(prompt: str, system_prompt: str = None, model: str = None) -> str:
    raw = f"{model}:{system_prompt}:{prompt}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# Initialize Groq client
groq_client = None
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    from groq import Groq
    groq_client = Groq(api_key=groq_key)

# Initialize Gemini client
gemini_client = None
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if gemini_key:
    import google.generativeai as genai
    genai.configure(api_key=gemini_key)
    gemini_client = genai

# Initialize OpenAI client
openai_client = None
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    from openai import OpenAI
    openai_client = OpenAI(api_key=openai_key)


def generate_text(prompt: str, system_prompt: str = None, temperature: float = 0.0, max_tokens: int = 150, provider: str = None) -> str:
    """
    Generates text using live LLM provider API calls (Groq, Gemini, OpenAI).
    Raises an Exception if no API key is configured or if all LLM API retries fail.
    NO hardcoded fallback functions exist in this codebase.
    """
    cache = _load_cache()
    cache_key = _get_cache_key(prompt, system_prompt, provider or "default")

    if cache_key in cache:
        return cache[cache_key]

    max_retries = 8
    backoff = 5
    last_exception = None

    for attempt in range(max_retries):
        try:
            res_text = None

            # Provider 1: Groq
            if (provider == "groq" or provider is None) and groq_client:
                model_name = os.getenv("GROQ_MODEL", "groq/compound-mini")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                res = groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                res_text = res.choices[0].message.content.strip()

            # Provider 2: Gemini
            elif (provider == "gemini" or provider is None) and gemini_client:
                model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
                if hasattr(gemini_client, "GenerativeModel"):
                    g_model = gemini_client.GenerativeModel(model_name)
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    res = g_model.generate_content(full_prompt)
                    res_text = res.text.strip()

            # Provider 3: OpenAI
            elif (provider == "openai" or provider is None) and openai_client:
                model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                res = openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                res_text = res.choices[0].message.content.strip()

            if res_text is not None:
                res_text = res_text.encode("ascii", "ignore").decode("ascii")
                cache[cache_key] = res_text
                _save_cache(cache)
                time.sleep(2.0)  # Rate limit safety delay
                return res_text

            raise RuntimeError("No LLM API client available. Please set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.")

        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str:
                print(f"[LLM Client] Groq Rate limit hit (attempt {attempt+1}/{max_retries}). Waiting {backoff}s for quota window reset...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                print(f"[LLM Client] API Error on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2)

    raise RuntimeError(f"Live LLM API call failed after retries: {last_exception}")


if __name__ == "__main__":
    test_prompt = "Classify this customer message into exactly one intent from ['delivery_issue', 'refund_return', 'other']: Customer Message: 'where is my package'"
    res = generate_text(test_prompt)
    print("Live LLM Output:", res)
