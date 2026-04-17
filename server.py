import logging
import os
from functools import lru_cache
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)


def _load_dotenv(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
            value = value[1:-1]
        os.environ[key] = value


# Load a .env sitting next to server.py, if present.
_load_dotenv(Path(__file__).with_name(".env"))

mcp = FastMCP("tokenizer")

# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

TIKTOKEN_ENCODINGS = {
    "o200k_base", "cl100k_base", "p50k_base", "p50k_edit", "r50k_base", "gpt2",
}

OPENAI_PREFIXES = (
    "gpt-5", "gpt-4", "gpt-3.5", "gpt-3", "chatgpt",
    "o1", "o3", "o4",
    "text-davinci", "text-embedding", "code-davinci",
    "text-curie", "text-babbage", "text-ada",
)

SDXL_TRIGGERS = {"sdxl", "sd-xl", "stable-diffusion-xl"}

# All models within a Qwen generation share the same tokenizer,
# so we map shorthands to the smallest repo for fast downloads.
QWEN_REPOS = {
    "qwen":     "Qwen/Qwen-7B",
    "qwen1":    "Qwen/Qwen-7B",
    "qwen1.5":  "Qwen/Qwen1.5-0.5B",
    "qwen2":    "Qwen/Qwen2-0.5B",
    "qwen2.5":  "Qwen/Qwen2.5-0.5B",
}

# ---------------------------------------------------------------------------
# Cached tokenizer loaders
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _get_tiktoken_enc(name: str):
    import tiktoken
    if name in TIKTOKEN_ENCODINGS:
        return tiktoken.get_encoding(name)
    return tiktoken.encoding_for_model(name)


@lru_cache(maxsize=4)
def _get_qwen_tokenizer(repo: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo, trust_remote_code=True)


@lru_cache(maxsize=1)
def _get_sdxl_tokenizers():
    from transformers import CLIPTokenizer
    tok_l = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tok_g = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    return tok_l, tok_g


@lru_cache(maxsize=1)
def _get_anthropic_client():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=api_key)

# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _count_tiktoken(text: str, model_or_enc: str) -> int:
    return len(_get_tiktoken_enc(model_or_enc).encode(text))


def _count_sdxl(text: str) -> int:
    tok_l, tok_g = _get_sdxl_tokenizers()
    return max(len(tok_l.encode(text)), len(tok_g.encode(text)))


def _count_qwen(text: str, model: str) -> int:
    model_lower = model.lower()
    if model_lower in QWEN_REPOS:
        repo = QWEN_REPOS[model_lower]
    elif "/" in model:
        repo = model
    else:
        repo = f"Qwen/{model}"
    return len(_get_qwen_tokenizer(repo).encode(text))


def _count_anthropic(text: str, model: str) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        # No API key available — fall back to tiktoken's o200k_base, which is
        # a close-enough approximation for modern chat-model token budgets.
        return _count_tiktoken(text, "o200k_base")
    response = _get_anthropic_client().messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

def _route(model: str) -> str:
    m = model.lower().replace(" ", "-").replace("_", "-")
    if m in TIKTOKEN_ENCODINGS:
        return "tiktoken"
    if any(m.startswith(p) for p in OPENAI_PREFIXES):
        return "openai"
    if m in SDXL_TRIGGERS:
        return "sdxl"
    if m.startswith("qwen"):
        return "qwen"
    return "anthropic"


def _do_count(text: str, model: str) -> int:
    if not text:
        return 0

    if not model:
        model = os.environ.get("ANTHROPIC_TOKEN_COUNT_MODEL", "claude-3-haiku-20240307").strip()

    backend = _route(model)

    if backend == "tiktoken":
        return _count_tiktoken(text, model.lower())
    if backend == "openai":
        return _count_tiktoken(text, model)
    if backend == "sdxl":
        return _count_sdxl(text)
    if backend == "qwen":
        return _count_qwen(text, model)
    return _count_anthropic(text, model)


@mcp.tool()
def count_tokens(text: str, model: str = "") -> int:
    """Count tokens in text using the Anthropic token counting API.

    Args:
        text: The text to count tokens for.
        model: Model to use for tokenization. Defaults to ANTHROPIC_TOKEN_COUNT_MODEL from .env.
    """
    return _do_count(text, model)


@mcp.tool()
def count_tokens_file(file_path: str, model: str = "") -> int:
    """Count tokens in a file using the Anthropic token counting API.

    Args:
        file_path: Absolute path to the file to count tokens for.
        model: Model to use for tokenization. Defaults to ANTHROPIC_TOKEN_COUNT_MODEL from .env.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = p.read_text(encoding="utf-8")
    return _do_count(text, model)


def _warmup():
    """Pre-load all tokenizers at startup so first tool calls are instant."""
    import threading

    def _load():
        logging.info("Warming up tokenizers...")

        # Tiktoken encodings
        for enc in TIKTOKEN_ENCODINGS:
            try:
                _get_tiktoken_enc(enc)
                logging.info(f"  tiktoken/{enc} loaded")
            except Exception as e:
                logging.warning(f"  tiktoken/{enc} failed: {e}")

        # SDXL CLIP tokenizers
        try:
            _get_sdxl_tokenizers()
            logging.info("  sdxl loaded")
        except Exception as e:
            logging.warning(f"  sdxl failed: {e}")

        # Qwen tokenizers (deduplicate repos)
        for repo in dict.fromkeys(QWEN_REPOS.values()):
            try:
                _get_qwen_tokenizer(repo)
                logging.info(f"  qwen/{repo} loaded")
            except Exception as e:
                logging.warning(f"  qwen/{repo} failed: {e}")

        # Anthropic client (optional — falls back to o200k_base without a key)
        if os.environ.get("ANTHROPIC_API_KEY", "").strip():
            try:
                _get_anthropic_client()
                logging.info("  anthropic client loaded")
            except Exception as e:
                logging.warning(f"  anthropic client failed: {e}")
        else:
            logging.info("  anthropic client skipped (no ANTHROPIC_API_KEY; using o200k_base fallback)")

        logging.info("Warmup complete.")

    threading.Thread(target=_load, daemon=True).start()


def main():
    _warmup()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
