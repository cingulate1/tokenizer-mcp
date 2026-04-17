# tokenizer-mcp

`tokenizer-mcp` is a small MCP server that lets an LLM harness like Claude Code count the exact tokens in any text — a string or a whole file — without dropping into a shell, installing a tokenizer, or writing a throwaway script. You hand it the text and a model name; it routes to the right backend: Anthropic's `messages.count_tokens` for Claude, `tiktoken` for OpenAI, HuggingFace `transformers` for Qwen, and CLIP-L / CLIP-bigG for SDXL. "How many tokens is this?" becomes a single tool call.

## Installation

```bash
uv sync
```

## Running it directly

```bash
uv run server.py
```

At startup the server warms every tokenizer in a background thread, so the first real tool call pays no load cost.

## API key setup

Counting tokens for Claude requires an Anthropic API key, since the exact count comes from `messages.count_tokens`. Without a key, the Claude path silently falls back to tiktoken's `o200k_base` encoding. OpenAI, Qwen, and SDXL counts work offline and need no key.

Set the key as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

On Windows (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Or drop a `.env` file next to `server.py` and the server will load it on startup:

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_TOKEN_COUNT_MODEL=claude-3-haiku-20240307
```

`ANTHROPIC_TOKEN_COUNT_MODEL` sets the default Claude model when the caller omits one; it defaults to `claude-3-haiku-20240307`.

## What it exposes

Two tools, both returning an integer:

- `count_tokens(text, model="")` — count tokens in a string.
- `count_tokens_file(file_path, model="")` — count tokens in a UTF-8 file at an absolute path.

When `model` is omitted, the server uses `ANTHROPIC_TOKEN_COUNT_MODEL`.

## How model routing works

The `model` argument is normalized — lowercased, with spaces and underscores turned to hyphens — then matched against a short set of rules:

| What you pass                                           | Backend used                                       |
| ------------------------------------------------------- | -------------------------------------------------- |
| A raw tiktoken encoding (`o200k_base`, `cl100k_base`, `p50k_base`, `p50k_edit`, `r50k_base`, `gpt2`) | `tiktoken.get_encoding`                            |
| Anything starting with an OpenAI prefix (`gpt-5`, `gpt-4`, `gpt-3.5`, `gpt-3`, `chatgpt`, `o1`, `o3`, `o4`, `text-davinci`, `text-embedding`, `code-davinci`, `text-curie`, `text-babbage`, `text-ada`) | `tiktoken.encoding_for_model`                      |
| `sdxl`, `sd-xl`, or `stable-diffusion-xl`               | CLIP-L + CLIP-bigG, taking the max of the two (SDXL concatenates both) |
| Anything starting with `qwen` (or a HuggingFace repo path containing `/`) | `transformers.AutoTokenizer`                       |
| Everything else                                         | Anthropic `messages.count_tokens`                  |

Qwen shorthands (`qwen`, `qwen1`, `qwen1.5`, `qwen2`, `qwen2.5`) resolve to the smallest repo in each generation (e.g. `Qwen/Qwen2.5-0.5B`) so the first download is quick. Models within a generation share a tokenizer, so the count is identical whichever you pick.


## Wiring it into an MCP client

Add an `mcpServers` entry pointing at this directory:

```json
{
  "mcpServers": {
    "tokenizer": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/tokenizer-mcp", "run", "server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

The `env` block is optional; omit it and Claude counts fall back to `o200k_base`, as above.