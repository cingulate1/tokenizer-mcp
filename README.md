# tokenizer-mcp

An MCP server that exposes a single, model-aware token counter. Give it any string (or a file path) plus a model name and it routes to the correct tokenizer:

| Model family                           | Backend                                          |
| -------------------------------------- | ------------------------------------------------ |
| Claude (`claude-*`, anything else)     | Anthropic `messages.count_tokens` API            |
| OpenAI (`gpt-*`, `o1`/`o3`/`o4`, etc.) | `tiktoken` via `encoding_for_model`              |
| Raw tiktoken encodings (`o200k_base`…) | `tiktoken.get_encoding`                          |
| Qwen (`qwen`, `qwen2.5`, `Qwen/…`)     | `transformers` tokenizer                         |
| SDXL (`sdxl`, `stable-diffusion-xl`)   | CLIP-L + CLIP-bigG (max of both, SDXL-accurate)  |

## Tools

- `count_tokens(text, model="")` — count tokens in a string.
- `count_tokens_file(file_path, model="")` — count tokens in a UTF-8 file.

When `model` is omitted, the server uses `ANTHROPIC_TOKEN_COUNT_MODEL` (default `claude-3-haiku-20240307`).

## Anthropic API key

The Anthropic backend calls `messages.count_tokens` and therefore needs an API key. Set it as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

On Windows (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Or drop a `.env` file next to `server.py`:

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_TOKEN_COUNT_MODEL=claude-3-haiku-20240307
```

### Fallback without a key

If `ANTHROPIC_API_KEY` is not set, requests that would route to Anthropic transparently fall back to tiktoken's `o200k_base` encoding. This is the encoding used by modern OpenAI chat models and is a reasonable approximation for Claude token budgets — usually within a few percent for English prose. Exact counts for Claude still require the API key.

## Installation

```bash
uv sync
```

## Running

```bash
uv run server.py
```

## MCP client configuration

Example `mcpServers` entry:

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

The `env` block is optional — omit it to use the `o200k_base` fallback.
