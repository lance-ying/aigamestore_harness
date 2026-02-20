# AI Gamestore Harness

A harness to run vision-language models (VLMs) on [AI Gamestore](https://aigamestore.com) games. The runner uses Playwright to drive a browser, captures game frames, sends them to a supported LLM provider for actions, and executes the returned key presses in the game. Results (screenshots, gameplay logs, GIFs) are saved under `results/`.

## Prerequisites

- **Python 3.10+**
- **Playwright** (Chromium): `playwright install chromium`
- **API keys** for at least one provider (see [Environment](#environment))
- Games served at a URL the harness can reach (e.g. local server or AI Gamestore)

## Installation

```bash
cd aigamestore_harness
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
playwright install chromium
```

## Environment

Create a `.env` file in the repo root (or set env vars) with the API key(s) for the provider(s) you use:

- **OpenAI:** `OPENAI_API_KEY`
- **Anthropic:** `ANTHROPIC_API_KEY`
- **Google:** `GOOGLE_API_KEY`
- **Together AI:** `TOGETHER_API_KEY`
- **xAI:** `XAI_API_KEY`

The script loads `.env` from the script directory automatically.

## Usage

Serve the games (e.g. at `http://localhost:8000`), then run the harness with **--model** and **--url** (full game URL).

```bash
python run_llm_eval.py --model openai:gpt-5 --url http://aigamestore.org/game1
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | Model in form `provider:model_name`, e.g. `openai:gpt-4o`, `anthropic:claude-sonnet-4-20250514`, `google:gemini-2.0-flash` |
| `--url` | *(required)* | Full game URL |
| `--max_seconds` | `120` | Max number of successful API calls per run (caps runtime) |
| `--headless` | off | Run the browser in headless mode |

## Supported providers and models

- **openai** – e.g. `gpt-4o`, `gpt-4o-mini`
- **anthropic** – e.g. `claude-sonnet-4-20250514`, `claude-3-5-sonnet-20241022`
- **google** / **gemini** – e.g. `gemini-2.0-flash`, `gemini-2.5-pro`
- **togetherai** / **together** – e.g. `meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo`
- **xai** – e.g. `grok-2-vision-1212`

Use the exact model name/ID expected by each provider.

## Output

Each run creates a directory under `results/` named like:

`YYYYmmdd_HHMMSS_<microseconds>_<model>_<game_name>/`

Contents include:

- **screenshots/** – Per-step screenshots
- **gameplay/** – Action frames, step JSON logs, and a `gameplay.gif`
- **prompts/** – Saved prompts (text + image metadata) sent to the model
- **content_history_*.json** – Full conversation history
- **gameplay_log_*.json** – Chronological gameplay log (scores, tokens, etc.)

## How it works

1. The script opens a Chromium page, navigates to the game URL, and locates the game canvas (`#defaultCanvas0` or similar).
2. It reads the on-page **game description** and **controls** (e.g. `#gameDescription`, `#gameControls`) and injects them into the prompt.
3. It captures a screenshot, pauses the game (ESC), and sends the last step’s result frames plus the current frame and a scratchpad to the LLM.
4. The model responds with `<keys>...</keys>` (five segments of 0.2s each) and optional `<scratchpad>...</scratchpad>`. The harness parses the keys, resumes the game, and executes each segment (instant and HOLD actions).
5. When the game signals an end state, the harness sends R + Enter to restart and continues until `--max_seconds` API calls are reached or the run is stopped.

## Project layout

- **vlm_fixed.py** – Main entrypoint (Playwright + LLM loop).
- **llm_interface/api/** – Provider-specific APIs (OpenAI, Anthropic, Google, Together, xAI).
- **utils/** – Parsing, gameplay helpers, logging, token/config utilities.
- **prompts/** – Markdown prompts (e.g. `prompt.md`).

## Troubleshooting

- **Canvas not found:** Ensure the game page exposes a canvas with id `defaultCanvas0` (or that the script’s fallback locators match your page).
- **Missing score / game end:** The script reads game state from the iframe when present (`document.querySelector('iframe').contentWindow.getGameState()`). If your game uses different globals, you may need to adapt the evaluation in `run_llm_eval.py`.
- **Rate limits:** Space out runs or reduce concurrency; use the provider’s recommended retries/backoff if you add retry logic.
- **API errors:** Check `.env` and that the model name matches the provider’s API (e.g. `google:gemini-2.0-flash`).

## License

See the repository license file.
