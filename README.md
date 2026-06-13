# minimal-agent

A tiny, from-scratch AI agent you can read top to bottom. No frameworks — just
the official `openai` Python SDK and a native tool-use API, pointed at a **local
Ollama model so it runs completely free, with no API key**. It exists to make
the **agent loop** obvious: ~150 lines of heavily-commented Python in `agent.py`,
three tools in `tools.py`, and a deliberately messy CSV to reason over.

## The agent loop, in one paragraph

An agent is a loop around a single API call. You send the conversation plus a
list of tool definitions to the model. The model replies; you look at what it
did. If it **asked to call tools** (`finish_reason == "tool_calls"`), you run
those tools, append the results to the conversation, and call the API again so
the model can read the results and keep going. If it **just answered**, it's
done and you print it. Repeat until done (or until a safety limit). The API is
stateless — *you* hold the conversation history and resend it every turn. That
send-message → run-tool → feed-result-back cycle is the whole thing.

## Setup

```bash
cd minimal-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Backend: local Ollama (free, no key)

1. Install [Ollama](https://ollama.com) and start it (it serves an
   OpenAI-compatible API on `http://localhost:11434`).
2. Pull a **tool-capable** model:
   ```bash
   ollama pull qwen2.5
   ```
3. That's it — `agent.py` defaults to `http://localhost:11434/v1` and
   `qwen2.5:latest`, so no `.env` is needed. To override (different model or a
   different OpenAI-compatible server like LM Studio), copy `.env.example` to
   `.env` and set `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`.

> **Tool use needs a tool-capable model.** Reasoning-only models such as
> `deepseek-r1` do **not** support tool calling — Ollama rejects them with
> `does not support tools`. Use `llama3.2`, `qwen2.5`, `llama3.1`,
> `mistral-nemo`, etc.

## Run

```bash
python agent.py "Are there any data quality issues in sample_data/sales.csv? Show me examples."
```

Watch the labeled output: `[MODEL]` (what it said) → `[TOOL CALL]` (which tool,
with what args) → `[TOOL RESULT]` → next iteration, until `done`.

> **Model quality note.** The default `qwen2.5` (7B) has strong tool use and
> reasoning. A smaller model like `llama3.2` (3B) drives the loop correctly too
> but its *analysis* is rougher — set `OPENAI_MODEL=llama3.2:latest` if you want
> something lighter. The loop is identical either way.

## Sample data

`sample_data/sales.csv` is **intentionally messy** so you can see the agent cope
with real-world data. Issue categories baked in: mixed date formats; inconsistent
category casing and typos (`Electronics`/`electronics`/`ELECTRONICS`/`Electonics`,
and the same for `Furniture`); leading/trailing whitespace in some names; blank
`quantity`/`unit_price` cells; rows where `total` ≠ `quantity × unit_price`;
duplicate `order_id` rows; negative quantities; and stray-character numerics
(`$45.99`, `1,250`).

## Example prompts to try

- `python agent.py "How many unique product categories are in sample_data/sales.csv?"`
  (trips on casing/typos — naive `COUNT(DISTINCT)` overcounts)
- `python agent.py "What's the total revenue?"`
  (trips on nulls, the bad totals, and the `$`/comma stray characters)
- `python agent.py "Are there any data quality issues in this file? Show examples."`
- `python agent.py "Which customers spent the most? Watch out for messy names."`
- `python agent.py "List the files in sample_data, then show me the first 5 rows of the CSV."`

## Tests

A stdlib-only smoke test exercises the tools against the sample CSV (no model,
no network, no API key):

```bash
python test_tools.py
```

## Where to go next (exercises)

1. **Streaming.** Pass `stream=True` to `client.chat.completions.create(...)` and
   print the text deltas as they arrive, so you watch the model think in real time.
2. **Conversation memory across runs.** Persist `messages` to a JSON file at the
   end of `run_agent` and reload it at the start, so a second `python agent.py`
   call continues the previous conversation.
3. **A data-cleaning tool.** Add a fourth tool (e.g. `clean_csv`) that normalizes
   category casing/typos and writes a cleaned copy — then ask the agent to clean
   the data before analyzing it.

## Note on tool-use formats

This agent uses the OpenAI Chat Completions tool-calling format. Anthropic's
Claude uses a slightly different native format (a `system=` argument,
`input_schema` instead of a `function` wrapper, `stop_reason == "tool_use"`, and
`tool_result` blocks). The `# vs Anthropic` comments in `agent.py` point out each
difference, in case Claude is your next stop.

## License

MIT — see [LICENSE](LICENSE).
