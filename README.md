# minimal-agent

A tiny, from-scratch AI agent you can read top to bottom. No frameworks — just
an official LLM SDK and a native tool-use API. It exists to make the **agent
loop** obvious: ~150 lines of heavily-commented Python, three tools in
`tools.py`, and a deliberately messy CSV to reason over.

There are two copies of the *same loop* so you can compare wire formats:

| File | SDK / format | Backends | Notes |
|------|--------------|----------|-------|
| `agent.py` | `anthropic` (Claude native tool use) | real Claude API, LM Studio, any Anthropic-compatible gateway | the reference implementation |
| `agent_openai.py` | `openai` (Chat Completions tool calls) | **local Ollama (free, no key)**, or any OpenAI-compatible server | runs free today |

`diff agent.py agent_openai.py` to see exactly how the two providers' tool-use
formats differ — the loop is identical; only the schema shape and message
plumbing change. Both reuse the same `tools.py`.

## The agent loop, in one paragraph

An agent is a loop around a single API call. You send the conversation plus a
list of tool definitions to the model. The model replies and tells you *why it
stopped* (`stop_reason`): if it stopped to **use a tool**, you run that tool,
append the result to the conversation, and call the API again so the model can
read the result and keep going; if it stopped with **`end_turn`**, it's done and
you print the answer. Repeat until done (or until a safety limit). The API is
stateless — *you* hold the conversation history and resend it every turn. That
send-message → inspect-stop-reason → run-tool → feed-result-back cycle is the
whole thing.

## Setup

```bash
cd minimal-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit .env (see Backends below)
```

## Backends (free first)

The `anthropic` SDK reads `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL` from the
environment, so you can point this exact code at any Anthropic-compatible
endpoint by editing three values in `.env`. **Tool use requires a tool-capable
model** on whichever backend you choose.

1. **LM Studio (free, local, offline) — the default.** Install
   [LM Studio](https://lmstudio.ai), download a tool-capable model, and start
   its local server. Then in `.env`:
   ```
   ANTHROPIC_BASE_URL=http://localhost:1234
   ANTHROPIC_API_KEY=lm-studio        # any non-empty string; LM Studio ignores it
   MODEL=<the model id shown in LM Studio>
   ```

2. **Hosted Anthropic-compatible gateway (free tier).** Some gateways expose an
   Anthropic `/v1/messages` endpoint. Set:
   ```
   ANTHROPIC_BASE_URL=https://<gateway-host>
   ANTHROPIC_API_KEY=<your gateway key>
   MODEL=<a tool-capable model the gateway offers>
   ```

3. **Real Anthropic API.** Leave `ANTHROPIC_BASE_URL` unset to hit
   `api.anthropic.com` directly:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   MODEL=claude-haiku-4-5
   ```

`.env` is git-ignored — never commit your real key.

## Run

**Free, local, no key (OpenAI format → Ollama):**
```bash
# Ollama running with a tool-capable model, e.g.:  ollama pull llama3.2
python agent_openai.py "Are there any data quality issues in sample_data/sales.csv? Show me examples."
```

**Claude (anthropic format), once you've set a backend in `.env`:**
```bash
python agent.py "Are there any data quality issues in sample_data/sales.csv? Show me examples."
```

Watch the labeled output: `[MODEL]` (what it said) → `[TOOL CALL]` (which tool,
with what args) → `[TOOL RESULT]` → next iteration, until `done`.

> Note: a small local model (e.g. llama3.2 3B) drives the loop correctly but its
> *analysis* is rough. Point `OPENAI_MODEL` at a larger local model, or use
> `agent.py` with Claude, for sharper answers. The loop is the same either way.

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

## Where to go next (exercises)

1. **Streaming.** Swap `client.messages.create(...)` for `client.messages.stream(...)`
   and print text deltas as they arrive, so you watch the model think in real time.
2. **Conversation memory across runs.** Persist `messages` to a JSON file at the
   end of `run_agent` and reload it at the start, so a second `python agent.py`
   call continues the previous conversation.
3. **A data-cleaning tool.** Add a fourth tool (e.g. `clean_csv`) that normalizes
   category casing/typos and writes a cleaned copy — then ask the agent to clean
   the data before analyzing it.
