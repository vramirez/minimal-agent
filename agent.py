"""A minimal AI agent you can read top to bottom.

An agent is just a loop around one API call:

    1. Send the conversation + the list of tools to the model.
    2. Look at what the model did:
         - it asked to call tools -> run them, append the results, loop.
         - it just answered          -> print it and stop.
    3. Repeat until done (or a safety limit).

The API is stateless, so WE hold the conversation history and resend it every
turn. That send -> run-tool -> feed-result-back cycle is the whole idea.

This talks the OpenAI Chat Completions format, which runs for **free** against a
local Ollama model (no API key, no bridge). The tool *functions* live in
tools.py; here we only handle the schemas and the message plumbing. Comments
marked "vs Anthropic" point out where Claude's native tool-use format differs,
since that's a common next thing to learn.

Run it (Ollama must be running with a tool-capable model, e.g. llama3.2):

    python agent.py "your question here"

Config via env (.env): OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL.
Note: reasoning-only models like deepseek-r1 do NOT support tool calling and
will be rejected by Ollama — use a tool-capable model (llama3.2, qwen2.5, ...).
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOLS, dispatch

load_dotenv()

# Point the OpenAI SDK at a local Ollama server by default. Ollama exposes an
# OpenAI-compatible API at /v1; it ignores the api_key but the SDK requires one.
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
)

MODEL = os.getenv("OPENAI_MODEL", "qwen2.5:latest")
MAX_ITERATIONS = 10
MAX_TOKENS = 2048

SYSTEM_PROMPT = (
    "You are a careful data analyst with access to a small set of tools for "
    "listing files, reading files, and querying CSVs with SQL (DuckDB). "
    "When asked about data, inspect it with the tools before answering. "
    "When you find data quality problems, show concrete examples from the data."
)

# --- Tool schemas ------------------------------------------------------------
# OpenAI wraps each tool as {"type": "function", "function": {name, description,
# parameters}}. We derive that shape from the schemas in tools.py so the schema
# lives in one place (DRY). (vs Anthropic: tools are {name, description,
# input_schema} passed directly — no "function" wrapper.)
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]


def run_tool_calls(tool_calls) -> list:
    """Execute each tool call and return matching 'tool' messages.

    OpenAI delivers tool arguments as a JSON *string* in `tc.function.arguments`,
    so we json.loads() them. A tool result goes back as a message with role
    "tool" carrying the `tool_call_id` it answers. (vs Anthropic: arguments
    arrive as an already-parsed dict, and results go in a tool_result block with
    a tool_use_id inside a user message.)
    """
    messages = []
    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        print(f"\n[TOOL CALL] {tc.function.name}({args})")
        try:
            output = dispatch(tc.function.name, args)
        except Exception as exc:  # noqa: BLE001 - surface ALL errors back to the model
            output = f"Error running {tc.function.name}: {exc}"

        preview = output if len(output) < 800 else output[:800] + "\n...[truncated in console]"
        print(f"[TOOL RESULT] {preview}")

        messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": output}
        )
    return messages


def run_agent(question: str) -> None:
    """Drive the agent loop for a single user question."""
    # The system prompt is the first message. (vs Anthropic: it's passed as a
    # separate `system=` argument instead of a message.)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_ITERATIONS + 1):
        print(f"\n===== iteration {step} =====")

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=OPENAI_TOOLS,
            messages=messages,
        )
        msg = response.choices[0].message

        # Show whatever the model said in words this turn.
        if msg.content and msg.content.strip():
            print(f"\n[MODEL] {msg.content.strip()}")

        # We branch on msg.tool_calls (finish_reason == "tool_calls"). (vs
        # Anthropic: you check stop_reason == "tool_use".)
        if msg.tool_calls:
            # Append the assistant turn, including its tool_calls, BEFORE the
            # results, so each result lines up with the call that produced it.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            messages.extend(run_tool_calls(msg.tool_calls))
            continue

        # No tool calls -> this turn is the final answer.
        print(f"\n===== done (finish_reason: {response.choices[0].finish_reason}) =====")
        return

    print(f"\n[SAFETY] Hit MAX_ITERATIONS ({MAX_ITERATIONS}). Stopping.")


def main() -> None:
    default_question = (
        "Are there any data quality issues in sample_data/sales.csv? "
        "Show me examples."
    )
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else default_question
    print(f"[QUESTION] {question}")
    run_agent(question)


if __name__ == "__main__":
    main()
