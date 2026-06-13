"""The SAME agent loop as agent.py, but in OpenAI wire format.

Why this file exists: the agent *loop* is provider-agnostic — send the
conversation + tools, look at why the model stopped, run any requested tools,
feed the results back, repeat. Only the **wire format** differs between
providers. This version talks the OpenAI Chat Completions format, which lets it
run for free against a local Ollama model (or any OpenAI-compatible server)
with no API key and no bridge.

`diff agent.py agent_openai.py` to see the format differences side by side. The
shared tool *functions* live in tools.py and are reused unchanged — only the
tool *schemas* and the message plumbing change.

Run it (Ollama must be running with a tool-capable model like llama3.2):

    python agent_openai.py "your question here"

Config via env (.env): OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL.
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

MODEL = os.getenv("OPENAI_MODEL", "llama3.2:latest")
MAX_ITERATIONS = 10
MAX_TOKENS = 2048

SYSTEM_PROMPT = (
    "You are a careful data analyst with access to a small set of tools for "
    "listing files, reading files, and querying CSVs with SQL (DuckDB). "
    "When asked about data, inspect it with the tools before answering. "
    "When you find data quality problems, show concrete examples from the data."
)

# --- Format difference #1: tool schemas -------------------------------------
# Anthropic tools are {name, description, input_schema}. OpenAI wraps each tool
# as {"type": "function", "function": {name, description, parameters}}. We DERIVE
# the OpenAI shape from the shared Anthropic TOOLS so the schema lives in one
# place (DRY) — no second copy to keep in sync.
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

    Format difference #2: OpenAI delivers tool arguments as a JSON *string* in
    `tc.function.arguments` (Anthropic gives a parsed dict in block.input), so
    we json.loads() them. And a tool result is a message with role "tool"
    carrying `tool_call_id` (Anthropic uses a tool_result block with
    tool_use_id inside a user message).
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
    """Drive the agent loop for a single user question (OpenAI format)."""
    # Format difference #3: the system prompt is the first message (Anthropic
    # passes it as a separate `system` argument).
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

        # Format difference #4: we check msg.tool_calls (and finish_reason ==
        # "tool_calls") instead of Anthropic's stop_reason == "tool_use".
        if msg.tool_calls:
            # Append the assistant turn, including its tool_calls, before the
            # results — same ordering rule as the Anthropic version.
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
