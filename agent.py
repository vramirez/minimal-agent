"""A minimal AI agent you can read top to bottom.

An "agent" is just a loop around the Messages API:

    1. Send the conversation (plus the list of tools) to the model.
    2. Look at WHY the model stopped (`stop_reason`):
         - "end_turn"  -> it's done talking; print the answer and exit.
         - "tool_use"  -> it wants to run one or more tools. We run them,
                          append the results to the conversation, and loop.
    3. Repeat until it's done (or we hit a safety limit).

That's the entire idea. Everything below is bookkeeping around those steps.
Run it with:

    python agent.py "your question here"

Backend is whatever ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY / MODEL point at
(see .env.example) — real Claude, a local LM Studio server, or a gateway.
"""

import os
import sys

import anthropic
from dotenv import load_dotenv

from tools import TOOLS, dispatch

# Load ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL / MODEL from a local .env file.
load_dotenv()

# The SDK reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from the environment by
# itself. That single fact is what lets us point the *same* code at Claude, a
# local LM Studio server, or any Anthropic-compatible gateway with zero changes.
client = anthropic.Anthropic()

MODEL = os.getenv("MODEL", "claude-haiku-4-5")

# Hard ceiling on loop iterations. Without this, a confused model could call
# tools forever. Crash-early beats an infinite bill (Pragmatic tip: Crash Early).
MAX_ITERATIONS = 10

# Cap on tokens per response. Small is fine for this learning agent.
MAX_TOKENS = 2048

# A short system prompt steering the model toward using the tools.
SYSTEM_PROMPT = (
    "You are a careful data analyst with access to a small set of tools for "
    "listing files, reading files, and querying CSVs with SQL (DuckDB). "
    "When asked about data, inspect it with the tools before answering. "
    "When you find data quality problems, show concrete examples from the data."
)


def print_assistant_text(content_blocks) -> None:
    """Print any plain-text the model produced this turn (its 'thinking out loud')."""
    for block in content_blocks:
        if block.type == "text" and block.text.strip():
            print(f"\n[MODEL] {block.text.strip()}")


def run_tool_calls(content_blocks) -> list:
    """Execute every tool_use block and return matching tool_result blocks.

    For each tool the model asked for we:
      - print the call so you can watch it happen,
      - run it (catching errors so one bad call doesn't kill the agent),
      - print the result,
      - build a tool_result block carrying that text back to the model.

    Returning the error text with is_error=True (instead of crashing) is what
    lets the agent self-correct: it sees the error and can try a different call.
    """
    results = []
    for block in content_blocks:
        if block.type != "tool_use":
            continue

        print(f"\n[TOOL CALL] {block.name}({block.input})")
        try:
            output = dispatch(block.name, block.input)
            is_error = False
        except Exception as exc:  # noqa: BLE001 - we deliberately surface ALL errors to the model
            output = f"Error running {block.name}: {exc}"
            is_error = True

        # Trim very long output in the console view (the model still gets it all).
        preview = output if len(output) < 800 else output[:800] + "\n...[truncated in console]"
        label = "TOOL ERROR" if is_error else "TOOL RESULT"
        print(f"[{label}] {preview}")

        # A tool_result MUST reference the tool_use id it answers, so the model
        # can match each result to the call it made.
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
                "is_error": is_error,
            }
        )
    return results


def run_agent(question: str) -> None:
    """Drive the agent loop for a single user question."""
    # The conversation is just a growing list of messages we resend every turn;
    # the Messages API is stateless, so WE hold the history.
    messages = [{"role": "user", "content": question}]

    for step in range(1, MAX_ITERATIONS + 1):
        print(f"\n===== iteration {step} =====")

        # STEP 1: send the whole conversation plus the tool definitions.
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Show whatever the model said in words this turn.
        print_assistant_text(response.content)

        # STEP 2: inspect WHY it stopped.
        if response.stop_reason == "tool_use":
            # The model wants tools. We MUST first append its turn (which
            # contains the tool_use blocks) so the conversation stays coherent...
            messages.append({"role": "assistant", "content": response.content})
            # ...then run the tools and append their results as a user turn.
            tool_results = run_tool_calls(response.content)
            messages.append({"role": "user", "content": tool_results})
            # Loop again so the model can read the results and continue.
            continue

        # Any other stop_reason ("end_turn", "max_tokens", ...) means the model
        # isn't asking for a tool, so this turn is the final answer. Stop.
        print(f"\n===== done (stop_reason: {response.stop_reason}) =====")
        return

    # If we fall out of the for-loop, the model kept calling tools past our
    # safety limit. Stop deliberately rather than looping forever.
    print(f"\n[SAFETY] Hit MAX_ITERATIONS ({MAX_ITERATIONS}). Stopping.")


def main() -> None:
    # Default to the data-quality prompt so `python agent.py` just works.
    default_question = (
        "Are there any data quality issues in sample_data/sales.csv? "
        "Show me examples."
    )
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else default_question
    print(f"[QUESTION] {question}")
    run_agent(question)


if __name__ == "__main__":
    main()
