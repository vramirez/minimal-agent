"""Tool implementations and their Anthropic tool schemas.

A "tool" here is two things glued together:
  1. A plain Python function that does the work.
  2. A JSON schema (in TOOLS) that tells the model the tool exists, what it
     does, and what arguments it takes.

The model never runs our code. It only ever *asks* us to run a tool by name
with some arguments; the agent loop (agent.py) does the actual calling. Keeping
that split clean is the whole point of tool use.
"""

import os

import duckdb


# --- The functions ------------------------------------------------------------
# One function = one job. Each takes simple arguments and returns a string,
# because a tool result must be text the model can read back.


def list_files(directory: str) -> str:
    """List the entries in `directory`, one per line."""
    names = sorted(os.listdir(directory))
    return "\n".join(names) if names else "(empty directory)"


def read_file(path: str) -> str:
    """Return the full text contents of the file at `path`."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def query_csv(file_path: str, sql: str) -> str:
    """Run a SQL query against a CSV and return the result as a text table.

    DuckDB can read a CSV file directly inside a query, e.g.:
        SELECT * FROM 'sample_data/sales.csv'
    so the model can explore the data with ordinary SQL without us writing any
    parsing code. We format the result ourselves (pipe-separated) rather than
    via pandas, to keep the dependency footprint tiny.
    """
    relation = duckdb.sql(sql)
    columns = relation.columns
    rows = relation.fetchall()
    header = " | ".join(columns)
    body = [
        " | ".join("NULL" if value is None else str(value) for value in row)
        for row in rows
    ]
    table = "\n".join([header, *body])
    return f"{table}\n({len(rows)} rows)"


# --- The schemas the model sees -----------------------------------------------
# `description` and the per-parameter descriptions are how the model decides
# *when* and *how* to call each tool. Be specific and prescriptive.

TOOLS = [
    {
        "name": "list_files",
        "description": "List the files in a directory. Use this first to "
        "discover what data files are available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory, e.g. 'sample_data'.",
                }
            },
            "required": ["directory"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file's full contents. Use this to look at "
        "raw data, e.g. the first lines of a CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "query_csv",
        "description": "Run a DuckDB SQL query against a CSV file and get the "
        "result back as a table. Query the CSV directly by quoting its path in "
        "the FROM clause, e.g. SELECT * FROM 'sample_data/sales.csv' LIMIT 5. "
        "Use this to count, aggregate, and inspect the data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV file being queried.",
                },
                "sql": {
                    "type": "string",
                    "description": "A DuckDB SQL query. Reference the CSV by its "
                    "quoted path in the FROM clause.",
                },
            },
            "required": ["file_path", "sql"],
        },
    },
]


# Map tool name -> function. This is the single place the agent loop looks up
# which function to run, so adding a tool means: write the function, add its
# schema to TOOLS, add one line here.
_DISPATCH = {
    "list_files": list_files,
    "read_file": read_file,
    "query_csv": query_csv,
}


def dispatch(name: str, tool_input: dict) -> str:
    """Run the named tool with the given arguments and return its text result.

    Precondition: `name` is a key in _DISPATCH and `tool_input` holds exactly
    the arguments that tool's schema declares. We let exceptions propagate; the
    agent loop catches them and feeds the error text back to the model so it can
    self-correct (e.g. fix a bad SQL query and try again).
    """
    return _DISPATCH[name](**tool_input)
