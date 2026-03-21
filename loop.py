"""
loop.py — The agentic loop: LLM → tool call → execute → repeat.

Runs up to MAX_ITERATIONS times per user message.
Streams tokens to on_token callback if provided.
Returns (new_messages, final_response) when the LLM stops calling tools.
"""

import base64
import json
import re
from pathlib import Path
from typing import Callable

MAX_ITERATIONS = 20

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


# ── Tool definition (auto-built from registry) ───────────────────────────────

def build_tool_definition(registry) -> dict:
    """
    Build the OpenAI tool definition for the single `run` tool,
    embedding the full help text so the LLM knows every command.
    """
    help_lines = "\n".join(f"  {name}: {desc}" for name, desc in registry.help().items())
    return {
        "type": "function",
        "function": {
            "name": "run",
            "description": (
                "Execute a shell-style command. "
                "Supports piping (|), sequencing (;), and conditionals (&& / ||).\n\n"
                "Available commands:\n" + help_lines
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command string to execute.",
                    },
                    "stdin": {
                        "type": "string",
                        "description": "Optional input to pass as stdin.",
                    },
                },
                "required": ["command"],
            },
        },
    }


# ── LLM call (streaming) ──────────────────────────────────────────────────────

def _call_llm(client, model: str, system_prompt: str, messages: list, tools: list, on_token: Callable | None):
    """
    Stream one LLM call. Returns the assembled assistant message dict.
    Tool calls are reassembled from streamed chunks by index.
    """
    content_parts = []
    tool_calls_map: dict[int, dict] = {}  # index → partial tool call

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        tools=tools,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            continue

        # Accumulate text content
        if delta.content:
            content_parts.append(delta.content)
            if on_token:
                on_token(delta.content)

        # Accumulate tool call chunks by index
        if delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                tc = tool_calls_map[idx]
                if tc_chunk.id:
                    tc["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tc["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tc["function"]["arguments"] += tc_chunk.function.arguments

    # Build the assembled assistant message
    msg: dict = {"role": "assistant"}
    if content_parts:
        msg["content"] = "".join(content_parts)
    if tool_calls_map:
        msg["tool_calls"] = [tool_calls_map[i] for i in sorted(tool_calls_map)]
    return msg


# ── Tool execution ────────────────────────────────────────────────────────────

def _exec_tool_call(registry, tool_call: dict) -> tuple[str, list]:
    """
    Execute a tool call. Returns (result_text, image_content_parts).
    image_content_parts is a list of vision content blocks for any image files found.
    """
    args = json.loads(tool_call["function"]["arguments"])
    command = args.get("command", "")
    stdin   = args.get("stdin", "")

    try:
        result, _ = registry.exec(command, stdin)
    except Exception as e:
        result = f"Error: {e}"

    images = _extract_images(result)
    return result, images


def _extract_images(result: str) -> list:
    """
    Scan result text for file:// image URLs.
    Returns OpenAI vision content blocks for each image found.
    """
    images = []
    for url in re.findall(r"file://([^\s)\"']+)", result):
        path = Path(url)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not path.exists():
            continue
        try:
            data = base64.b64encode(path.read_bytes()).decode()
            mime = {
                ".png":  "image/png",
                ".jpg":  "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif":  "image/gif",
                ".webp": "image/webp",
            }.get(path.suffix.lower(), "image/png")
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })
        except Exception:
            pass
    return images


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_loop(
    client,
    model: str,
    system_prompt: str,
    messages: list,
    registry,
    on_token: Callable | None = None,
    on_tool_call: Callable | None = None,
    on_tool_result: Callable | None = None,
) -> tuple[list, str]:
    """
    Run the agentic loop until the LLM stops calling tools or MAX_ITERATIONS is reached.

    Callbacks:
        on_token(text)          — called for each streamed text token
        on_tool_call(command)   — called when the LLM issues a run() command
        on_tool_result(result)  — called with the command's output

    Returns:
        new_messages   — all messages generated during this run (for saving to DB)
        final_response — the last text response from the LLM
    """
    tools = [build_tool_definition(registry)]
    new_messages: list = []
    final_response = ""

    for iteration in range(MAX_ITERATIONS):
        assistant_msg = _call_llm(client, model, system_prompt, messages + new_messages, tools, on_token)
        new_messages.append(assistant_msg)

        # No tool call — the LLM is done
        if not assistant_msg.get("tool_calls"):
            final_response = assistant_msg.get("content", "")
            break

        # Execute each tool call and append results
        for tc in assistant_msg["tool_calls"]:
            # Parse command for the callback before executing
            if on_tool_call:
                try:
                    import json as _json
                    _args = _json.loads(tc["function"]["arguments"])
                    _cmd = _args.get("command", "")
                    if _args.get("stdin"):
                        _cmd += f"  (stdin: {_args['stdin'][:80]})"
                    on_tool_call(_cmd)
                except Exception:
                    on_tool_call(tc["function"]["arguments"])

            result_text, images = _exec_tool_call(registry, tc)

            if on_tool_result:
                on_tool_result(result_text)

            # Build content: text result, optionally followed by image blocks
            if images:
                content = [{"type": "text", "text": result_text}] + images
            else:
                content = result_text

            new_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": content,
            })

    return new_messages, final_response
