"""Minimal stdio MCP server for the battery-lab spec runner.

The server speaks JSON-RPC 2.0 with Content-Length framing. It exposes
validation, dry-run, and execution of YAML specs. COMSOL is not required.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from . import __version__
from .spec import load_spec_text, run_spec, run_spec_file, validate_spec, validate_spec_file

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_NAME = "battery-lab"

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "validate_spec",
        "description": "Validate a battery-lab YAML spec from inline text or a file path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml": {"type": "string", "description": "YAML spec text"},
                "path": {"type": "string", "description": "Path to a YAML spec file"},
            },
        },
    },
    {
        "name": "run_spec",
        "description": (
            "Validate and run a simulation or surrogate-benchmark spec. "
            "Set dry_run to resolve the spec without executing it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml": {"type": "string"},
                "path": {"type": "string"},
                "dry_run": {"type": "boolean"},
                "output_dir": {"type": "string"},
            },
        },
    },
    {
        "name": "doctor",
        "description": "Report installed battery-lab dependencies and repository config status.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_configs",
        "description": "List YAML configs or specs under a directory.",
        "inputSchema": {
            "type": "object",
            "properties": {"root": {"type": "string"}},
        },
    },
]


def serve(stdin=None, stdout=None) -> None:
    incoming = sys.stdin.buffer if stdin is None else stdin
    outgoing = sys.stdout.buffer if stdout is None else stdout
    while True:
        message = read_message(incoming)
        if message is None:
            return
        response = handle_message(message)
        if response is not None:
            write_message(outgoing, response)


def read_message(stream) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("ascii", errors="replace").strip()
        if not decoded:
            break
        key, _, value = decoded.partition(":")
        headers[key.strip().lower()] = value.strip()
    if "content-length" not in headers:
        return None
    length = int(headers["content-length"])
    payload = stream.read(length)
    if len(payload) < length:
        return None
    return json.loads(payload.decode("utf-8"))


def write_message(stream, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stream.write(body)
    stream.flush()


def handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    if not isinstance(method, str):
        return _error(message.get("id"), -32600, "invalid request")
    if "id" not in message:
        return None
    request_id = message.get("id")
    if method == "initialize":
        params = message.get("params") or {}
        requested = params.get("protocolVersion")
        protocol = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1]
        return _result(
            request_id,
            {
                "protocolVersion": protocol,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": __version__},
            },
        )
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        return _result(request_id, {"tools": _TOOLS})
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _tool_error(request_id, "arguments must be an object")
        try:
            payload = _call_tool(str(name), arguments)
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
            return _tool_error(request_id, str(exc))
        is_error = not payload.get("ok", True)
        return _result(
            request_id,
            {
                "content": [
                    {"type": "text", "text": json.dumps(payload, indent=2, sort_keys=True)}
                ],
                "isError": is_error,
            },
        )
    return _error(request_id, -32601, f"unknown method: {method}")


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "validate_spec":
        spec, errors = _load_argument_spec(arguments)
        if errors:
            return {"ok": False, "errors": errors}
        if "yaml" in arguments:
            found = validate_spec(spec)
        else:
            found = validate_spec_file(arguments["path"])
        return {
            "ok": not found,
            "kind": spec.get("kind"),
            "name": spec.get("name"),
            "errors": found,
        }
    if name == "run_spec":
        dry_run = bool(arguments.get("dry_run", False))
        output_dir = arguments.get("output_dir")
        if "yaml" in arguments and "path" in arguments:
            return {"ok": False, "errors": ["provide yaml or path, not both"]}
        if "yaml" in arguments:
            spec = load_spec_text(str(arguments["yaml"]))
            return run_spec(spec, output_dir=output_dir, dry_run=dry_run)
        if "path" in arguments:
            return run_spec_file(arguments["path"], output_dir=output_dir, dry_run=dry_run)
        return {"ok": False, "errors": ["provide yaml or path"]}
    if name == "doctor":
        from .cli import doctor_report

        report = doctor_report()
        report["ok"] = report["ok"]
        return report
    if name == "list_configs":
        from pathlib import Path

        root = Path(str(arguments.get("root") or "configs"))
        if not root.exists():
            return {"ok": False, "errors": [f"path does not exist: {root}"]}
        paths = [root] if root.is_file() else sorted(root.glob("*.yaml"))
        return {"ok": True, "paths": [str(path) for path in paths]}
    return {"ok": False, "errors": [f"unknown tool: {name}"]}


def _load_argument_spec(arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if "yaml" in arguments and "path" in arguments:
        return {}, ["provide yaml or path, not both"]
    if "yaml" in arguments:
        return load_spec_text(str(arguments["yaml"])), []
    if "path" in arguments:
        from .spec import load_spec

        return load_spec(arguments["path"]), []
    return {}, ["provide yaml or path"]


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_error(request_id: Any, message: str) -> dict[str, Any]:
    return _result(
        request_id,
        {
            "content": [{"type": "text", "text": json.dumps({"ok": False, "errors": [message]})}],
            "isError": True,
        },
    )


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
