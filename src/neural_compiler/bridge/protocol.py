"""Qwen native tool calls and chat-template-derived return turns."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from transformers import PreTrainedTokenizerBase

from .execution import ExecutionResult, validate_code

RUN_TOOL = {
    "type": "function",
    "function": {
        "name": "run",
        "description": "Run Python code and return structured stdout or error information.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

_CALL = re.compile(
    r"\s*<function=([^>\s]+)>\s*<parameter=([^>\s]+)>(.*?)</parameter>\s*</function>\s*",
    re.S,
)


class ProtocolError(ValueError):
    """Malformed, truncated, or unsupported native tool framing."""


@dataclass(frozen=True)
class ToolCall:
    code: str


class QwenToolProtocol:
    """One bounded stream monitor per sequence; only run(code: str) is supported."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_call_tokens: int = 4096,
        max_response_tokens: int = 2048,
    ):
        for name, value in (
            ("max_call_tokens", max_call_tokens),
            ("max_response_tokens", max_response_tokens),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.tokenizer = tokenizer
        self.max_call_tokens = max_call_tokens
        self.max_response_tokens = max_response_tokens
        markers = {}
        for marker in (
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<|im_start|>",
            "<|im_end|>",
        ):
            ids = tokenizer.encode(marker, add_special_tokens=False)
            if len(ids) != 1:
                raise ProtocolError(f"{marker} must be a single tokenizer token")
            markers[marker] = ids[0]
        if len(set(markers.values())) != len(markers):
            raise ProtocolError(
                "Native tool/turn delimiters must have distinct token IDs"
            )
        self._start = markers["<tool_call>"]
        self._end = markers["</tool_call>"]
        self._reserved = set(markers.values())
        self._tokens: list[int] | None = None

    def feed(self, token_id: int) -> ToolCall | None:
        if token_id == self._start:
            if self._tokens is not None:
                raise ProtocolError("Nested tool calls are not supported")
            self._tokens = []
            return None
        if token_id == self._end:
            if self._tokens is None:
                raise ProtocolError("Tool-call closing token without an opening token")
            body = self.tokenizer.decode(self._tokens, skip_special_tokens=False)
            self._tokens = None
            return self._parse(body)
        if self._tokens is not None:
            if token_id in self._reserved:
                raise ProtocolError("Unexpected turn/control token inside a tool call")
            if len(self._tokens) >= self.max_call_tokens:
                raise ProtocolError("Tool call exceeded its token budget")
            self._tokens.append(token_id)
        return None

    def finish(self) -> None:
        if self._tokens is not None:
            raise ProtocolError("Incomplete tool call at end of generation")

    @staticmethod
    def _parse(body: str) -> ToolCall:
        match = _CALL.fullmatch(body)
        if match is None:
            raise ProtocolError("Expected one <function=run><parameter=code> call")
        name, parameter, code = match.groups()
        if name != "run" or parameter != "code":
            raise ProtocolError("Only run with one code parameter is supported")
        if re.search(r"</parameter>\s*<parameter=|</function>\s*<function=", code):
            raise ProtocolError("Multiple functions or parameters are not supported")
        # Strip only the template's framing newline, never Python indentation.
        code = code.removeprefix("\n").removesuffix("\n")
        if not code.strip():
            raise ProtocolError("Tool code must not be empty")
        try:
            validate_code(code)
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc
        return ToolCall(code)

    def response_ids(self, call: ToolCall, result: ExecutionResult) -> list[int]:
        payload = asdict(result)
        while True:
            ids = self._response_ids(
                call, json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            )
            if len(ids) <= self.max_response_tokens:
                return ids
            stdout, stderr = payload["stdout"], payload["stderr"]
            if not stdout and not stderr:
                raise ProtocolError(
                    "Response token budget cannot fit the tool-result envelope"
                )
            payload["stdout"] = stdout[: len(stdout) // 2]
            payload["stderr"] = stderr[: len(stderr) // 2]
            payload["truncated"] = True

    def _response_ids(self, call: ToolCall, payload: str) -> list[int]:
        # Escaping angle brackets preserves JSON values without letting guest
        # output tokenize into chat-control delimiters.
        payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
        messages = [
            {"role": "user", "content": ""},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "run", "arguments": {"code": call.code}},
                    },
                ],
            },
        ]
        completed = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        resumed = self.tokenizer.apply_chat_template(
            [*messages, {"role": "tool", "content": payload}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(completed, str) or not isinstance(resumed, str):
            raise ProtocolError("Chat template must render text")
        end = completed.rfind("</tool_call>")
        if end < 0 or not resumed.startswith(completed):
            raise ProtocolError(
                "Chat template cannot extend a completed native tool call"
            )
        suffix = completed[end + len("</tool_call>") :] + resumed[len(completed) :]
        return self.tokenizer.encode(suffix, add_special_tokens=False)
