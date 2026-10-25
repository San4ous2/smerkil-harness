"""OpenAI-compatible chat client over stdlib (http.client), with SSE streaming,
exponential-backoff retries honoring Retry-After, and token accounting."""
from __future__ import annotations

import json
import random
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from http.client import HTTPConnection, HTTPSConnection
from typing import Any, Callable, Dict, List, Optional

_CTX = ssl.create_default_context()

TRANSIENT_STATUSES = (429, 500, 502, 503, 504)


class LLMError(Exception):
    pass


class ModelFallbackExhausted(LLMError):
    """Every model in the chain (primary + fallbacks) failed."""

    def __init__(self, attempts: List[str]):
        self.attempts = attempts
        super().__init__("all models failed: " + "; ".join(attempts))


class LLMClient:
    """Low-level chat client for any OpenAI-compatible endpoint.

    Supports SSE streaming with a token-delta callback, and automatic
    non-streaming fallback when the endpoint rejects streaming.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.last_good_model: str = ""  # sticky failover: model that last served

    # ---------------- retries ----------------
    def _retry_after(self, headers: Any) -> Optional[float]:
        try:
            ra = headers.get("Retry-After") if headers is not None else None
        except Exception:
            ra = None
        if ra is None:
            return None
        try:
            return max(0.0, float(ra))
        except (TypeError, ValueError):
            return None

    def _request(self, payload: Dict[str, Any],
                 on_delta: Optional[Callable[[str], None]] = None,
                 max_attempts: int = 4,
                 on_tool: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
        url = self.base_url + "/chat/completions"
        body = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream" if payload.get("stream") else "application/json",
        }
        last_err = "unknown error"
        for attempt in range(max_attempts):
            try:
                if payload.get("stream"):
                    return self._request_stream(url, body, headers, on_delta, on_tool)
                return self._request_plain(url, body, headers)
            except LLMError:
                # Non-retryable: the endpoint answered definitively.
                # Special case: some gateways reject streaming outright.
                raise
            except urllib.error.HTTPError as e:
                if e.code in TRANSIENT_STATUSES:
                    last_err = f"HTTP {e.code}"
                    if attempt == max_attempts - 1:
                        break
                    ra = self._retry_after(e.headers)
                    time.sleep(ra if ra is not None else 1.5 ** attempt + random.uniform(0, 0.5))
                    continue
                raise LLMError(f"HTTP {e.code}: {_safe_body(e)}") from e
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt == max_attempts - 1:
                    break
                time.sleep(1.5 ** attempt + random.uniform(0, 0.5))
        raise LLMError(f"LLM unreachable after {max_attempts} attempts: {last_err}")

    def chat(self, payload: Dict[str, Any],
             on_delta: Optional[Callable[[str], None]] = None,
             on_tool: Optional[Callable[[str, str], None]] = None,
             max_attempts: int = 4,
             model_chain: Optional[List[str]] = None) -> Dict[str, Any]:
        """Public entry. If on_delta is set, tries streaming; falls back to
        non-streaming when the endpoint doesn't support SSE.

        on_tool: fn(tool_name, args_fragment) — streamed tool-call deltas
        (called incrementally as the model writes arguments, best-effort).

        model_chain: optional ordered list of models. The first is tried with
        full per-model retries; on definitive (non-transient) failure or final
        transient exhaustion the next model is tried. Primary model first.
        """
        models = [m for m in (model_chain or []) if m] or [payload.get("model", "")]
        # sticky failover: once a fallback has served, prefer it on later calls
        # (skips re-paying the dead primary's failure latency every turn)
        if self.last_good_model and self.last_good_model in models[1:]:
            models = [self.last_good_model] + [m for m in models if m != self.last_good_model]
        attempts_log: List[str] = []
        for i, model in enumerate(models):
            p = dict(payload, model=model)
            try:
                if on_delta is not None or on_tool is not None:
                    stream_payload = dict(p, stream=True)
                    try:
                        out = self._request(stream_payload, on_delta, max_attempts,
                                            on_tool=on_tool)
                    except LLMError as e:
                        msg = str(e)
                        if "stream" in msg.lower() or "SSE" in msg or "event-stream" in msg:
                            # endpoint lacks stream support → plain request
                            out = self._request(dict(p, stream=False), None, max_attempts)
                        else:
                            raise
                else:
                    out = self._request(dict(p, stream=False), None, max_attempts)
                self.last_good_model = model
                return out
            except LLMError as e:
                attempts_log.append(f"{model}: {str(e)[:200]}")
                if i == len(models) - 1:
                    if len(models) > 1:
                        raise ModelFallbackExhausted(attempts_log) from e
                    raise
                continue
        raise ModelFallbackExhausted(attempts_log)

    # ---------------- transport ----------------
    def _conn(self, url: str):
        p = urllib.parse.urlsplit(url)
        if p.scheme == "https":
            return HTTPSConnection(p.hostname, p.port or 443, timeout=self.timeout, context=_CTX)
        return HTTPConnection(p.hostname, p.port or 80, timeout=self.timeout)

    @staticmethod
    def _path(url: str) -> str:
        p = urllib.parse.urlsplit(url)
        return p.path + ("?" + p.query if p.query else "")

    def _request_plain(self, url: str, body: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
        conn = self._conn(url)
        try:
            conn.request("POST", self._path(url), body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            if resp.status in TRANSIENT_STATUSES:
                err = urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
                try:
                    err.headers = resp.headers  # keep Retry-After if present
                except Exception:
                    pass
                raise err
            if resp.status != 200:
                raise LLMError(f"HTTP {resp.status}: {data[:300].decode(errors='replace')}")
            return json.loads(data.decode())
        finally:
            conn.close()

    def _request_stream(self, url: str, body: bytes, headers: Dict[str, str],
                        on_delta: Optional[Callable[[str], None]],
                        on_tool: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
        conn = self._conn(url)
        try:
            conn.request("POST", self._path(url), body=body, headers=headers)
            resp = conn.getresponse()
            if resp.status in TRANSIENT_STATUSES:
                err = urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
                try:
                    err.headers = resp.headers
                except Exception:
                    pass
                raise err
            if resp.status != 200:
                raise LLMError(f"HTTP {resp.status}: {resp.read()[:300].decode(errors='replace')}")
            content_type = resp.headers.get("Content-Type") or ""
            if "text/event-stream" not in content_type:
                # endpoint doesn't stream — treat whole body as JSON
                data = resp.read()
                try:
                    return json.loads(data.decode())
                except json.JSONDecodeError as e:
                    raise LLMError(
                        f"stream unsupported: endpoint returned {content_type}") from e
            return self._read_sse(resp, on_delta, on_tool)
        finally:
            conn.close()

    # ---------------- SSE parsing ----------------
    def _read_sse(self, resp: Any, on_delta: Optional[Callable[[str], None]],
                  on_tool: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
        content_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        _tool_emitted: set = set()  # (idx, len(args_emitted)) for delta callback
        role = None
        usage: Dict[str, Any] = {}
        finish_reason = None
        model = None
        done = False
        buf = b""
        while not done:
            chunk = resp.read(8192)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf and not done:
                line_b, buf = buf.split(b"\n", 1)
                line = line_b.decode(errors="replace").rstrip("\r")
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                model = obj.get("model") or model
                if isinstance(obj.get("usage"), dict) and obj["usage"]:
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("role"):
                        role = delta["role"]
                    piece = delta.get("content")
                    if piece:
                        content_parts.append(piece)
                        if on_delta:
                            try:
                                on_delta(piece)
                            except Exception:
                                pass  # callbacks must never break the stream
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0) or 0)
                        slot = tool_calls.setdefault(idx, {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                            if on_tool:
                                try:
                                    key = (idx, len(slot["function"]["arguments"]))
                                    if key not in _tool_emitted:
                                        _tool_emitted.add(key)
                                        on_tool(slot["function"]["name"] or "?",
                                                fn["arguments"])
                                except Exception:
                                    pass  # callbacks must never break the stream
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        return self._assemble(content_parts, tool_calls, role, usage, finish_reason, model)

    @staticmethod
    def _assemble(content_parts: List[str], tool_calls: Dict[int, Dict[str, Any]],
                  role: Optional[str], usage: Dict[str, Any],
                  finish_reason: Optional[str], model: Optional[str]) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": role or "assistant",
                                   "content": "".join(content_parts) or None}
        if tool_calls:
            message["tool_calls"] = [tool_calls[k] for k in sorted(tool_calls)]
            for tc in message["tool_calls"]:
                if not tc["id"]:
                    tc["id"] = f"call_{tc['function']['name'] or 'unknown'}"
            if message["content"] is None:
                message["content"] = ""
        out: Dict[str, Any] = {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
        }
        if model:
            out["model"] = model
        return out


def _safe_body(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode(errors="replace")[:300]
    except Exception:
        return ""
