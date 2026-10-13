import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import litellm

from core.api_resilience import is_infrastructure_error
from core.api_circuit_breaker import get_shared_circuit_breaker

logger = logging.getLogger(__name__)

_default_env_path = Path(__file__).parent.parent / ".env"
_configured_env_path = os.environ.get("EVOTEAM_ENV_FILE", "").strip()
_env_path = (
    Path(_configured_env_path).expanduser()
    if _configured_env_path
    else _default_env_path
)
if _env_path.exists():
    override_env = bool(_configured_env_path)
    for line in _env_path.read_text().strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, val = line.split("=", 1)
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            else:
                if " #" in val:
                    val = val[:val.index(" #")].strip()
            env_key = key.strip()
            if override_env or env_key not in os.environ:
                os.environ[env_key] = val
else:
    config_hint = "EVOTEAM_ENV_FILE" if _configured_env_path else ".env"
    logger.warning(
        "%s file not found at %s — API configuration must be provided "
        "via environment variables. See .env.example for the template.",
        config_hint, _env_path,
    )

_llm_provider = os.environ.get(
    "LLM_PROVIDER",
    "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "dev",
).lower().strip()

_PROVIDER_CONFIGS = {
    "venus": {
        "base": os.environ.get("VENUS_API_BASE", os.environ.get("ANTHROPIC_API_BASE", "")),
        "key": os.environ.get("VENUS_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")),
    },
    "dev": {
        "base": os.environ.get("DEV_API_BASE", ""),
        "key": os.environ.get("DEV_API_KEY", "not-needed"),
    },
    "anthropic": {
        "base": os.environ.get("ANTHROPIC_API_BASE", os.environ.get("ANTHROPIC_BASE_URL", "")),
        "key": os.environ.get("ANTHROPIC_API_KEY", ""),
    },
}

if _llm_provider not in _PROVIDER_CONFIGS:
    logger.warning("Unknown LLM_PROVIDER=%r, falling back to 'dev'", _llm_provider)
    _llm_provider = "dev"

_cfg = _PROVIDER_CONFIGS[_llm_provider]

if not _cfg["base"] and _llm_provider != "anthropic":
    raise RuntimeError(
        f"LLM_PROVIDER='{_llm_provider}' but API base URL is empty. "
        f"Please set {_llm_provider.upper()}_API_BASE in .env file. "
        f"See .env.example for the configuration template."
    )
if _llm_provider in ("venus", "anthropic") and not _cfg["key"]:
    raise RuntimeError(
        f"LLM_PROVIDER='{_llm_provider}' but its API key is empty. "
        f"Please set {_llm_provider.upper()}_API_KEY in .env file."
    )

if _llm_provider == "anthropic":
    os.environ["ANTHROPIC_API_KEY"] = _cfg["key"]
    if _cfg["base"]:
        os.environ["ANTHROPIC_API_BASE"] = _cfg["base"]
else:
    os.environ["OPENAI_API_BASE"] = _cfg["base"]
    os.environ["OPENAI_API_KEY"] = _cfg["key"]
logger.info("LLM Provider: %s → %s", _llm_provider, _cfg["base"])

_VENUS_MODEL_MAP: dict[str, str] = {
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-opus-4.6": "claude-opus-4-6",
}


def _map_model_name(model: str) -> str:
    if _llm_provider == "venus" and model in _VENUS_MODEL_MAP:
        mapped = _VENUS_MODEL_MAP[model]
        logger.debug("Venus model mapping: %s → %s", model, mapped)
        return mapped
    if _llm_provider == "anthropic":
        if model in _VENUS_MODEL_MAP:
            model = os.environ.get("ANTHROPIC_MODEL", model)
        return model if "/" in model else f"anthropic/{model}"
    return model

litellm.set_verbose = False
litellm.drop_params = True



_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.BadGatewayError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.APIError,
    litellm.exceptions.APIResponseValidationError,
    litellm.exceptions.MidStreamFallbackError,
    litellm.exceptions.Timeout,
    litellm.exceptions.InternalServerError,
)

MAX_RETRIES = 5
BASE_DELAY = 1.5
MAX_DELAY = 60.0
JITTER_RANGE = 1.0
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT", "300"))
_shared_circuit = None


def _circuit():
    global _shared_circuit
    if _shared_circuit is None:
        _shared_circuit = get_shared_circuit_breaker()
    return _shared_circuit


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, asyncio.CancelledError):
        return False
    if hasattr(exc, "status_code"):
        code = exc.status_code
        if code == 429 or code >= 500:
            return True
    msg = str(exc).lower()
    transient_keywords = (
        "no available passageway",
        "no available channel",
        "upstream timeout",
        "upstream connection",
        "connection reset",
        "connection aborted",
        "remote protocol error",
        "incomplete chunked read",
        "temporarily unavailable",
        "try again later",
        "bad gateway",
        "status 500", "status 502", "status 503", "status 504", "status 529",
        "http 500", "http 502", "http 503", "http 504", "http 529",
        "service unavailable",
        "gateway timeout",
        "overloaded",
        "model service error",
        "model service error (cn)",
        "provider error",
        "provider failure",
        "request failed",
        "request error",
        "invalid upstream response",
        "malformed upstream response",
        "mid-stream",
    )
    if any(kw in msg for kw in transient_keywords):
        return True
    return False


async def complete(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 1.0,
    max_tokens: int = 16384,
    stop: list[str] | None = None,
    api_tracker: Any | None = None,
) -> Any:
    model = _map_model_name(model)

    if "/" not in model and os.environ.get("OPENAI_API_BASE"):
        model = f"openai/{model}"

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": REQUEST_TIMEOUT_SECONDS,
    }
    if tools:
        kwargs["tools"] = tools
    if stop:
        kwargs["stop"] = stop

    last_exc: Exception | None = None
    total_retry_delay = 0.0
    for attempt in range(MAX_RETRIES + 1):
        attempt_started = time.monotonic()
        breaker = None
        try:
            breaker = _circuit()
            if breaker is not None:
                breaker.check()
            response = await asyncio.wait_for(
                litellm.acompletion(**kwargs), timeout=REQUEST_TIMEOUT_SECONDS
            )
            if attempt > 0:
                logger.info(
                    "LLM call succeeded after %d retries (total retry delay: %.1fs)",
                    attempt, total_retry_delay,
                )
            if breaker is not None:
                breaker.record_success()
            return response
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable(exc)
            will_retry = attempt < MAX_RETRIES and retryable
            delay = 0.0
            if will_retry:
                delay = min(
                    BASE_DELAY * (2 ** attempt) + random.uniform(0, JITTER_RANGE),
                    MAX_DELAY,
                )
            if api_tracker is not None:
                try:
                    api_tracker.record_failure(
                        started_at=attempt_started,
                        ended_at=time.monotonic(),
                        error=exc,
                        retryable=retryable,
                        infrastructure=is_infrastructure_error(exc),
                        retry_delay_seconds=delay,
                        final_failure=not will_retry,
                    )
                except Exception:
                    logger.debug("failed to record API incident", exc_info=True)
            if (
                breaker is not None
                and is_infrastructure_error(exc)
                and not isinstance(exc, asyncio.CancelledError)
                and type(exc).__name__ != "APICircuitOpenError"
            ):
                if not will_retry:
                    breaker.record_infrastructure_failure(exc)
            if not will_retry:
                if attempt > 0:
                    logger.error(
                        "LLM call failed permanently after %d retries "
                        "(total retry delay: %.1fs): [%s] %s",
                        attempt, total_retry_delay,
                        type(exc).__name__, exc,
                    )
                raise
            total_retry_delay += delay
            status_code = getattr(exc, "status_code", None)
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %.1fs: "
                "[%s] %s%s",
                attempt + 1, MAX_RETRIES + 1, delay,
                type(exc).__name__, exc,
                f" (status={status_code})" if status_code else "",
            )
            await asyncio.sleep(delay)

    raise last_exc


def extract_text(response) -> str:
    if not response.choices:
        return ""
    msg = response.choices[0].message
    return msg.content or ""


def extract_tool_calls(response) -> list[dict]:
    if not response.choices:
        return []
    msg = response.choices[0].message
    if not msg.tool_calls:
        return []
    result = []
    for tc in msg.tool_calls:
        if isinstance(tc, dict):
            func = tc.get("function", {})
            tc_id = tc.get("id", "")
            name = func.get("name", "")
            args = func.get("arguments", "{}")
        else:
            tc_id = tc.id
            name = tc.function.name
            args = tc.function.arguments
        if isinstance(args, str):
            raw_args_str = args
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                try:
                    decoder = json.JSONDecoder()
                    prefix, end = decoder.raw_decode(raw_args_str.lstrip())
                    remainder = raw_args_str.lstrip()[end:].lstrip()
                    parsed, parsed_end = decoder.raw_decode(remainder)
                    if prefix != {} or remainder[parsed_end:].strip():
                        raise ValueError("unsupported concatenated JSON")
                    args = parsed
                    logger.info("Stripped empty JSON prefix from tool arguments for '%s'", name)
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Invalid JSON in tool_call arguments for '%s': %s", name, args[:200])
                    args = {"_parse_error": f"Invalid JSON from LLM. Raw text: {args[:500]}"}
            if name in ("set_final_output", "finalize_task") and not args:
                logger.info(
                    "[diag] %s called with empty args. Raw arguments string "
                    "from LLM: %r (len=%d). msg.content len=%d",
                    name, raw_args_str, len(raw_args_str),
                    len(getattr(msg, "content", "") or ""),
                )
        result.append({
            "id": tc_id,
            "name": name,
            "arguments": args,
        })
    return result


def has_tool_calls(response) -> bool:
    if not response.choices:
        return False
    msg = response.choices[0].message
    return bool(msg.tool_calls)


def response_to_message(response) -> dict:
    if not response.choices:
        return {"role": "assistant", "content": ""}
    msg = response.choices[0].message
    result = msg.model_dump(exclude_none=True)

    parsed_by_id = {
        call["id"]: call["arguments"] for call in extract_tool_calls(response)
    }
    for tool_call in result.get("tool_calls", []):
        function = tool_call.get("function", {})
        parsed = parsed_by_id.get(tool_call.get("id"))
        if isinstance(parsed, dict) and isinstance(function.get("arguments"), str):
            function["arguments"] = json.dumps(parsed, ensure_ascii=False)
    return result
