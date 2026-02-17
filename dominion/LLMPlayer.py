"""Dominion player backed by a configurable LLM provider."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

from dominion import Phase, Prompt
from dominion.Option import Option
from dominion.TextPlayer import TextPlayer

if TYPE_CHECKING:
    from dominion.Game import Game


###############################################################################
###############################################################################
###############################################################################
class LLMPlayer(TextPlayer):
    """Player implementation that uses a configured LLM provider to pick options."""

    def __init__(self, game: "Game", name: str = "", quiet: bool = False, **kwargs: Any) -> None:
        self.llm_provider = str(kwargs.pop("llm_provider", "") or "").strip().lower()
        self.llm_model = str(kwargs.pop("llm_model", "") or "").strip()

        legacy_ollama_model = kwargs.pop("ollama_model", None)
        legacy_openrouter_model = kwargs.pop("openrouter_model", None)
        if not self.llm_model and legacy_ollama_model:
            self.llm_model = str(legacy_ollama_model).strip()
        if not self.llm_model and legacy_openrouter_model:
            self.llm_model = str(legacy_openrouter_model).strip()

        if not self.llm_provider:
            if legacy_openrouter_model and not legacy_ollama_model:
                self.llm_provider = "openrouter"
            else:
                self.llm_provider = "ollama"

        self.ollama_url = str(kwargs.pop("ollama_url", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").rstrip(
            "/"
        )
        self.openrouter_url = str(
            kwargs.pop("openrouter_url", "https://openrouter.ai/api/v1") or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.openrouter_api_key = str(kwargs.pop("openrouter_api_key", "") or os.getenv("OPENROUTER_API_KEY", "")).strip()
        self.openrouter_referer = str(kwargs.pop("openrouter_referer", "") or "").strip()
        self.openrouter_title = str(kwargs.pop("openrouter_title", "pydominion") or "").strip()

        gameengine_prompt_path = kwargs.pop("llm_gameengine_prompt_path", kwargs.pop("ollama_gameengine_prompt_path", None))
        if gameengine_prompt_path is None:
            self.llm_gameengine_prompt_path = str(Path(__file__).resolve().parents[1] / "gameengine.prompt")
        else:
            self.llm_gameengine_prompt_path = str(gameengine_prompt_path)
        self.llm_gameengine_prompt = self._load_prompt(self.llm_gameengine_prompt_path)

        prompt_path = kwargs.pop("llm_gameplay_prompt_path", kwargs.pop("ollama_gameplay_prompt_path", None))
        if prompt_path is None:
            self.llm_gameplay_prompt_path = str(Path(__file__).resolve().parents[1] / "gameplay.prompt")
        else:
            self.llm_gameplay_prompt_path = str(prompt_path)
        self.llm_gameplay_prompt = self._load_prompt(self.llm_gameplay_prompt_path)

        self.llm_temperature = float(kwargs.pop("llm_temperature", kwargs.pop("ollama_temperature", 0.1)))
        self.llm_auto_spend_all_treasures = self._coerce_bool(
            kwargs.pop(
                "llm_auto_spend_all_treasures",
                kwargs.pop("ollama_auto_spend_all_treasures", kwargs.pop("auto_spend_all_treasures", False)),
            )
        )

        self.llm_calls = 0
        self.llm_failures = 0
        self.llm_last_error = ""
        self.llm_last_http_status = ""
        self.llm_last_latency_ms = -1
        self.llm_last_thinking_full = ""
        self.llm_last_output_full = ""
        self.llm_last_prompt_tokens = 0
        self.llm_last_eval_tokens = 0
        self.llm_last_thinking_tokens_est = 0
        self.llm_last_output_tokens_est = 0
        self.llm_total_prompt_tokens = 0
        self.llm_total_eval_tokens = 0
        self.llm_total_thinking_tokens_est = 0
        self.llm_total_output_tokens_est = 0
        self.llm_failure_reasons: dict[str, int] = {}

        self._sync_legacy_ollama_compat()
        super().__init__(game, name=name, quiet=quiet, **kwargs)

    ###########################################################################
    def user_input(self, options: list[Option], prompt: str) -> Option:
        """Pick one option, usually by querying the configured LLM provider."""
        for opt in options:
            assert isinstance(opt, Option), f"user_input {opt=} {type(opt)=}"

        available = [opt for opt in options if opt["selector"] not in (None, "", "-")]
        if not available:
            raise RuntimeError("LLMPlayer received no selectable options")
        spend_all = self._select_spend_all_treasures(available)
        if spend_all is not None:
            return spend_all
        if len(available) == 1:
            return available[0]

        selector = self._query_selector(prompt, available, options)
        if selector is None:
            detail = self.llm_last_error or "no selector produced"
            raise RuntimeError(
                f"LLM did not return a legal selector "
                f"(provider={self.llm_provider} model={self.llm_model} detail={detail})"
            )
        for opt in available:
            if str(opt["selector"]) == selector:
                return opt

        raise RuntimeError(
            f"LLM returned unknown selector {selector!r} "
            f"(provider={self.llm_provider} model={self.llm_model})"
        )

    ###########################################################################
    def _query_selector(
        self, prompt: str, legal_options: list[Option], display_options: Optional[list[Option]] = None
    ) -> Optional[str]:
        selectors = [str(opt["selector"]) for opt in legal_options]
        if not selectors:
            return None
        selectors_text = ",".join(selectors)

        system_prompt = (
            f"{self._system_bootstrap()}\n\n"
            "Decision protocol for this move:\n"
            "- Pick exactly one legal selector from the list.\n"
            "- Return JSON only in this format: {\"selector\":\"x\"}\n"
            "- Do not include any explanation or extra keys."
        )

        shown_options = display_options if display_options is not None else legal_options
        user_prompt = self._textplayer_style_prompt(prompt, shown_options, selectors)

        llm_call_ref = self._start_matchup_llm_log(
            prompt=prompt,
            selectors=selectors,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        response = self._call_llm(system_prompt, user_prompt)
        selector: Optional[str] = None
        if response is None:
            self._finish_matchup_llm_log(llm_call_ref, response=None, selector=None)
            return None
        response_preview = self._truncate(response)
        selector = self._extract_selector(response, selectors)
        if selector is None:
            self._record_failure(
                "selector_parse",
                f"response={response_preview} legal={self._truncate(selectors_text, 80)}",
            )
        else:
            self.llm_last_error = ""
            self._sync_legacy_ollama_compat()
        self._finish_matchup_llm_log(llm_call_ref, response=response, selector=selector)
        return selector

    ###########################################################################
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        self._reset_last_call_stats()

        if not self.llm_model:
            self._record_failure("missing_model")
            return None

        self.llm_calls += 1
        provider = self.llm_provider.strip().lower()
        if provider == "ollama":
            response = self._call_ollama(system_prompt, user_prompt)
        elif provider == "openrouter":
            response = self._call_openrouter(system_prompt, user_prompt)
        else:
            self._record_failure("unsupported_provider", provider)
            return None

        self._sync_legacy_ollama_compat()
        return response

    ###########################################################################
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        payload = {
            "model": self.llm_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": self.llm_temperature},
        }
        request = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        body = ""
        start = time.monotonic()
        try:
            response_stream = urllib.request.urlopen(request)
            with response_stream as response:
                self.llm_last_http_status = str(response.status)
                body = response.read().decode("utf-8")
            self.llm_last_latency_ms = int((time.monotonic() - start) * 1000)
            parsed = json.loads(body)
            self.llm_last_prompt_tokens = int(parsed.get("prompt_eval_count", 0) or 0)
            self.llm_last_eval_tokens = int(parsed.get("eval_count", 0) or 0)
            self.llm_total_prompt_tokens += self.llm_last_prompt_tokens
            self.llm_total_eval_tokens += self.llm_last_eval_tokens
            try:
                message = parsed["message"]
                content = message["content"]
            except (KeyError, TypeError) as exc:
                self._record_failure("response_parse", f"{exc}; body={self._truncate(body)}")
                return None
            if not isinstance(content, str):
                self._record_failure("missing_content", f"body={self._truncate(body)}")
                return None
            thinking = message.get("thinking", "")
            if thinking is None:
                thinking = ""
            elif not isinstance(thinking, str):
                thinking = str(thinking)
            self.llm_last_thinking_tokens_est = self._token_estimate(thinking)
            self.llm_last_output_tokens_est = self._token_estimate(content)
            self.llm_total_thinking_tokens_est += self.llm_last_thinking_tokens_est
            self.llm_total_output_tokens_est += self.llm_last_output_tokens_est
            self.llm_last_thinking_full = thinking
            self.llm_last_output_full = content
            self.llm_last_error = ""
            return content.strip()
        except urllib.error.HTTPError as exc:
            resp = ""
            try:
                resp = exc.read().decode("utf-8", errors="replace")
            except OSError:
                pass
            self.llm_last_http_status = str(exc.code)
            self.llm_last_latency_ms = int((time.monotonic() - start) * 1000)
            self._record_failure("http_error", f"status={exc.code} body={self._truncate(resp)}")
            return None
        except urllib.error.URLError as exc:
            self.llm_last_latency_ms = int((time.monotonic() - start) * 1000)
            self._record_failure("url_error", self._truncate(str(exc.reason)))
            return None
        except json.JSONDecodeError as exc:
            self.llm_last_latency_ms = int((time.monotonic() - start) * 1000)
            self._record_failure("json_decode", f"{exc.msg}; body={self._truncate(body)}")
            return None
        except OSError as exc:
            self.llm_last_latency_ms = int((time.monotonic() - start) * 1000)
            self._record_failure("os_error", self._truncate(str(exc)))
            return None

    ###########################################################################
    def _call_openrouter(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.openrouter_api_key:
            self._record_failure("missing_api_key", "OPENROUTER_API_KEY")
            return None

        payload = {
            "model": self.llm_model,
            "stream": False,
            "temperature": self.llm_temperature,
            "reasoning": {"enabled": True},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        endpoint = self.openrouter_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openrouter_api_key}",
        }
        if self.openrouter_referer:
            headers["HTTP-Referer"] = self.openrouter_referer
        if self.openrouter_title:
            headers["X-Title"] = self.openrouter_title

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        body = ""
        start = time.monotonic()
        try:
            response_stream = urllib.request.urlopen(request)
            with response_stream as response:
                self.llm_last_http_status = str(response.status)
                body = response.read().decode("utf-8")
            self.llm_last_latency_ms = int((time.monotonic() - start) * 1000)
            parsed = json.loads(body)

            usage = parsed.get("usage", {})
            reasoning_tokens = 0
            if isinstance(usage, dict):
                self.llm_last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                self.llm_last_eval_tokens = int(usage.get("completion_tokens", 0) or 0)
                reasoning_tokens = self._extract_reasoning_tokens(usage)
                self.llm_total_prompt_tokens += self.llm_last_prompt_tokens
                self.llm_total_eval_tokens += self.llm_last_eval_tokens

            try:
                choice = parsed["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError) as exc:
                self._record_failure("response_parse", f"{exc}; body={self._truncate(body)}")
                return None

            thinking = self._extract_reasoning(message, choice)
            content = message.get("content", "")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                content = "".join(parts)

            if not isinstance(content, str):
                self._record_failure("missing_content", f"body={self._truncate(body)}")
                return None

            self.llm_last_thinking_tokens_est = reasoning_tokens if reasoning_tokens > 0 else self._token_estimate(thinking)
            self.llm_last_output_tokens_est = self._token_estimate(content)
            self.llm_total_thinking_tokens_est += self.llm_last_thinking_tokens_est
            self.llm_total_output_tokens_est += self.llm_last_output_tokens_est
            self.llm_last_thinking_full = thinking
            self.llm_last_output_full = content
            self.llm_last_error = ""
            return content.strip()
        except urllib.error.HTTPError as exc:
            resp = ""
            try:
                resp = exc.read().decode("utf-8", errors="replace")
            except OSError:
                pass
            self.llm_last_http_status = str(exc.code)
            self.llm_last_latency_ms = int((time.monotonic() - start) * 1000)
            self._record_failure("http_error", f"status={exc.code} body={self._truncate(resp)}")
            return None
        except urllib.error.URLError as exc:
            self.llm_last_latency_ms = int((time.monotonic() - start) * 1000)
            self._record_failure("url_error", self._truncate(str(exc.reason)))
            return None
        except json.JSONDecodeError as exc:
            self.llm_last_latency_ms = int((time.monotonic() - start) * 1000)
            self._record_failure("json_decode", f"{exc.msg}; body={self._truncate(body)}")
            return None
        except OSError as exc:
            self.llm_last_latency_ms = int((time.monotonic() - start) * 1000)
            self._record_failure("os_error", self._truncate(str(exc)))
            return None

    ###########################################################################
    @staticmethod
    def _extract_reasoning(message: dict[str, Any], choice: dict[str, Any]) -> str:
        reasoning_parts: list[str] = []
        for key in ("reasoning", "thinking", "reasoning_content"):
            value = message.get(key)
            if isinstance(value, str):
                reasoning_parts.append(value)
        for key in ("reasoning", "thinking", "reasoning_content"):
            value = choice.get(key)
            if isinstance(value, str):
                reasoning_parts.append(value)

        if not reasoning_parts:
            return ""
        return "\n\n".join(part for part in reasoning_parts if part)

    ###########################################################################
    @staticmethod
    def _extract_reasoning_tokens(usage: dict[str, Any]) -> int:
        direct = usage.get("reasoning_tokens")
        if isinstance(direct, int):
            return max(0, direct)

        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            value = details.get("reasoning_tokens")
            if isinstance(value, int):
                return max(0, value)
        return 0

    ###########################################################################
    def _extract_selector(self, text: str, selectors: list[str]) -> Optional[str]:
        stripped = text.strip()
        if not stripped:
            return None

        if stripped in selectors:
            return stripped

        wrapped = stripped.strip("`\"' ")
        if wrapped in selectors:
            return wrapped

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                selector = parsed.get("selector")
                if selector is not None:
                    selector_str = str(selector).strip()
                    if selector_str in selectors:
                        return selector_str
            elif isinstance(parsed, str):
                parsed_str = parsed.strip()
                if parsed_str in selectors:
                    return parsed_str
        except json.JSONDecodeError:
            pass

        matched = re.search(r"selector\s*[:=]\s*['\"]?([^'\"\s,}]+)", stripped, flags=re.IGNORECASE)
        if matched:
            selector_str = matched.group(1).strip()
            if selector_str in selectors:
                return selector_str

        for selector in selectors:
            if f'"{selector}"' in stripped or f"'{selector}'" in stripped or f"`{selector}`" in stripped:
                return selector

        return None

    ###########################################################################
    def _select_spend_all_treasures(self, legal_options: list[Option]) -> Optional[Option]:
        if not self.llm_auto_spend_all_treasures or self.phase != Phase.BUY:
            return None
        for opt in legal_options:
            if opt["verb"] == "Spend all treasures":
                return opt
        return None

    ###########################################################################
    def _textplayer_style_prompt(self, prompt: str, options: list[Option], legal_selectors: list[str]) -> str:
        stats = f"({self.get_score()} points, {self.count_cards()} cards)"
        phase_name = self.phase.name.title()
        lines = [
            f"{'#' * 30} Turn {self.turn_number} {'#' * 30}",
            f"{self.name}'s Turn {stats}",
            f"************ {phase_name} Phase ************",
            "",
            *Prompt.overview_lines(self),
            "",
            *self._purchase_history_lines(),
        ]
        for opt in options:
            lines.append(self.selector_line(opt))
        lines.append(prompt)
        lines.append(f"Legal selectors: {', '.join(legal_selectors)}")
        lines.append("Choose one selector now.")
        return "\n".join(lines)

    ###########################################################################
    def _purchase_history_lines(self) -> list[str]:
        history = list(getattr(self.game, "purchase_history", []))
        lines = ["Purchase history (all turns, oldest to newest):"]
        if not history:
            lines.append("- None yet")
            return lines

        for turn_number, player_name, card_name in history:
            lines.append(f"- T{turn_number} {player_name}: {card_name}")
        return lines

    ###########################################################################
    def _record_failure(self, reason: str, detail: str = "") -> None:
        self.llm_failures += 1
        self.llm_failure_reasons[reason] = self.llm_failure_reasons.get(reason, 0) + 1
        if detail:
            self.llm_last_error = f"{reason}: {detail}"
        else:
            self.llm_last_error = reason
        self._sync_legacy_ollama_compat()

    ###########################################################################
    def _reset_last_call_stats(self) -> None:
        self.llm_last_error = ""
        self.llm_last_http_status = ""
        self.llm_last_latency_ms = -1
        self.llm_last_prompt_tokens = 0
        self.llm_last_eval_tokens = 0
        self.llm_last_thinking_tokens_est = 0
        self.llm_last_output_tokens_est = 0
        self.llm_last_thinking_full = ""
        self.llm_last_output_full = ""
        self._sync_legacy_ollama_compat()

    ###########################################################################
    def _sync_legacy_ollama_compat(self) -> None:
        """Keep legacy Ollama attribute names populated for existing scripts."""
        self.ollama_model = self.llm_model
        self.ollama_temperature = self.llm_temperature
        self.ollama_gameengine_prompt_path = self.llm_gameengine_prompt_path
        self.ollama_gameengine_prompt = self.llm_gameengine_prompt
        self.ollama_gameplay_prompt_path = self.llm_gameplay_prompt_path
        self.ollama_gameplay_prompt = self.llm_gameplay_prompt
        self.ollama_auto_spend_all_treasures = self.llm_auto_spend_all_treasures

        self.ollama_calls = self.llm_calls
        self.ollama_failures = self.llm_failures
        self.ollama_last_error = self.llm_last_error
        self.ollama_last_http_status = self.llm_last_http_status
        self.ollama_last_latency_ms = self.llm_last_latency_ms
        self.ollama_last_thinking_full = self.llm_last_thinking_full
        self.ollama_last_output_full = self.llm_last_output_full
        self.ollama_last_prompt_tokens = self.llm_last_prompt_tokens
        self.ollama_last_eval_tokens = self.llm_last_eval_tokens
        self.ollama_last_thinking_tokens_est = self.llm_last_thinking_tokens_est
        self.ollama_last_output_tokens_est = self.llm_last_output_tokens_est
        self.ollama_total_prompt_tokens = self.llm_total_prompt_tokens
        self.ollama_total_eval_tokens = self.llm_total_eval_tokens
        self.ollama_total_thinking_tokens_est = self.llm_total_thinking_tokens_est
        self.ollama_total_output_tokens_est = self.llm_total_output_tokens_est
        self.ollama_failure_reasons = self.llm_failure_reasons

    ###########################################################################
    def _system_bootstrap(self) -> str:
        prompt_sections = []
        if self.llm_gameengine_prompt:
            prompt_sections.append(self.llm_gameengine_prompt)
        if self.llm_gameplay_prompt:
            prompt_sections.append(self.llm_gameplay_prompt)
        if prompt_sections:
            return "\n\n".join(prompt_sections)
        return (
            "You are a Dominion bot. Make legal moves and maximize end-game VP. "
            "If card text conflicts with default rules, follow card text."
        )

    ###########################################################################
    @staticmethod
    def _load_prompt(path: str) -> str:
        try:
            content = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        return content

    ###########################################################################
    @staticmethod
    def _truncate(text: str, size: int = 180) -> str:
        s = str(text)
        if size <= 0:
            return s
        if len(s) <= size:
            return s
        return s[: size - 3] + "..."

    ###########################################################################
    @staticmethod
    def _token_estimate(text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\S+", text))

    ###########################################################################
    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "f", "no", "n", "off", ""}:
                return False
        return bool(value)

    ###########################################################################
    def _start_matchup_llm_log(
        self,
        prompt: str,
        selectors: list[str],
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[dict[str, str]]:
        logger = getattr(self.game, "matchup_logger", None)
        if logger is None:
            return None
        try:
            return logger.start_llm_call(
                player_name=self.name,
                model=self.llm_model,
                prompt=prompt,
                legal_selectors=selectors,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except OSError:
            return None

    ###########################################################################
    def _finish_matchup_llm_log(
        self,
        llm_call_ref: Optional[dict[str, str]],
        response: Optional[str],
        selector: Optional[str],
    ) -> None:
        logger = getattr(self.game, "matchup_logger", None)
        if logger is None:
            return
        try:
            logger.finish_llm_call(
                llm_call_ref,
                selected_selector=selector,
                http_status=self.llm_last_http_status,
                latency_ms=self.llm_last_latency_ms,
                prompt_tokens=self.llm_last_prompt_tokens,
                output_tokens=self.llm_last_eval_tokens,
                thinking_tokens=self.llm_last_thinking_tokens_est,
                output_tokens_est=self.llm_last_output_tokens_est,
                thinking_text=self.llm_last_thinking_full,
                output_text=self.llm_last_output_full,
                response_text=response,
                error_text=self.llm_last_error,
            )
        except OSError:
            pass


# EOF
