"""Logging helpers for LLM matchup runs."""

from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


###############################################################################
###############################################################################
###############################################################################
class MatchupLogger:
    """Persist per-call LLM traces."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.llm_dir = run_dir / "llm"
        self._lock = threading.Lock()
        self._llm_counter = 0

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.llm_dir.mkdir(parents=True, exist_ok=True)

    ###########################################################################
    @classmethod
    def create_default(cls, logs_root: Path | str = "logs") -> "MatchupLogger":
        """Create logs/<timestamp>/ with the expected file structure."""
        base = Path(logs_root)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir = base / timestamp
        return cls(run_dir)

    ###########################################################################
    def _now(self) -> str:
        return datetime.now().isoformat(timespec="milliseconds")

    ###########################################################################
    def _next_llm_id(self) -> int:
        with self._lock:
            self._llm_counter += 1
            return self._llm_counter

    ###########################################################################
    def start_llm_call(
        self,
        *,
        player_name: str,
        model: str,
        legal_selectors: list[str],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, str]:
        """Create an LLM call log file and return its pointer metadata."""
        call_id = self._next_llm_id()
        pointer = f"LLM-{call_id:06d}"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", player_name).strip("_") or "player"
        filename = f"{pointer}_{safe_name}.log"
        abs_path = self.llm_dir / filename
        rel_path = Path("llm") / filename

        with abs_path.open("w", encoding="utf-8") as handle:
            handle.write(f"pointer: {pointer}\n")
            handle.write(f"created_at: {self._now()}\n")
            handle.write(f"player: {player_name}\n")
            handle.write(f"model: {model}\n")
            handle.write(f"legal_selectors: {', '.join(legal_selectors)}\n")
            handle.write("\n\n=== SYSTEM PROMPT ===\n")
            handle.write(system_prompt)
            handle.write("\n\n=== USER PROMPT ===\n")
            handle.write(user_prompt)
            handle.write("\n")

        return {
            "pointer": pointer,
            "path": str(abs_path),
            "relative_path": rel_path.as_posix(),
        }

    ###########################################################################
    def finish_llm_call(
        self,
        call_ref: Optional[dict[str, str]],
        *,
        selected_selector: Optional[str],
        http_status: str,
        latency_ms: int,
        prompt_tokens: int,
        output_tokens: int,
        thinking_tokens: int,
        output_tokens_est: int,
        thinking_text: str,
        output_text: str,
        response_text: Optional[str],
        error_text: str,
    ) -> None:
        """Finalize a per-call LLM trace file."""
        if not call_ref:
            return

        llm_path = Path(call_ref["path"])
        with llm_path.open("a", encoding="utf-8") as handle:
            handle.write("\n=== RESULT ===\n")
            handle.write(f"selected_selector: {selected_selector if selected_selector is not None else '<fallback>'}\n")
            handle.write(f"http_status: {http_status or '-'}\n")
            handle.write(f"latency_ms: {latency_ms}\n")
            handle.write(f"prompt_tokens: {prompt_tokens}\n")
            handle.write(f"output_tokens: {output_tokens}\n")
            handle.write(f"thinking_tokens_est: {thinking_tokens}\n")
            handle.write(f"output_tokens_est: {output_tokens_est}\n")
            if error_text:
                handle.write(f"error: {error_text}\n")
            handle.write("\n=== THINKING ===\n")
            handle.write(thinking_text or "<none>")
            handle.write("\n\n=== OUTPUT ===\n")
            handle.write(output_text or "<none>")
            handle.write("\n\n=== RAW RESPONSE ===\n")
            handle.write((response_text or "<none>").strip() or "<none>")
            handle.write("\n")
