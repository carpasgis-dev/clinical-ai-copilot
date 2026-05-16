"""Errores con código estable para logs y respuesta HTTP (sin fallbacks silenciosos)."""
from __future__ import annotations


class CopilotError(Exception):
    """Fallo controlado: ``code`` estable (snake UPPER), ``message`` legible."""

    def __init__(self, code: str, message: str, *, cause: BaseException | None = None) -> None:
        self.code = (code or "UNKNOWN").strip().upper()
        self.message = (message or "").strip() or self.code
        super().__init__(f"[{self.code}] {self.message}")
        self.__cause__ = cause
