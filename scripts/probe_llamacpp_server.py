#!/usr/bin/env python3
"""
Comprueba si el servidor **llama.cpp** (API HTTP nativa ``POST /completion``) responde.

Uso::

    python scripts/probe_llamacpp_server.py
    python scripts/probe_llamacpp_server.py http://127.0.0.1:8080

Variables (opcional, desde ``.env`` cargado manualmente o exportadas)::

    LLAMA_CPP_SERVER_URL   Base sin ``/completion``, ej. ``http://127.0.0.1:8080``
    LLM_BASE_URL           Si termina en ``/v1``, se prueba la misma base sin ``/v1``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Falta httpx: pip install httpx", file=sys.stderr)
    sys.exit(1)


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parent.parent
    env = root / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _resolve_base_url(explicit: str | None) -> str | None:
    if explicit:
        return explicit.rstrip("/")
    u = (os.getenv("LLAMA_CPP_SERVER_URL") or "").strip().rstrip("/")
    if u:
        return u
    bu = (os.getenv("LLM_BASE_URL") or "").strip().rstrip("/")
    if bu.endswith("/v1"):
        return bu[:-3].rstrip("/")
    return None


def probe(base: str, *, n_predict: int = 32, timeout: float = 8.0) -> int:
    """POST /completion mínimo (prompt corto). Devuelve 0 si OK."""
    url = f"{base.rstrip('/')}/completion"
    # ChatML (Qwen / varios modelos en llama.cpp). Si tu modelo usa otro EOT, ajusta ``im_end``.
    im_end = "<|" + "im_end" + "|>"
    prompt = (
        "<|im_start|>user\nDi solo: OK\n" + im_end + "\n"
        "<|im_start|>assistant\n"
    )
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.1,
        "stop": [im_end],
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
    except httpx.ConnectError as e:
        print(f"No hay conexión a {url!r}: {e}")
        return 1
    except httpx.TimeoutException:
        print(f"Timeout al llamar a {url!r}")
        return 1
    except OSError as e:
        print(f"Error de red: {e}")
        return 1

    if r.status_code != 200:
        print(f"HTTP {r.status_code} en {url}")
        try:
            print(r.text[:500])
        except Exception:
            pass
        return 1

    try:
        data = r.json()
    except json.JSONDecodeError:
        print("Respuesta no JSON")
        print(r.text[:500])
        return 1

    content = (data.get("content") or "").strip()
    print(f"OK — {url}")
    print(f"  content (primeros 200 chars): {content[:200]!r}")
    if data.get("truncated"):
        print("  truncated: true")
    return 0


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Probar servidor llama.cpp /completion")
    ap.add_argument(
        "base_url",
        nargs="?",
        default=None,
        help="Base URL del server (sin /completion). Si se omite, usa env o LLM_BASE_URL.",
    )
    args = ap.parse_args()
    base = _resolve_base_url(args.base_url)
    if not base:
        print(
            "Indica la URL base del server llama.cpp, p. ej.:\n"
            "  python scripts/probe_llamacpp_server.py http://127.0.0.1:8080\n"
            "O define LLAMA_CPP_SERVER_URL en .env",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(probe(base))


if __name__ == "__main__":
    main()
