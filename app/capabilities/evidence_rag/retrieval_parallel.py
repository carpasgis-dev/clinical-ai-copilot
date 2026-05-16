"""
Recuperación de evidencia en paralelo (``asyncio.gather`` + semáforo de concurrencia).

Activa con ``COPILOT_PARALLEL_RETRIEVAL=1`` (por defecto). Limita peticiones simultáneas
con ``COPILOT_RETRIEVAL_MAX_PARALLEL`` (default 8) para no saturar NCBI / Europe PMC.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TypeVar

T = TypeVar("T")


def parallel_retrieval_enabled() -> bool:
    v = (os.getenv("COPILOT_PARALLEL_RETRIEVAL") or "1").strip().lower()
    return v not in ("0", "false", "no", "off", "disabled")


def retrieval_max_parallel() -> int:
    try:
        return max(1, min(32, int(os.getenv("COPILOT_RETRIEVAL_MAX_PARALLEL", "8"))))
    except (TypeError, ValueError):
        return 8


def run_coroutine_sync(coro: Awaitable[T]) -> T:
    """Ejecuta una corrutina desde código síncrono (FastAPI sync, LangGraph)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def gather_limited(
    awaitables: list[Awaitable[T]],
    *,
    limit: int,
) -> list[T]:
    sem = asyncio.Semaphore(max(1, limit))

    async def _wrap(aw: Awaitable[T]) -> T:
        async with sem:
            return await aw

    return list(await asyncio.gather(*(_wrap(a) for a in awaitables)))


async def gather_sync_calls(
    callables: list[Callable[[], T]],
    *,
    limit: int,
) -> list[T]:
    """``asyncio.gather`` sobre funciones bloqueantes (I/O HTTP síncrono existente)."""

    async def _one(fn: Callable[[], T]) -> T:
        return await asyncio.to_thread(fn)

    return await gather_limited([_one(fn) for fn in callables], limit=limit)


def gather_sync_calls_blocking(
    callables: list[Callable[[], T]],
    *,
    limit: int | None = None,
) -> list[T]:
    lim = limit if limit is not None else retrieval_max_parallel()
    return run_coroutine_sync(gather_sync_calls(callables, limit=lim))


def partial_call(fn: Callable[..., T], /, *args, **kwargs) -> Callable[[], T]:
    return partial(fn, *args, **kwargs)
