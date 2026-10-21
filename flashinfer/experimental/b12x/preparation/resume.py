"""Prepare additional programs while retaining explicitly drained sessions."""

from contextlib import contextmanager
from threading import Lock
from weakref import WeakSet

from b12x._lib.runtime_control import kernel_resolution_guard
from .session import PreparationSession

_ACTIVE = WeakSet()
_LOCK = Lock()


@contextmanager
def additional_preparation(*sessions: PreparationSession):
    """Temporarily release only the named sessions' kernel-resolution guards.

    The caller must first stop native execution and confirm every participating
    stream has drained. This function cannot establish external CUDA ownership.
    Retained sessions must be frozen, idle and owned by the calling thread. Their
    plans, program leases and frozen state remain unchanged; declare additional
    work in a separate session. Do not close or retune retained sessions inside
    this scope. Capture guards and guards from unnamed sessions remain active.

    Every released guard is restored when the scope exits, including when
    preparation raises. This is a preparation boundary, not a serving fallback.
    """
    if not sessions or len({id(session) for session in sessions}) != len(sessions):
        raise ValueError("additional preparation requires distinct retained sessions")
    with _LOCK:
        for session in sessions:
            if not isinstance(session, PreparationSession):
                raise TypeError("retained owner must be a PreparationSession")
            session._check_thread()
            if session in _ACTIVE:
                raise RuntimeError("retained session already has an active preparation scope")
            if session.state != "FROZEN" or session._guard is None:
                raise RuntimeError("additional preparation requires a frozen guarded session")
            pending = session._pending_collective_barrier
            if (session._job is not None or session._pool is not None
                    or pending is not None and not pending.is_set()):
                raise RuntimeError("additional preparation requires drained jobs and compiler pools")
        _ACTIVE.update(sessions)
    released = []
    try:
        for session in sessions:
            guard = session._guard
            guard.__exit__(None, None, None)
            session._guard = None
            released.append(session)
        yield
    finally:
        try:
            invalid = []
            for session in reversed(released):
                if session.state == "CLOSED":
                    invalid.append("retained session was closed inside preparation scope")
                    continue
                if session.state != "FROZEN":
                    invalid.append("retained session changed state inside preparation scope")
                if session._guard is None:
                    guard = kernel_resolution_guard("prepared serving session")
                    guard.__enter__()
                    session._guard = guard
            if invalid:
                raise RuntimeError("; ".join(invalid))
        finally:
            with _LOCK:
                for session in sessions:
                    _ACTIVE.discard(session)
