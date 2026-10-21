"""Host-only boundaries for preparation that retains frozen program owners."""
import threading

import pytest

from b12x.preparation import DetectedDevice, PreparationSession
from b12x.preparation.resume import additional_preparation
from b12x._lib.runtime_control import kernel_resolution_guard, kernel_resolution_frozen


@pytest.fixture
def session():
    retained = PreparationSession(device=DetectedDevice(None, None), compile_workers=0)
    retained.freeze()
    yield retained
    retained.close()


def test_scope_retains_frozen_state_and_payload_collections(session):
    plans, shared = session._plans, session._shared
    payload = object()
    shared['fixture'] = payload
    assert kernel_resolution_frozen()
    with additional_preparation(session):
        assert not kernel_resolution_frozen()
        assert session.state == 'FROZEN'
        assert session._plans is plans and session._shared is shared
        assert shared['fixture'] is payload
    assert kernel_resolution_frozen()
    assert session._plans is plans and shared['fixture'] is payload


def test_exception_restores_guard(session):
    with pytest.raises(ValueError, match='fixture error'):
        with additional_preparation(session):
            raise ValueError('fixture error')
    assert kernel_resolution_frozen() and session.state == 'FROZEN'


def test_other_session_and_capture_guards_remain_active(session):
    other = PreparationSession(device=DetectedDevice(None, None), compile_workers=0)
    other.freeze()
    try:
        with additional_preparation(session):
            assert kernel_resolution_frozen()
        with additional_preparation(session, other):
            assert not kernel_resolution_frozen()
            with kernel_resolution_guard('fixture capture'):
                assert kernel_resolution_frozen()
        assert kernel_resolution_frozen()
    finally:
        other.close()


def test_all_owners_validated_before_any_guard_is_removed(session):
    with pytest.raises(TypeError):
        with additional_preparation(session, object()):
            pytest.fail('invalid owner entered scope')
    assert kernel_resolution_frozen() and session._guard is not None


def test_reentry_and_duplicate_sessions_are_rejected(session):
    with pytest.raises(ValueError):
        with additional_preparation(session, session):
            pass
    with additional_preparation(session):
        with pytest.raises(RuntimeError, match='active preparation scope'):
            with additional_preparation(session):
                pass
    assert kernel_resolution_frozen()


@pytest.mark.parametrize('field', ['_job', '_pool', '_pending_collective_barrier'])
def test_pending_preparation_rejected_without_changing_guard(session, field):
    setattr(session, field, threading.Event() if field == '_pending_collective_barrier' else object())
    try:
        with pytest.raises(RuntimeError, match='drained'):
            with additional_preparation(session):
                pass
        assert kernel_resolution_frozen()
    finally:
        setattr(session, field, None)


def test_owning_thread_required(session):
    errors = []
    def enter():
        try:
            with additional_preparation(session):
                pass
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=enter)
    thread.start(); thread.join()
    assert len(errors) == 1 and 'another thread' in errors[0]
    assert kernel_resolution_frozen()


def test_open_and_closed_sessions_rejected():
    retained = PreparationSession(device=DetectedDevice(None, None), compile_workers=0)
    with pytest.raises(RuntimeError, match='frozen'):
        with additional_preparation(retained):
            pass
    retained.close()
    with pytest.raises(RuntimeError, match='closed'):
        with additional_preparation(retained):
            pass


def test_closed_session_does_not_receive_a_replacement_guard(session):
    with pytest.raises(RuntimeError, match='closed inside'):
        with additional_preparation(session):
            session.close()
    assert session.state == 'CLOSED' and session._guard is None
    assert not kernel_resolution_frozen()


def test_completed_collective_barrier_is_drained(session):
    completed = threading.Event()
    completed.set()
    session._pending_collective_barrier = completed
    with additional_preparation(session):
        assert not kernel_resolution_frozen()
        assert session._pending_collective_barrier is completed
    assert kernel_resolution_frozen()
