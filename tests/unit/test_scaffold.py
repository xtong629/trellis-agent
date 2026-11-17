"""M0 sanity checks: package imports, and the test environment cannot reach the network."""

import socket

import pytest
from pytest_socket import SocketBlockedError

import trellis_agent


def test_package_exposes_version() -> None:
    assert trellis_agent.__version__ == "0.0.1"


def test_subpackages_import() -> None:
    import trellis_agent.eval
    import trellis_agent.llm
    import trellis_agent.memory
    import trellis_agent.planner
    import trellis_agent.retrieval  # noqa: F401


def test_external_network_is_blocked() -> None:
    # Guards the "CI never calls a real API" promise (pytest-socket via addopts).
    # pytest-socket warns before raising in --allow-hosts mode, and filterwarnings=error turns the
    # warning into UserWarning first; either outcome means the connection was blocked.
    # 192.0.2.1 is TEST-NET-1 (RFC 5737): no DNS lookup, never routable.
    with pytest.raises((SocketBlockedError, UserWarning)):
        socket.create_connection(("192.0.2.1", 80), timeout=1)
