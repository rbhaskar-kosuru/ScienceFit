"""Blank the .env settings before src.config loads them: tests use the code defaults and never hold a real key.
And no test can reach the network: only this machine answers."""
import os
import socket

import pytest

BLANKED = ("LLM_BASE_URL", "LLM_MODEL", "LLM_PROMPT_VERSION", "LLM_API_KEY", "LLM_EXTRA_PARAMS",
           "OPENALEX_API_KEY", "OPENALEX_MAILTO")
for name in BLANKED:
    os.environ[name] = ""

LOCAL = {"localhost", "127.0.0.1", "::1"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    connect, lookup = socket.socket.connect, socket.getaddrinfo

    def guarded_connect(self, address):
        if isinstance(address, tuple) and address[0] not in LOCAL:
            raise OSError("network blocked in tests")
        return connect(self, address)

    def guarded_lookup(host, *args, **kwargs):
        if host not in LOCAL:
            raise OSError("network blocked in tests")
        return lookup(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_lookup)
