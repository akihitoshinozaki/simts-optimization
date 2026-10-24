"""Local POSM protocol tests; no board or laser required."""

import json

import pytest

from simts_optimization.io.redpitaya_client import POSMProtocolError, RedPitayaClient


class FakeWebSocket:
    def __init__(self, *responses):
        self.responses = [json.dumps(response) for response in responses]
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(json.loads(message))

    def recv(self, *, timeout):
        assert timeout > 0
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def test_viewer_hello_and_read_only_snapshot(monkeypatch):
    ws = FakeWebSocket(
        {"type": "hello_ack", "role": "viewer", "sim_mode": False},
        {"type": "event", "name": "server_started"},
        {"type": "snapshot", "values": {"p_gain": 0.5}},
    )
    monkeypatch.setattr("simts_optimization.io.redpitaya_client.connect", lambda *a, **kw: ws)
    with RedPitayaClient() as client:
        assert client.hello["role"] == "viewer"
        assert client.get_all()["values"]["p_gain"] == 0.5
    assert ws.closed
    assert ws.sent == [
        {"type": "hello", "role": "viewer", "name": "simts-optimization"},
        {"type": "get_all"},
    ]


def test_get_one_parameter(monkeypatch):
    ws = FakeWebSocket(
        {"type": "hello_ack", "role": "viewer"},
        {"type": "value", "param": "p_gain", "value": 0.5},
    )
    monkeypatch.setattr("simts_optimization.io.redpitaya_client.connect", lambda *a, **kw: ws)
    with RedPitayaClient() as client:
        assert client.get("p_gain")["value"] == 0.5
    assert ws.sent[-1] == {"type": "get", "param": "p_gain"}


def test_server_rejection_closes_connection(monkeypatch):
    ws = FakeWebSocket({"type": "error", "message": "viewer denied"})
    monkeypatch.setattr("simts_optimization.io.redpitaya_client.connect", lambda *a, **kw: ws)
    with pytest.raises(POSMProtocolError):
        with RedPitayaClient():
            pass
    assert ws.closed
