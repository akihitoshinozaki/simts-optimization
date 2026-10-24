"""Read-only WebSocket client for the existing POSM server on a Red Pitaya.

This is a communication test, not a hardware experiment driver: it does not
set parameters, trigger captures, or feed real data into the BO campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from time import monotonic
from typing import Any

from websockets.exceptions import WebSocketException
from websockets.sync.client import ClientConnection, connect


DEFAULT_URL = "ws://rp-f0fbe0.local:8000/ws"


class POSMProtocolError(RuntimeError):
    """Unexpected or rejected message from the POSM server."""


class RedPitayaClient:
    """Read-only connection to POSM over Ethernet.

    Use as a context manager to ensure the WebSocket closes after each test::

        with RedPitayaClient() as client:
            print(client.hello)
            print(client.get_all())

    The client always requests the ``viewer`` role and exposes no hardware
    mutation commands. The POSM server (not WebSockets) talks to FPGA registers.
    """

    def __init__(self, url: str = DEFAULT_URL, *, timeout: float = 10.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.url = url
        self.timeout = timeout
        self.hello: dict[str, Any] = {}
        self._ws: ClientConnection | None = None

    def __enter__(self) -> RedPitayaClient:
        self._ws = connect(
            self.url, open_timeout=self.timeout, close_timeout=self.timeout
        )
        try:
            self._send({"type": "hello", "role": "viewer", "name": "simts-optimization"})
            self.hello = self._receive("hello_ack")
            if self.hello.get("role") != "viewer":
                raise POSMProtocolError("POSM did not grant the requested viewer role")
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._ws is not None:
            ws, self._ws = self._ws, None
            ws.close()

    def _send(self, message: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Connect using 'with RedPitayaClient() as client:' first")
        self._ws.send(json.dumps(message))

    def _receive(self, expected_type: str) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Red Pitaya client is not connected")
        deadline = monotonic() + self.timeout
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for POSM {expected_type!r}")
            response = self._ws.recv(timeout=remaining)
            try:
                message = json.loads(response)
            except (TypeError, ValueError) as exc:
                raise POSMProtocolError("POSM returned invalid JSON") from exc
            if not isinstance(message, dict) or not isinstance(message.get("type"), str):
                raise POSMProtocolError("POSM returned an unexpected message format")
            if message["type"] == "error":
                raise POSMProtocolError(f"POSM error: {message}")
            if message["type"] == expected_type:
                return message
            # Status/events may arrive independently of a request; skip them.

    def get_all(self) -> dict[str, Any]:
        """Request the POSM parameter snapshot without changing hardware."""
        self._send({"type": "get_all"})
        return self._receive("snapshot")

    def get(self, parameter: str) -> dict[str, Any]:
        """Read one POSM parameter; returns the unmodified server response."""
        if not parameter:
            raise ValueError("parameter must be nonempty")
        self._send({"type": "get", "param": parameter})
        while True:
            response = self._receive("value")
            if response.get("param") == parameter:
                return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read the Red Pitaya POSM server (viewer only)")
    parser.add_argument("--url", default=DEFAULT_URL, help="POSM WebSocket address")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout in seconds")
    args = parser.parse_args(argv)
    try:
        with RedPitayaClient(args.url, timeout=args.timeout) as client:
            print("Connected:", json.dumps(client.hello))
            print("Parameters:", json.dumps(client.get_all(), indent=2))
    except (OSError, TimeoutError, WebSocketException, POSMProtocolError) as exc:
        print(f"POSM connection failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
