# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator

import httpx2
import pytest

from huggingface_hub._hot_reload import client as client_module
from huggingface_hub._hot_reload.client import ReloadClient, multi_replica_reload_events


def _mock_replica(monkeypatch: pytest.MonkeyPatch, statuses: list[int]) -> list[str]:
    requests: list[str] = []
    responses = iter(statuses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.url.path)
        status = next(responses)
        content = b'data: {"kind": "done"}\n\n' if status == 200 else b""
        return httpx2.Response(status, content=content)

    original_client = httpx2.Client

    def fake_client(**kwargs):
        return original_client(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(client_module.httpx2, "Client", fake_client)
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    return requests


@pytest.mark.parametrize("status", [204, 404, 500])
def test_get_reload_returns_non_200_status(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    _mock_replica(monkeypatch, [status])
    client = ReloadClient(host="https://user-space.hf.space", subdomain="user-space", replica_hash="abc", token=None)

    assert client.get_reload("deadbeef") == status


def test_get_reload_streams_200_events(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_replica(monkeypatch, [200])
    client = ReloadClient(host="https://user-space.hf.space", subdomain="user-space", replica_hash="abc", token=None)

    events = client.get_reload("deadbeef")
    assert isinstance(events, Iterator)
    assert list(events) == [{"kind": "done"}]


def test_multi_replica_retries_204_then_reads_events(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _mock_replica(monkeypatch, [204, 200])

    events = list(
        multi_replica_reload_events(
            commit_sha="deadbeef",
            host="https://user-space.hf.space",
            subdomain="user-space",
            replica_hashes=["abc"],
            token=None,
        )
    )

    assert [event["kind"] for event in events] == ["warning", "event"]
    assert len(requests) == 2


def test_multi_replica_stops_after_retry_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _mock_replica(monkeypatch, [204, 204, 204])

    with pytest.raises(Exception, match="Too many retries reached"):
        list(
            multi_replica_reload_events(
                commit_sha="deadbeef",
                host="https://user-space.hf.space",
                subdomain="user-space",
                replica_hashes=["abc"],
                token=None,
                max_retries=2,
            )
        )
    assert len(requests) == 3


def test_multi_replica_rejects_unexpected_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_replica(monkeypatch, [404])

    with pytest.raises(Exception, match="status_code=404"):
        list(
            multi_replica_reload_events(
                commit_sha="deadbeef",
                host="https://user-space.hf.space",
                subdomain="user-space",
                replica_hashes=["abc"],
                token=None,
            )
        )
