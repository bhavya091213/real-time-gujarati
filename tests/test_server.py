import pytest
from fastapi.testclient import TestClient

from gujusub import server
from gujusub.streaming import TranscriptEvent


@pytest.fixture
def client():
    return TestClient(server.app)


def test_display_page_served(client):
    r = client.get("/display")
    assert r.status_code == 200
    assert "ws/view" in r.text


def test_filtered_strips_fillers():
    ev = TranscriptEvent("partial", 1, "um hello", "there uh")
    out = server.filtered(ev)
    assert (out.committed, out.tail) == ("hello", "there")
    assert out.utterance_id == 1


def test_filtered_respects_no_filter(monkeypatch):
    monkeypatch.setattr(server, "filter_fillers", False)
    ev = TranscriptEvent("partial", 1, "um hello", "")
    assert server.filtered(ev) is ev


def test_viewer_receives_broadcast(client):
    with client, client.websocket_connect("/ws/view") as viewer:
        assert len(server.viewers) == 1
        payload = {"type": "final", "utterance_id": 1, "committed": "hi", "tail": ""}
        client.portal.call(server.broadcast, payload)
        assert viewer.receive_json() == payload
    assert len(server.viewers) == 0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("This is my project", False),
        ("આ મારું પ્રોજેક્ટ છે અને translation.", True),
        ("My name is ભવ્ય", False),
        ("", False),
        ("...", False),
    ],
)
def test_looks_untranslated(text, expected):
    from gujusub.translator import looks_untranslated

    assert looks_untranslated(text) is expected
