import numpy as np
import pytest
from fastapi.testclient import TestClient
from signstream.app import app
from signstream.data import fixture, dataset
from signstream.features import features, LABELS, TokenBuffer
from signstream.model import Recognizer


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("label", LABELS)
def test_real_model_recognizes_unseen_fixture(client, label):
    frames = fixture(label, 887766).tolist()
    response = client.post("/api/predict", json={"frames": frames})
    assert response.status_code == 200
    assert response.json()["label"] == label
    assert sum(response.json()["scores"].values()) == pytest.approx(1, abs=1e-5)


def test_held_out_accuracy():
    model = Recognizer().model
    x, y = dataset(5000, 15)
    assert (np.asarray(model(x, training=False)).argmax(axis=1) == y).mean() >= 0.95


def test_translation_and_scale_invariance():
    x = fixture("wave")
    transformed = x.copy()
    valid = np.any(x != 0, axis=2)
    transformed[valid] = x[valid] * 1.2 + np.array([0.1, -0.1, 0])
    np.testing.assert_allclose(features(x), features(transformed), atol=1e-6)


@pytest.mark.parametrize("length", [8, 17, 96])
def test_resampling(length):
    assert features(fixture("circle", count=length)).shape == (32, 8)


@pytest.mark.parametrize("bad", [[], [[[0, 0, 0]]] * 32, "oops"])
def test_invalid_shape_rejected(client, bad):
    assert client.post("/api/predict", json={"frames": bad}).status_code == 422


def test_missing_detection(client):
    assert client.post("/api/predict", json={"frames": np.zeros((32, 543, 3)).tolist()}).status_code == 422
    x = fixture("wave")
    x[:, 522:] = 0
    with pytest.raises(ValueError, match="Right hand"):
        features(x)


def test_nonfinite_and_outlier():
    for value in [float("nan"), float("inf"), 100]:
        x = fixture("wave")
        x[0, 522, 0] = value
        with pytest.raises(ValueError, match="finite"):
            features(x)


def test_fixture_validation_and_ui(client):
    assert client.get("/api/fixture/unknown").status_code == 400
    assert client.get("/api/fixture/wave?seed=-1").status_code == 400
    assert "No validated sign language" in client.get("/").text
    assert len(client.get("/api/fixture/wave").json()["frames"]) == 32


def test_websocket_recovery_reset_and_session_isolation(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("bad json")
        assert "error" in ws.receive_json()
        ws.send_json({"frames": fixture("wave").tolist()})
        assert ws.receive_json()["tokens"] == ["wave"]
        ws.send_json({"frames": fixture("wave").tolist()})
        assert ws.receive_json()["tokens"] == ["wave"]
        with client.websocket_connect("/ws") as other:
            other.send_json({"reset": True})
            assert other.receive_json()["tokens"] == []
        ws.send_json({"reset": True})
        assert ws.receive_json()["tokens"] == []


def test_token_bound_and_neutral():
    buffer = TokenBuffer()
    buffer.accept("wave")
    buffer.accept(None)
    assert buffer.accept("wave") == ["wave", "wave"]
    for i in range(50):
        buffer.accept(LABELS[i % 4])
    assert len(buffer.tokens) == 24


def test_cross_origin_socket_rejected(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "https://untrusted.example"}):
            pass
