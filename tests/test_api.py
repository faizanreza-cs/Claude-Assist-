from fastapi.testclient import TestClient

from pharmapulse.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_share_endpoint():
    r = client.get("/share", params={"by": "region"})
    assert r.status_code == 200
    assert {"region", "brand", "share"} <= set(r.json()["rows"][0])


def test_forecast_endpoint_validates():
    assert client.get("/forecast/Glyvance", params={"horizon": 4}).status_code == 200
    assert client.get("/forecast/Unknown").status_code == 422


def test_ask_endpoint_mock():
    r = client.post("/ask", json={"question": "Forecast Glyvance TRx for 4 weeks"})
    body = r.json()
    assert r.status_code == 200
    assert body["tools_used"] == ["forecast_brand"] and body["answer"]
