import pytest
from fastapi.testclient import TestClient

from healthcare_web.app import db
from healthcare_web.app.auth import hash_password
from src.storage.dialog_store import DialogStore


@pytest.fixture
def env(tmp_path, monkeypatch):
    db_path = tmp_path / "dialogs.db"
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DIALOGS_DB", db_path)
    store = DialogStore(db_path=db_path)  # creates bot-side tables (messages, alerts, ...)
    db.ensure_tables()

    owner_id = db.create_user("owner@example.com", hash_password("pw-owner"))
    db.create_user("other@example.com", hash_password("pw-other"))

    with store._connect() as conn:
        conn.execute(
            "INSERT INTO invites (code, user_id, expires_at) VALUES ('AAAA0001', ?, datetime('now', '+1 hour'));",
            (owner_id,),
        )
        conn.commit()
    store.link_patient_with_invite("AAAA0001", 1001, "Patient One")
    store.add_message(1001, "user", "hello")
    store.add_summary(
        chat_id=1001, user_count=3, summary_text="**Support Summary** <script>alert(1)</script>",
        emotion="negative", risk_level="high", depression_score=0.8, anxiety_score=0.2,
        stress_score=0.3, profile_note="",
    )
    store.add_alert(1001, reason="high_risk_message", risk_level="high")
    patient = db.get_patient_by_chat_id("1001")

    from healthcare_web.app.main import app
    with TestClient(app) as client:
        yield client, store, patient


def _login(client, email, password):
    client.cookies.clear()
    response = client.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert response.status_code == 302
    assert "session" in response.cookies or client.cookies.get("session")


def test_owner_sees_patient_and_summary_is_escaped(env):
    client, _, patient = env
    _login(client, "owner@example.com", "pw-owner")
    response = client.get(f"/patients/{patient['id']}")
    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;" in response.text
    assert client.get(f"/api/patients/{patient['id']}/messages").json()[0]["text"] == "hello"
    assert len(client.get("/api/alerts").json()) == 1


def test_other_specialist_cannot_access_foreign_patient(env):
    client, _, patient = env
    _login(client, "other@example.com", "pw-other")
    for url in (
        f"/patients/{patient['id']}",
        f"/api/patients/{patient['id']}/messages",
        f"/api/patients/{patient['id']}/summaries",
        f"/patients/{patient['id']}/export.csv",
    ):
        assert client.get(url, follow_redirects=False).status_code == 404, url
    assert client.get("/api/alerts").json() == []
    assert "1001" not in client.get("/export/alerts.csv").text

    alert_id = db.list_alerts(active_only=True)[0]["id"]
    response = client.post(f"/alerts/{alert_id}/resolve", follow_redirects=False)
    assert response.status_code == 404
    assert db.count_active_alerts() == 1


def test_unauthenticated_redirects_to_login(env):
    client, _, patient = env
    client.cookies.clear()
    response = client.get(f"/patients/{patient['id']}", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_patient_portal_code_is_single_use(env):
    client, store, _ = env
    code = store.create_patient_portal_token(1001, ttl_minutes=5)
    client.cookies.clear()
    first = client.post("/patient/login", data={"code": code.upper()}, follow_redirects=False)
    assert first.status_code == 302
    assert first.headers["location"] == "/patient/preferences"
    assert client.get("/patient/preferences").status_code == 200

    client.cookies.clear()
    second = client.post("/patient/login", data={"code": code}, follow_redirects=False)
    assert second.status_code == 200
    assert "Invalid or expired code" in second.text


def test_expired_portal_code_rejected(env):
    _, store, _ = env
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO patient_portal_tokens (code, chat_id, expires_at) "
            "VALUES ('0ldc0de0', '1001', datetime('now', '-1 minute'));"
        )
        conn.commit()
    assert db.consume_patient_portal_token("0ldc0de0") is None


def test_unknown_alert_is_404_not_500(env):
    client, _, _ = env
    _login(client, "owner@example.com", "pw-owner")
    assert client.post("/alerts/999999/resolve", follow_redirects=False).status_code == 404
    assert client.post("/settings", data={"report_language": "xx"}).status_code == 400
