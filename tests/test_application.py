from datetime import datetime, timezone

from fastapi.testclient import TestClient


def csrf_headers(client):
    response = client.get("/api/auth/csrf")
    assert response.status_code == 200
    return {"X-CSRF-Token": response.json()["csrf_token"]}


def signup(client, email="analyst@example.org"):
    return client.post(
        "/api/auth/signup",
        headers=csrf_headers(client),
        json={
            "full_name": "Case Analyst",
            "email": email,
            "password": "Secure-Example-Passphrase-4",
        },
    )


def test_dashboard_requires_authentication_and_starts_empty(client):
    assert client.get("/api/dashboard").status_code == 401
    created = signup(client)
    assert created.status_code == 201

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["total"] == 0

    other_client = TestClient(client.app)
    assert other_client.get("/api/dashboard").status_code == 401


def test_dashboard_analytics_are_empty_without_saved_assessments(client):
    assert signup(client).status_code == 201

    dashboard = client.get("/api/dashboard").json()

    assert dashboard["total"] == 0
    assert dashboard["evidence_types"] == [
        {"evidence_type": evidence_type, "count": 0}
        for evidence_type in ("Biological", "Trace", "Digital", "Firearms", "Document", "Other")
    ]
    assert all(indicator["count"] == 0 for indicator in dashboard["indicators"])
    assert dashboard["monthly_trend"] == []
    assert dashboard["high_risk_assessments"] == []
    assert dashboard["model_evaluation"]["status"] == "experimental"
    assert dashboard["model_evaluation"]["dataset_type"] == "synthetic"
    assert dashboard["model_evaluation"]["accuracy"] is not None


def test_dashboard_analytics_aggregate_only_saved_user_assessments(client):
    assert signup(client).status_code == 201
    common = {
        "evidence_type": "Trace",
        "collection_conditions": "controlled",
        "ppe_compliant": True,
        "tool_condition": "clean",
        "packaging_condition": "intact",
        "storage_condition": "controlled",
        "separately_packaged": True,
        "chain_of_custody_documented": True,
    }
    assessments = [
        {**common, "evidence_ref": "ANALYTICS-LOW"},
        {
            **common,
            "evidence_ref": "ANALYTICS-MEDIUM",
            "evidence_type": "Digital",
            "collection_conditions": "challenging",
            "storage_condition": "uncontrolled",
        },
        {
            **common,
            "evidence_ref": "ANALYTICS-HIGH",
            "evidence_type": "Biological",
            "collection_conditions": "challenging",
            "ppe_compliant": False,
            "tool_condition": "compromised",
            "packaging_condition": "compromised",
            "storage_condition": "uncontrolled",
            "separately_packaged": False,
            "chain_of_custody_documented": False,
        },
    ]
    for assessment in assessments:
        response = client.post(
            "/api/assessments",
            headers=csrf_headers(client),
            json=assessment,
        )
        assert response.status_code == 201

    dashboard = client.get("/api/dashboard").json()
    evidence_types = {
        item["evidence_type"]: item["count"] for item in dashboard["evidence_types"]
    }
    indicators = {item["key"]: item["count"] for item in dashboard["indicators"]}

    assert (dashboard["total"], dashboard["low"], dashboard["medium"], dashboard["high"]) == (3, 1, 1, 1)
    assert evidence_types == {
        "Biological": 1,
        "Trace": 1,
        "Digital": 1,
        "Firearms": 0,
        "Document": 0,
        "Other": 0,
    }
    assert indicators == {
        "ppe": 1,
        "collection": 2,
        "packaging": 1,
        "separate_packaging": 1,
        "storage": 2,
        "custody": 1,
    }
    assert dashboard["monthly_trend"] == [
        {"month": datetime.now(timezone.utc).strftime("%Y-%m"), "count": 3}
    ]
    assert len(dashboard["high_risk_assessments"]) == 1
    assert dashboard["high_risk_assessments"][0]["evidence_ref"] == "ANALYTICS-HIGH"
    assert dashboard["high_risk_assessments"][0]["risk_score"] == 14

    second_client = TestClient(client.app)
    second_client.post(
        "/api/auth/signup",
        headers=csrf_headers(second_client),
        json={
            "full_name": "Second Analyst",
            "email": "second@example.org",
            "password": "Another-Secure-Passphrase-5",
        },
    )
    second_dashboard = second_client.get("/api/dashboard").json()
    assert second_dashboard["total"] == 0
    assert second_dashboard["monthly_trend"] == []
    assert second_dashboard["high_risk_assessments"] == []


def test_login_logout_and_user_scoped_assessments(client):
    assert signup(client).status_code == 201
    own_assessment = {
        "evidence_ref": "OWNER-ONLY-001",
        "evidence_type": "Trace",
        "collection_conditions": "controlled",
        "ppe_compliant": True,
        "tool_condition": "clean",
        "packaging_condition": "intact",
        "storage_condition": "controlled",
        "separately_packaged": True,
        "chain_of_custody_documented": True,
    }
    assert client.post(
        "/api/assessments",
        headers=csrf_headers(client),
        json=own_assessment,
    ).status_code == 201
    assert client.post(
        "/api/auth/logout", headers=csrf_headers(client)
    ).status_code == 200
    assert client.get("/api/dashboard").status_code == 401

    login = client.post(
        "/api/auth/login",
        headers=csrf_headers(client),
        json={
            "email": "analyst@example.org",
            "password": "Secure-Example-Passphrase-4",
        },
    )
    assert login.status_code == 200
    assert client.get("/api/auth/me").json()["user"]["email"] == "analyst@example.org"
    assert client.get("/api/dashboard").json()["total"] == 1

    second_client = TestClient(client.app)
    second_client.post(
        "/api/auth/signup",
        headers=csrf_headers(second_client),
        json={
            "full_name": "Second Analyst",
            "email": "second@example.org",
            "password": "Another-Secure-Passphrase-5",
        },
    )
    assert second_client.get("/api/dashboard").json()["total"] == 0
    assert second_client.get("/api/assessments").json() == []
    assert len(client.get("/api/assessments").json()) == 1


def test_assessment_is_saved_explained_and_filterable(client):
    assert signup(client).status_code == 201
    assessment = {
        "evidence_ref": "FS-2026-0042",
        "evidence_type": "Biological",
        "collection_conditions": "challenging",
        "ppe_compliant": False,
        "tool_condition": "clean",
        "packaging_condition": "intact",
        "storage_condition": "controlled",
        "separately_packaged": True,
        "chain_of_custody_documented": False,
    }
    response = client.post(
        "/api/assessments",
        headers=csrf_headers(client),
        json=assessment,
    )
    assert response.status_code == 201
    result = response.json()
    assert result["risk_category"] == "high"
    assert result["risk_score"] == 6
    assert result["risk_factors"]
    assert result["recommendations"]
    assert result["evidence_ref"] == assessment["evidence_ref"]

    assert client.get("/api/assessments?risk=high").json()[0]["id"] == result["id"]
    assert client.get("/api/assessments?risk=low").json() == []
    assert client.get("/api/dashboard").json()["high"] == 1


def test_rejects_bad_credentials_validation_and_missing_csrf(client):
    blocked = client.post(
        "/api/auth/signup",
        json={"full_name": "A", "email": "not-an-email", "password": "short"},
    )
    assert blocked.status_code == 403
    assert blocked.headers["x-content-type-options"] == "nosniff"

    cross_origin = client.post(
        "/api/auth/signup",
        headers={**csrf_headers(client), "Origin": "https://untrusted.example"},
        json={
            "full_name": "Case Analyst",
            "email": "analyst@example.org",
            "password": "Secure-Example-Passphrase-4",
        },
    )
    assert cross_origin.status_code == 403

    assert signup(client).status_code == 201
    login = client.post(
        "/api/auth/login",
        headers=csrf_headers(client),
        json={"email": "analyst@example.org", "password": "incorrect-password"},
    )
    assert login.status_code == 401
    duplicate = signup(client)
    assert duplicate.status_code == 409


def test_assessment_reference_cannot_be_reused(client):
    assert signup(client).status_code == 201
    payload = {
        "evidence_ref": "ITEM-001",
        "evidence_type": "Trace",
        "collection_conditions": "controlled",
        "ppe_compliant": True,
        "tool_condition": "clean",
        "packaging_condition": "intact",
        "storage_condition": "controlled",
        "separately_packaged": True,
        "chain_of_custody_documented": True,
    }
    first = client.post(
        "/api/assessments", headers=csrf_headers(client), json=payload
    )
    second = client.post(
        "/api/assessments", headers=csrf_headers(client), json=payload
    )
    assert first.status_code == 201
    assert second.status_code == 409
    assert first.json()["risk_category"] == "low"
