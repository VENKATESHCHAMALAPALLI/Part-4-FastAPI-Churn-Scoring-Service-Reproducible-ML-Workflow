from fastapi.testclient import TestClient

from app.main import app

# TestClient wraps the FastAPI app so tests can call endpoints directly
client = TestClient(app)

# A representative, valid customer payload used by multiple tests. Keep
# this aligned with the API's required fields so tests remain meaningful.
VALID_CUSTOMER = {
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "acquisition_channel": "Google Search",
    "loyalty_tier": "Gold",
    "preferred_category": "Hair Care",
    "marketing_consent": "Yes",
    "recency_days": 30.0,
    "frequency_180d": 4.0,
    "monetary_180d": 260.0,
    "return_rate_180d": 0.1,
    "avg_discount_pct_180d": 0.15,
    "avg_rating_180d": 4.2,
    "category_diversity_180d": 3.0,
    "ticket_count_90d": 1.0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 12.0,
    "days_since_signup": 120.0,
    "sessions_30d": 20.0,
    "product_views_30d": 60.0,
    "cart_adds_30d": 5.0,
    "wishlist_adds_30d": 2.0,
    "abandoned_carts_30d": 1.0,
    "email_opens_30d": 3.0,
    "campaign_clicks_30d": 1.0,
    "last_visit_days_ago": 7.0,
}


def test_health_endpoint():
    # Verify health endpoint responds with status ok
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_expected_fields():
    # Check predict endpoint returns expected keys and value ranges
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    payload = response.json()
    # Ensure the API returns the canonical prediction fields consumed by
    # downstream services: probability, class, human-readable risk, and
    # a brief natural-language explanation.
    assert "churn_probability" in payload
    assert "predicted_class" in payload
    assert "risk_level" in payload
    assert "risk_explanation" in payload
    assert payload["predicted_class"] in [0, 1]
    assert payload["risk_level"] in ["low", "medium", "high"]


def test_batch_predict_endpoint_returns_multiple_predictions():
    # Ensure batch endpoint returns a prediction for each input record
    response = client.post("/batch_predict", json=[VALID_CUSTOMER, VALID_CUSTOMER])
    assert response.status_code == 200
    payload = response.json()
    assert "predictions" in payload
    assert len(payload["predictions"]) == 2
    assert all("churn_probability" in item for item in payload["predictions"])


def test_predict_endpoint_rejects_missing_fields():
    # Missing required fields should return 422 validation error
    invalid_payload = VALID_CUSTOMER.copy()
    invalid_payload.pop("recency_days")
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_category_value():
    # Invalid categorical inputs should trigger a 422 with field error
    invalid_payload = VALID_CUSTOMER.copy()
    invalid_payload["acquisition_channel"] = "online"
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    assert "acquisition_channel" in response.json()["detail"]


def test_predict_endpoint_accepts_case_insensitive_categories():
    # Case-insensitive labels should be normalized and accepted
    case_payload = VALID_CUSTOMER.copy()
    case_payload["city_tier"] = "tier 1"
    case_payload["age_group"] = "25-34"
    case_payload["acquisition_channel"] = "google search"
    case_payload["loyalty_tier"] = "gold"
    case_payload["preferred_category"] = "hair care"
    case_payload["marketing_consent"] = "yes"
    response = client.post("/predict", json=case_payload)
    assert response.status_code == 200
    assert response.json()["predicted_class"] in [0, 1]
