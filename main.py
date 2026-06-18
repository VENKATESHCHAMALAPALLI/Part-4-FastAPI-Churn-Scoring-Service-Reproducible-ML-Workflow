# Part 4 FastAPI churn scoring service
# - Loads a pre-trained sklearn pipeline from `model.pkl`
# - Exposes `/health`, `/predict`, and `/batch_predict` endpoints

# Name: Chamalapalli venkatesh
# Id : iitp_aiml_2506014


from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field

# - Input payloads are validated with Pydantic `CustomerFeatures`.
# - Categorical inputs are validated and mapped to the numeric encoding
#   expected by the saved pipeline. The service constructs a single-row
#   DataFrame with the exact `FEATURE_NAMES` used at training time.
# - Prediction output uses the model's `predict_proba` and `predict` calls
#   and returns a small risk explanation computed from input signals.

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Churn Scoring Service",
    description="Internal churn scoring API that returns churn probability, predicted class, and risk explanation for customer records.",
    version="1.0.0",
)

NUMERIC_FEATURES = [
    "recency_days",
    "frequency_180d",
    "monetary_180d",
    "return_rate_180d",
    "avg_discount_pct_180d",
    "avg_rating_180d",
    "category_diversity_180d",
    "ticket_count_90d",
    "negative_ticket_rate_90d",
    "avg_resolution_hours_90d",
    "days_since_signup",
    "sessions_30d",
    "product_views_30d",
    "cart_adds_30d",
    "wishlist_adds_30d",
    "abandoned_carts_30d",
    "email_opens_30d",
    "campaign_clicks_30d",
    "last_visit_days_ago",
]

CATEGORICAL_FEATURES = [
    "city_tier",
    "age_group",
    "acquisition_channel",
    "loyalty_tier",
    "preferred_category",
    "marketing_consent",
]

CITY_TIER_CATEGORIES = ["Tier 1", "Tier 2", "Tier 3"]
AGE_GROUP_CATEGORIES = ["18-24", "25-34", "35-44", "45+"]
ACQUISITION_CHANNEL_CATEGORIES = [
    "Google Search",
    "Instagram",
    "Influencer",
    "Referral",
    "Marketplace",
    "Organic",
]
PREFERRED_CATEGORY_CATEGORIES = [
    "Baby Care",
    "Fragrance",
    "Hair Care",
    "Makeup",
    "Skin Care",
    "Wellness",
]

MARKETING_CONSENT_MAPPING = {
    "yes": 1,
    "no": 0,
}
LOYALTY_TIER_MAPPING = {
    "none": 0,
    "silver": 1,
    "gold": 2,
    "platinum": 3,
}

FEATURE_NAMES = list(model.feature_names_in_)


class CustomerFeatures(BaseModel):
    city_tier: str = Field(..., description="Customer city tier")
    age_group: str = Field(..., description="Customer age group")
    acquisition_channel: str = Field(..., description="Acquisition channel")
    loyalty_tier: str = Field(..., description="Loyalty tier")
    preferred_category: str = Field(..., description="Preferred product category")
    marketing_consent: str = Field(..., description="Marketing consent status")
    recency_days: float = Field(..., ge=0, description="Days since last order")
    frequency_180d: float = Field(..., ge=0, description="Number of orders in the last 180 days")
    monetary_180d: float = Field(..., ge=0, description="Monetary spend in the last 180 days")
    return_rate_180d: float = Field(..., ge=0, le=1, description="Return rate in the last 180 days")
    avg_discount_pct_180d: float = Field(..., ge=0, le=1, description="Average discount percentage in the last 180 days")
    avg_rating_180d: float = Field(..., ge=0, le=5, description="Average order rating in the last 180 days")
    category_diversity_180d: float = Field(..., ge=0, description="Number of unique product categories purchased in the last 180 days")
    ticket_count_90d: float = Field(..., ge=0, description="Support ticket count in the last 90 days")
    negative_ticket_rate_90d: float = Field(..., ge=0, le=1, description="Proportion of negative tickets in the last 90 days")
    avg_resolution_hours_90d: float = Field(..., ge=0, description="Average ticket resolution time in hours")
    days_since_signup: float = Field(..., ge=0, description="Days since customer signup")
    sessions_30d: float = Field(..., ge=0, description="Web/app sessions in the last 30 days")
    product_views_30d: float = Field(..., ge=0, description="Product views in the last 30 days")
    cart_adds_30d: float = Field(..., ge=0, description="Cart adds in the last 30 days")
    wishlist_adds_30d: float = Field(..., ge=0, description="Wishlist adds in the last 30 days")
    abandoned_carts_30d: float = Field(..., ge=0, description="Abandoned carts in the last 30 days")
    email_opens_30d: float = Field(..., ge=0, description="Email opens in the last 30 days")
    campaign_clicks_30d: float = Field(..., ge=0, description="Campaign clicks in the last 30 days")
    last_visit_days_ago: float = Field(..., ge=0, description="Days since last web/app visit")


class PredictionResponse(BaseModel):
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_explanation: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


# Return a trimmed string for consistent downstream validation
def normalize_str(value: str) -> str:
    # Normalize strings coming from client payloads.
    # This centralizes any trimming/cleaning logic so validation is consistent
    # across the service (case-insensitive matching happens elsewhere).
    return str(value).strip()


# Validate a categorical choice and return the canonical training label
def validate_choice(value: str, choices: List[str], field_name: str) -> str:
    normalized = normalize_str(value)
    for choice in choices:
        if normalized.lower() == choice.lower():
            return choice
    raise HTTPException(
        status_code=422,
        detail=(
            f"Invalid value for '{field_name}': '{value}'. "
            f"Expected one of: {choices}."
        ),
    )

# Mapping helpers for categorical fields -------------------------------------------------


# Map marketing consent ('Yes'/'No') to binary integer
def map_marketing_consent(value: str) -> int:
    normalized = normalize_str(value).lower()
    if normalized not in MARKETING_CONSENT_MAPPING:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid value for 'marketing_consent': '{value}'. "
                "Expected one of: ['Yes', 'No']."
            ),
        )
    return MARKETING_CONSENT_MAPPING[normalized]


# Map loyalty tier label to ordinal integer used by the model
def map_loyalty_tier(value: str) -> int:
    normalized = normalize_str(value).lower()
    if normalized not in LOYALTY_TIER_MAPPING:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid value for 'loyalty_tier': '{value}'. "
                "Expected one of: ['None', 'Silver', 'Gold', 'Platinum']."
            ),
        )
    return LOYALTY_TIER_MAPPING[normalized]


# Convert numeric probability into a human-readable risk level
def determine_risk_level(probability: float) -> str:
    # Simple threshold-based buckets for human-readable risk levels.
    # These thresholds are intentionally conservative and can be tuned
    # based on business needs or calibration experiments.
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


# Produce a short natural-language explanation for why the risk was computed
def explain_risk(customer: CustomerFeatures, probability: float) -> str:
    reasons = []
    if customer.recency_days >= 90:
        reasons.append("low recent purchase activity")
    elif customer.recency_days >= 45:
        reasons.append("moderate recent purchase activity")
    if customer.ticket_count_90d >= 3:
        reasons.append("high support ticket volume")
    if customer.return_rate_180d >= 0.4:
        reasons.append("elevated return rate")
    if customer.campaign_clicks_30d < 1 and customer.sessions_30d < 5:
        reasons.append("low engagement with campaigns and web activity")
    if len(reasons) == 0:
        reasons.append("the customer profile appears stable across the provided retention signals")
    # Join and capitalize the first word to produce a short human-friendly
    # explanation attached to the prediction response.
    return f"{', '.join(reasons).capitalize()}."


# Build model features from validated input and return the prediction result
def make_prediction(customer: CustomerFeatures) -> PredictionResponse:
    # Convert incoming Pydantic model to a plain dict. Using `model_dump`
    # (Pydantic v2) keeps the code compatible with current dependency pins.
    customer_data = customer.model_dump()

    # Build the processed feature dict expected by the saved model. It must
    # match `FEATURE_NAMES` exactly (order and column set) so the pipeline
    # receives the same structure used during training.
    processed = {
        "loyalty_tier": map_loyalty_tier(customer_data["loyalty_tier"]),
        "marketing_consent": map_marketing_consent(customer_data["marketing_consent"]),
        "recency_days": customer_data["recency_days"],
        "frequency_180d": customer_data["frequency_180d"],
        "monetary_180d": customer_data["monetary_180d"],
        "return_rate_180d": customer_data["return_rate_180d"],
        "avg_discount_pct_180d": customer_data["avg_discount_pct_180d"],
        "avg_rating_180d": customer_data["avg_rating_180d"],
        "category_diversity_180d": customer_data["category_diversity_180d"],
        "ticket_count_90d": customer_data["ticket_count_90d"],
        "negative_ticket_rate_90d": customer_data["negative_ticket_rate_90d"],
        "avg_resolution_hours_90d": customer_data["avg_resolution_hours_90d"],
        "days_since_signup": customer_data["days_since_signup"],
        "sessions_30d": customer_data["sessions_30d"],
        "product_views_30d": customer_data["product_views_30d"],
        "cart_adds_30d": customer_data["cart_adds_30d"],
        "wishlist_adds_30d": customer_data["wishlist_adds_30d"],
        "abandoned_carts_30d": customer_data["abandoned_carts_30d"],
        "email_opens_30d": customer_data["email_opens_30d"],
        "campaign_clicks_30d": customer_data["campaign_clicks_30d"],
        "last_visit_days_ago": customer_data["last_visit_days_ago"],
    }

    # Validate categorical choices against the allowed training-time
    # categories. `validate_choice` is case-insensitive but returns the
    # canonical training label to simplify one-hot creation below.
    validated_city_tier = validate_choice(
        customer_data["city_tier"], CITY_TIER_CATEGORIES, "city_tier"
    )
    validated_age_group = validate_choice(
        customer_data["age_group"], AGE_GROUP_CATEGORIES, "age_group"
    )
    validated_acquisition_channel = validate_choice(
        customer_data["acquisition_channel"], ACQUISITION_CHANNEL_CATEGORIES, "acquisition_channel"
    )
    validated_preferred_category = validate_choice(
        customer_data["preferred_category"], PREFERRED_CATEGORY_CATEGORIES, "preferred_category"
    )

    for category in CITY_TIER_CATEGORIES:
        processed[f"city_tier_{category}"] = 1.0 if validated_city_tier == category else 0.0
    for category in AGE_GROUP_CATEGORIES:
        processed[f"age_group_{category}"] = 1.0 if validated_age_group == category else 0.0
    for category in ACQUISITION_CHANNEL_CATEGORIES:
        processed[f"acquisition_channel_{category}"] = (
            1.0 if validated_acquisition_channel == category else 0.0
        )
    for category in PREFERRED_CATEGORY_CATEGORIES:
        processed[f"preferred_category_{category}"] = (
            1.0 if validated_preferred_category == category else 0.0
        )

    # Create a single-row DataFrame with columns in the same order as the
    # model expects. This ensures the Numpy array inputs align correctly.
    payload = pd.DataFrame([processed], columns=FEATURE_NAMES)

    try:
        # `predict_proba` returns an array of shape (n_samples, n_classes).
        probability = float(model.predict_proba(payload)[:, 1][0])
        predicted = int(model.predict(payload)[0])
    except Exception as exc:
        # Convert unexpected model failures into HTTP 500 errors so clients
        # receive a meaningful message instead of a stack trace.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    probability = max(0.0, min(1.0, probability))
    risk_level = determine_risk_level(probability)
    risk_explanation = explain_risk(customer, probability)
    return PredictionResponse(
        churn_probability=round(probability, 4),
        predicted_class=predicted,
        risk_level=risk_level,
        risk_explanation=risk_explanation,
    )


@app.get("/", include_in_schema=False)
# Redirect root to the interactive Swagger UI
def root():
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
# Return no content for favicon requests to avoid 404 noise
def favicon():
    return Response(status_code=204)


@app.get("/health")
# Simple health check endpoint used by load balancers and monitoring
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
# Single-record prediction endpoint: accepts `CustomerFeatures` JSON
def predict(customer: CustomerFeatures):
    return make_prediction(customer)


@app.post("/batch_predict", response_model=BatchPredictionResponse)
# Batch prediction endpoint: accepts a list of customer records
def batch_predict(customers: List[CustomerFeatures]):
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="At least one customer record is required.")
    predictions = [make_prediction(customer) for customer in customers]
    return BatchPredictionResponse(predictions=predictions)
