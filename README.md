# Part 4 — FastAPI Churn Scoring Service

This folder contains the Part 4 deliverable: a runnable FastAPI application that loads a pre-trained churn model and exposes prediction endpoints for CRM integration.

## Folder structure
- `app/main.py` — FastAPI application with `/health`, `/predict`, and `/batch_predict` endpoints.
- `model.pkl` — saved churn model pipeline used by the API.
- `requirements4.txt` — Python dependencies for the service.
- `tests/test_api.py` — automated API test cases.
- `monitoring_plan.md` — deployment monitoring plan.

## Setup
1. Open a terminal and change into the service folder:
```bash
cd part4
```
2. Create and activate a Python virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements4.txt
```

## Run the API
From the `part4` folder with the virtual environment activated:
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
If `python` is not on your PATH, use:
```powershell
py -3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the browser at `http://127.0.0.1:8000/docs` to access Swagger UI.

## Endpoints

### GET /health
Health check endpoint to confirm the API is running.

Example request:
```bash
curl http://127.0.0.1:8000/health
```

Example response:
```json
{
  "status": "ok"
}
```

### POST /predict
Accepts a single customer record and returns churn risk output.

Example request body:
```json
{
  "city_tier": "Tier 1",
  "age_group": "25-34",
  "acquisition_channel": "Google Search",
  "loyalty_tier": "Gold",
  "preferred_category": "Hair Care",
  "marketing_consent": "Yes",
  "recency_days": 30,
  "frequency_180d": 4,
  "monetary_180d": 260.0,
  "return_rate_180d": 0.1,
  "avg_discount_pct_180d": 0.15,
  "avg_rating_180d": 4.2,
  "category_diversity_180d": 3.0,
  "ticket_count_90d": 1,
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
  "last_visit_days_ago": 7.0
}
```

Example response:
```json
{
  "churn_probability": 0.2483,
  "predicted_class": 0,
  "risk_level": "low",
  "risk_explanation": "The customer profile appears stable across the provided retention signals."
}
```

### POST /batch_predict
Accepts multiple customer records and returns a prediction for each input item.

Example request body:
```json
[
  {
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "acquisition_channel": "Google Search",
    "loyalty_tier": "Gold",
    "preferred_category": "Hair Care",
    "marketing_consent": "Yes",
    "recency_days": 30,
    "frequency_180d": 4,
    "monetary_180d": 260.0,
    "return_rate_180d": 0.1,
    "avg_discount_pct_180d": 0.15,
    "avg_rating_180d": 4.2,
    "category_diversity_180d": 3.0,
    "ticket_count_90d": 1,
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
    "last_visit_days_ago": 7.0
  }
]
```

## Test execution
Run the test suite from the `part4` folder:
```bash
pytest tests
```

## Model/source data notes
- The API loads a pre-trained pipeline from `model.pkl`.
- The saved model was trained using the Part 3 churn modeling workflow.
- Input payloads must include the full feature set expected by the model.
- The following categorical values are accepted by the API:
  - `city_tier`: `Tier 1`, `Tier 2`, `Tier 3`
  - `age_group`: `18-24`, `25-34`, `35-44`, `45+`
  - `acquisition_channel`: `Google Search`, `Instagram`, `Influencer`, `Referral`, `Marketplace`, `Organic`
  - `preferred_category`: `Baby Care`, `Fragrance`, `Hair Care`, `Makeup`, `Skin Care`, `Wellness`
  - `marketing_consent`: `Yes`, `No`
  - `loyalty_tier`: `None`, `Silver`, `Gold`, `Platinum`
- The service is self-contained and can be deployed independently of the larger project.

## Responsible use guidance
- Use the churn score as an advisory signal, not as the sole decision criterion.
- Combine API output with customer service context before acting.
- Do not use the prediction to automatically cancel accounts or apply punitive actions.
- Review high-risk recommendations with business stakeholders before executing campaigns.
