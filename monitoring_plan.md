# Monitoring Plan for Churn Scoring Service

## Data drift
- Track the distribution of incoming feature values over time.
- Monitor key signal changes for recency, frequency, monetary value, ticket volumes, and engagement metrics.
- Track categorical feature distributions for `city_tier`, `age_group`, `acquisition_channel`, `preferred_category`, `marketing_consent`, and `loyalty_tier`.
- Alert when categorical values drift or when incoming values fall outside the expected training categories.
- Alert when feature distribution shifts beyond a configured threshold compared to the training baseline.

## Prediction distribution
- Track the percentage of low, medium, and high-risk predictions.
- Monitor the average churn probability and the share of predicted churn cases.
- Alert on sudden increases in high-risk predictions or collapsing prediction variance.

## Business outcomes
- Measure actual churn outcomes for customers that receive retention treatment.
- Compare predicted risk groups against observed churn rates.
- Use business KPI trends to validate that the model supports improved retention decisions.

## API errors
- Track request validation failures and response errors.
- Monitor invalid categorical values and schema mismatches from client payloads.
- Monitor endpoint latency and HTTP 5xx/4xx counts.
- Alert on repeated invalid requests or service availability issues.

## Retraining triggers
- Trigger retraining when categorical input coverage changes significantly or new product/customer behavior segments appear.
- Retrain if model performance degrades on recent business outcomes.
- Retrain after major product, pricing, or campaign changes that alter customer behavior.
- Trigger retraining when data drift is detected in key input features.
- Retrain if model performance degrades on recent business outcomes.
- Retrain after major product, pricing, or campaign changes that alter customer behavior.
