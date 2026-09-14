# Golden Evaluation Set: Sampling & Labeling Methodology

## 1. Dataset Scope & Stratified Sampling Strategy
- **Sample Size:** 198 customer–brand conversational turns from `@Uber_Support` threads.
- **Source:** Verified historical threads reconstructed via `in_response_to_tweet_id` from the Kaggle Customer Support on Twitter dataset.
- **Length Stratification:**
  - Short queries (< 50 characters): 66 samples
  - Medium queries (50–120 characters): 66 samples
  - Long inquiries (> 120 characters): 66 samples

This stratification prevents evaluation bias toward brief or long inquiries.

## 2. Intent Taxonomy
Every example was assigned one of nine canonical intents:
1. `trip_fare_dispute`: Billing errors, surge overcharging, incorrect route fees.
2. `safety_incident`: Threat of violence, collision, intoxicated driving, harassment (**CRITICAL**).
3. `lost_item`: Belongings left in transit vehicles.
4. `driver_conduct`: Rude language, refusing service animals, destination discrimination.
5. `account_access`: 2FA delivery failures, credential theft, unauthorized trips.
6. `promo_credit_issue`: Unapplied coupons, missing Uber Cash vouchers.
7. `ride_cancellation_delay`: Driver refusing to move, false no-show fees.
8. `general_inquiry`: Luggage capacity, vehicle passenger limits, business tax receipts.
9. `other_unroutable`: Non-actionable remarks, contextual follow-ups ("sent you DM").

## 3. Escalation Ground Truth Policy
- **ESCALATE:** Mandatory for all `safety_incident`, `driver_conduct`, and `account_access` cases regardless of classifier confidence. Also applied to unroutable fragments requiring human context.
- **AUTO_HANDLE:** Permitted only for low/medium risk intents (`lost_item`, `promo_credit_issue`, `general_inquiry`, `ride_cancellation_delay`, `trip_fare_dispute`) where standard brand routing (e.g., directing to in-app Help or initiating DM handoff) resolves the initial turn safely.