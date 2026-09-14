import pandas as pd
import re
from pathlib import Path

TAXONOMY_INTENTS = [
    "trip_fare_dispute",
    "safety_incident",
    "lost_item",
    "driver_conduct",
    "account_access",
    "promo_credit_issue",
    "ride_cancellation_delay",
    "general_inquiry",
    "other_unroutable"
]

def rule_based_label(text: str):
    t = text.lower()

    # 1. Safety Incident (Critical risk -> Escalate)
    if any(w in t for w in ["accident", "danger", "police", "threat", "assault", "harass", "attack", "hospital", "emergency"]):
        return "safety_incident", "ESCALATE", "Contains critical safety/emergency keywords."

    # 2. Driver Misconduct (High risk -> Escalate)
    if any(w in t for w in ["screamed", "yelled", "rude", "refused", "service dog", "conduct", "attitude", "swearing"]):
        return "driver_conduct", "ESCALATE", "Customer reports abusive or non-compliant driver behavior."

    # 3. Account Access / Security (High risk -> Escalate)
    if any(w in t for w in ["hack", "hacked", "unauthorized", "login", "sign in", "password", "2fa", "verification code", "sms"]):
        return "account_access", "ESCALATE", "Security/account credential issue requiring human verification."

    # 4. Lost Item (Low risk -> Auto-handle eligible)
    if any(w in t for w in ["lost", "left my", "forgot my", "wallet", "keys", "phone in", "backpack", "belongings", "backseat"]):
        return "lost_item", "AUTO_HANDLE", "Report of forgotten or lost property."

    # 5. Promo / Credit / Coupon (Low risk -> Auto-handle eligible)
    if any(w in t for w in ["promo", "discount", "code", "coupon", "credit", "voucher", "uber cash", "dozen box"]):
        return "promo_credit_issue", "AUTO_HANDLE", "Promotion, voucher, or referral discount inquiry."

    # 6. Cancellation & Driver Delays (Medium risk -> Auto-handle eligible)
    if any(w in t for w in ["cancellation fee", "cancel", "cancelled", "waiting", "never showed", "never moved", "delay", "ghost ride"]):
        return "ride_cancellation_delay", "AUTO_HANDLE", "Dispute regarding driver cancellation or prolonged wait."

    # 7. Fare Dispute / Overcharge (Medium risk -> Auto-handle eligible)
    if any(w in t for w in ["charge", "charged", "fare", "overcharged", "refund", "receipt", "surge", "$", "price"]):
        return "trip_fare_dispute", "AUTO_HANDLE", "Billing discrepancy or overcharge claim."

    # 8. General Inquiries (Low risk -> Auto-handle eligible)
    if any(w in t for w in ["how do i", "how can i", "can ubersuv", "passengers", "tax", "invoice", "information", "question"]):
        return "general_inquiry", "AUTO_HANDLE", "General procedural inquiry or operational clarification."

    # 9. Unroutable / Follow-up spam / Pure handles
    return "other_unroutable", "ESCALATE", "Follow-up message, missing context, or unroutable dialogue snippet."

def annotate():
    csv_path = "data/golden/golden_set.csv"
    df = pd.read_csv(csv_path)

    intents, escalations, notes = [], [], []
    for _, row in df.iterrows():
        intent, esc, note = rule_based_label(str(row["customer_text"]))
        intents.append(intent)
        escalations.append(esc)
        notes.append(note)

    df["true_intent"] = intents
    df["true_escalation"] = escalations
    df["notes"] = notes

    df.to_csv(csv_path, index=False)
    print(f"Successfully labeled {len(df)} golden examples in {csv_path}!")
    print("\nClass Distribution:")
    print(df["true_intent"].value_counts())
    print("\nEscalation Split:")
    print(df["true_escalation"].value_counts())

if __name__ == "__main__":
    annotate()