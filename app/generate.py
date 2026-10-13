"""
ClaimPilot data generator.
Creates fake insurance claims + eval cases.
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)

ROOT = Path(__file__).parent
CLAIMS_DIR = ROOT / "claims"
CLAIMS_DIR.mkdir(exist_ok=True)

FIRST_NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Jamie",
               "Priya", "Rahul", "Arjun", "Ananya", "Vikram", "Neha", "Rohan", "Meera"]
LAST_NAMES = ["Chen", "Patel", "Garcia", "Nguyen", "Smith", "Kim", "Brown", "Lopez",
              "Sharma", "Verma", "Reddy", "Iyer", "Nair", "Gupta", "Khan", "Singh"]

# Incidents keyed by approximate Star Health policy coverage themes
INCIDENT_TEMPLATES = [
    # ---- Likely approved ----
    ("Hospitalized for dengue fever, 4-day admission, IV fluids and monitoring.", "approve"),
    ("Emergency appendectomy performed after severe abdominal pain.", "approve"),
    ("Cataract surgery on right eye, day-care procedure.", "approve"),
    ("Road accident, fractured tibia, surgery and 5-day hospitalization.", "approve"),
    ("Angioplasty with stent placement for blocked coronary artery.", "approve"),
    ("Normal delivery at network hospital, 3-day admission.", "approve"),
    ("Kidney stone removal via lithotripsy, day-care procedure.", "approve"),
    ("Hospitalized for pneumonia, 4 days on antibiotics.", "approve"),
    ("Hysterectomy for fibroids, 3-day hospital stay.", "approve"),
    ("Chemotherapy session for breast cancer, day-care.", "approve"),

    # ---- Likely rejected (exclusions) ----
    ("Routine dental cleaning and whitening at a dental clinic.", "reject"),
    ("Cosmetic rhinoplasty to change nose shape.", "reject"),
    ("Bariatric surgery for weight loss, BMI 28.", "reject"),
    ("IVF treatment for infertility at private clinic.", "reject"),
    ("Treatment for alcohol de-addiction at rehab center.", "reject"),
    ("Spectacles and contact lenses prescribed by optician.", "reject"),
    ("Routine annual health checkup with no symptoms.", "reject"),
    ("Vaccination for seasonal flu at local clinic.", "reject"),
    ("LASIK eye surgery for refractive error correction.", "reject"),
    ("Hair transplant procedure for male pattern baldness.", "reject"),

    # ---- Ambiguous (escalate) ----
    ("Admitted for chest pain, tests inconclusive, discharged after 2 days.", "escalate"),
    ("Hospitalized for high fever, cause not confirmed, 3-day stay.", "escalate"),
    ("Underwent procedure for chronic back pain, MRI showed mild disc bulge.", "escalate"),
    ("Admitted for sudden dizziness and fainting, no clear diagnosis.", "escalate"),
    ("Surgery for lump in breast, biopsy pending.", "escalate"),
    ("Hospitalized for severe headache, CT scan normal.", "escalate"),
    ("Treatment for persistent cough, suspected but not confirmed TB.", "escalate"),
    ("Admitted for abdominal pain, no clear diagnosis after tests.", "escalate"),
    ("Unplanned admission for monitoring after minor accident, no injury.", "escalate"),
    ("Treatment for unexplained weight loss, tests ongoing.", "escalate"),
]


def make_claim(idx):
    incident, expected = random.choice(INCIDENT_TEMPLATES)
    claim_id = "CLM-" + str(idx).zfill(4)

    return {
        "claim_id": claim_id,
        "policy_id": "POL-" + str(random.choice(["AUTO", "HOME", "HEALTH"])) + "-" + str(random.randint(1, 20)).zfill(3),
        "policy_type": "HEALTH",
        "claimant": random.choice(FIRST_NAMES) + " " + random.choice(LAST_NAMES),
        "filed_date": (datetime.now() - timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d"),
        "incident_description": incident,
        "amount_claimed": random.randint(5000, 250000),
        "expected_decision": expected,
    }


def generate_claims(n=100):
    claims = []
    for i in range(1, n + 1):
        claim = make_claim(i)
        path = CLAIMS_DIR / (claim["claim_id"] + ".json")
        path.write_text(json.dumps(claim, indent=2, ensure_ascii=False), encoding="utf-8")
        claims.append(claim)
    return claims


def generate_eval_cases(claims, n=50):
    by_decision = {"approve": [], "reject": [], "escalate": []}
    for c in claims:
        by_decision[c["expected_decision"]].append(c)

    per_class = n // 3
    selected = []
    for decision, items in by_decision.items():
        selected.extend(random.sample(items, min(per_class, len(items))))

    out_path = ROOT / "eval_cases.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in selected:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("Wrote " + str(len(selected)) + " eval cases to " + str(out_path))


if __name__ == "__main__":
    print("Generating claims...")
    claims = generate_claims(100)
    print("Generating eval cases...")
    generate_eval_cases(claims, 50)
    print("Done.")
    print("  Claims: " + str(len(claims)))