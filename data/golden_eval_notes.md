# Golden Evaluation Set — Sampling & Labeling Methodology

## Overview

A 200-sample hand-labeled evaluation set was constructed from the cleaned AmazonHelp Twitter
conversation corpus (8,000 `customer_message → brand_reply` pairs) to serve as ground truth
for evaluating the intent classifier, reply generator, and escalation engine.

---

## Sampling Strategy

**Source**: `data/cleaned_pairs.csv` — 8,000 AmazonHelp conversation pairs extracted from the
Kaggle Customer Support on Twitter dataset (thoughtvector/customer-support-on-twitter).

**Method**: Stratified random sampling was used rather than pure random sampling to ensure
representative coverage across all 8 intent categories. The full corpus was first tagged
with the keyword-baseline classifier to get approximate intent distribution, and then
samples were drawn proportionally from each intent bucket (e.g., ~30% `delivery_issue`,
~15% `refund_return`, etc.) so that minority intents like `account_access` and
`billing_payment` were not under-represented in the evaluation set.

**Size**: 200 samples — at the midpoint of the 150–250 range specified in the assignment —
chosen to balance labeling effort (~3 hours) against statistical reliability.

---

## Labeling Criteria

Each sample was labeled along three axes:

### 1. Intent Label (one of 8 categories)

| Intent | Labeling Rule |
|:---|:---|
| `delivery_issue` | Mentions late, missing, undelivered, or delayed shipments |
| `damaged_or_wrong_item` | Item arrived broken, defective, incorrect, or incomplete |
| `refund_return` | Requests for refund, return, replacement, or compensation |
| `account_access` | Login failures, password resets, suspicious activity, phishing alerts |
| `billing_payment` | Incorrect charges, payment failures, gift card issues, card holds |
| `general_complaint` | Venting, sarcasm, or dissatisfaction without a specific operational issue |
| `product_inquiry` | Questions about product specs, availability, or Prime membership |
| `other` | Everything else: greetings, vague follow-ups, non-complaint content |

**Ambiguous cases**: When a message contained multiple overlapping intents (e.g., partial
delivery AND refund request), the **primary root cause** was chosen as the label — the
issue that, if resolved, would most satisfy the customer. Secondary intents were noted in
a comment but not used in evaluation.

### 2. Escalation Flag (`yes` / `no`)

Messages were flagged for human escalation if they exhibited any of the following:

- **Angry or threatening tone**: Explicit threats to cancel Prime membership, contact legal
  teams, or post publicly about failures.
- **Monetary severity**: Refund disputes above typical amounts, billing errors, or charge
  backs that require human authorization.
- **Security or safety risk**: Account lockouts, suspected fraud, phishing reports, or
  personal data exposure on a public thread.
- **Account-specific verification requirement**: Issues that cannot be resolved without
  accessing the customer's private order, account, or payment history.
- **Repeated unresolved issues**: Messages explicitly stating prior contact without
  resolution ("this is the third time I'm contacting you").

### 3. Ideal Reply Guideline (free-text note)

For each sample, a brief note was written on what the ideal reply should accomplish:
e.g., "acknowledge the delay empathetically, ask for order number via DM, do not promise
a specific delivery date."

---

## Labeling Process & Consistency

- **Solo labeling**: All 200 samples were labeled by the same annotator in a single 3-hour
  sitting to eliminate inter-session criteria drift.
- **Calibration pass**: The first 20 samples were labeled, then re-read after completing
  all 200 to verify that early and late labeling criteria were consistent. 3 labels were
  revised during this calibration check.
- **Tie-breaking rule**: In genuinely ambiguous cases, the more operationally actionable
  intent was chosen (e.g., `delivery_issue` over `general_complaint` when both applied)
  to produce a more useful routing label for the support system.

---

## Final Distribution

| Intent | Count | % of Set |
|:---|:---:|:---:|
| `delivery_issue` | 77 | 38.5% |
| `refund_return` | 30 | 15.0% |
| `general_complaint` | 28 | 14.0% |
| `damaged_or_wrong_item` | 22 | 11.0% |
| `other` | 18 | 9.0% |
| `billing_payment` | 12 | 6.0% |
| `product_inquiry` | 8 | 4.0% |
| `account_access` | 5 | 2.5% |

**Escalation**: 58 out of 200 samples (29%) were flagged for human escalation.