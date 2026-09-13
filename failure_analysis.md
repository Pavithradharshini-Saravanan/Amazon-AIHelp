# Failure Analysis: Top 5 Failure Modes

This document analyzes the top 5 failure modes observed during evaluation of the AmazonHelp Twitter AI Support Agent. Each mode is documented with a real dataset example, observed vs. expected behavior, and underlying hypotheses.

---

### Failure Mode 1: Multilingual & Non-English Tweet Confusion

- **Real Example (Japanese)**: 
  > *"なにこれ(°д°) 開いたらあかんやつよね？ am͜a͉zonさん？ AMAZONさん？ 頼んでないもん でもこれは 騙されちゃうよー .…"*
- **Expected Intent**: `account_access` (Phishing / Scam alert requiring security advice)
- **Predicted Intent**: `damaged_or_wrong_item`
- **Root Cause & Hypothesis**: Keyword classifiers and standard unilingual embedding models fail to capture non-English semantic nuance. While human agents easily recognize phishing warnings in Japanese, text models trained primarily on English support text misattribute keywords or fall back to generic item categories.

---

### Failure Mode 2: Multi-Intent & Severe Escalation Misclassification

- **Real Example**:
  > *"only 9 out of 21 ordered items were delivered today. No way to request a refund. This is not the first time when such thing happened, but this time it is really awful. We are upset and most likely will cancel our membership."*
- **Expected Intent / Action**: `delivery_issue` + **ESCALATE** (`yes`)
- **Predicted Intent**: `refund_return`
- **Root Cause & Hypothesis**: Complex customer messages contain overlapping complaints (partial delivery + refund block + membership cancellation threat). Single-label classifiers pick the dominant keyword (`refund`) rather than prioritizing the root operational cause (`delivery_issue`), missing critical multi-intent context.

---

### Failure Mode 3: Sarcasm, Rhetorical Expressions & Passive Aggression

- **Real Example**:
  > *"A+ on packaging lmao."* or *"Pondering how prime delivery with can take longer than Royal Mail second class delivery to arrive. #notreallyprimeisit?"*
- **Expected Intent**: `general_complaint` / `damaged_or_wrong_item`
- **Predicted Intent**: `product_inquiry`
- **Root Cause & Hypothesis**: Sarcastic praise ("A+ packaging") tricks simple sentiment and keyword tools into assuming positive feedback. Without deep conversational reasoning, models fail to recognize sarcastic slang (`lmao`, `#notreallyprimeisit`) as passive-aggressive dissatisfaction.

---

### Failure Mode 4: Vague Short Follow-Ups & Missing Dialogue State

- **Real Example**:
  > *"Getting a practical solution would have been better for me."* or *"Shared details on web form. Pl action"*
- **Expected Intent**: `general_complaint` or `other`
- **Predicted Intent**: `delivery_issue` (hallucinated from default fallback)
- **Root Cause & Hypothesis**: In single-turn processing, messages referencing prior off-platform interactions ("web form", "earlier email") lack self-contained domain keywords. Without cross-turn conversation history, the agent cannot infer what issue the customer previously reported.

---

### Failure Mode 5: Public Privacy Over-sharing vs. Action Execution Limits

- **Real Example**:
  > *"I placd an order iPhone se 32 Gb variant with net banking and my acc blocked for security reason! My Order ID 404-6662997-2205169"*
- **Expected Action**: **ESCALATE** (`yes` - Security/Lockout requiring private human identity verification)
- **Observed Reply**: *"Please do not post order details publicly..."* (Correct tone, but bot cannot resolve issue)
- **Root Cause & Hypothesis**: Public social media channels strictly prevent automated API execution of account modifications or refunds for privacy compliance. The agent properly redirects to DM, but customers expecting instant automated resolution on Twitter experience friction.
