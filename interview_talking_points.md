# Interview talking points — CKD screening project

Keep this for yourself; don't submit it. Use it to rehearse.

## The 60-second pitch (open with this if they ask "walk me through a project")

"I built a point-of-care CKD risk screening model using the UCI CKD dataset, and I framed it
around your actual product — screening diabetic and hypertensive patients before referral. I
spent most of the effort on the parts that map to how you'd actually deploy something like
this: cleaning genuinely messy clinical data, engineering a few nephrology-informed features,
comparing an interpretable baseline against XGBoost, and then using SHAP to sanity-check that
the model's reasoning matches clinical intuition instead of just chasing an accuracy number.
I also wrote up explicitly what I'd need to change before it touched a real patient — bigger
multi-site data, calibration, drift monitoring, SaMD-style documentation."

## Questions they might ask, and how to answer

**"Why is your AUC 1.0? That's suspicious."**
Say this proactively before they even ask, ideally. Your answer: this specific UCI dataset is
a known easy benchmark — several of the "risk factor" columns (hemoglobin, specific gravity,
RBC count) are close to diagnostic markers rather than just predictive signals, and it's a small,
likely single-site sample. High separability is well documented on this exact dataset elsewhere.
The number itself isn't the achievement — the pipeline, the feature engineering, and the honest
limitations section are. This is your chance to show maturity: a junior candidate brags about
1.0 AUC, a strong candidate flags why it's not trustworthy.

**"How would you get this to production quality?"**
Point straight to your README's "what I'd change" section: bigger multi-site prospective data,
calibration curves not just AUC, subgroup performance checks, drift monitoring (tie to your
Evidently AI / MLflow experience at Mediassist), SaMD-aligned documentation.

**"Why XGBoost over logistic regression, or vice versa?"**
Logistic regression is fully interpretable by coefficient — good for a first regulatory-friendly
baseline. XGBoost captures nonlinear interactions and handles missing values natively (useful
given the real missingness in this data) but needs SHAP to stay interpretable. You compared
both deliberately, not just picked one.

**"What does bun_creatinine_ratio actually tell you clinically?"**
A high BUN:creatinine ratio suggests a prerenal cause (dehydration, reduced kidney perfusion)
rather than intrinsic kidney damage — useful for distinguishing reversible from progressive
kidney issues. Know this cold since you engineered it.

**"Would this work on your RxT 21 machine's live data?"**
Be honest: no, not as-is. This models point-of-care *screening* labs (pre-dialysis), not
intradialytic vitals from the machine itself. If asked to extend it, you'd pivot to time-series
work — e.g., predicting intradialytic hypotension from real-time vitals during a session — and
that's a natural "if I joined, here's where I'd take this next" answer.

**"How long did this take you?"**
Be honest and match the scope: this was a focused pre-interview build, not months of work.
Frame it as proof of how you approach a new problem quickly, not as your life's work.

## Don't do this

- Don't claim the 1.0 AUC as a strength without immediately caveating it — it will read as
  either not understanding overfitting or trying to hide it. Either is worse than just naming it.
- Don't claim FDA/IEC 62304 expertise you don't have. Say you understand the *principle*
  (documented validation, traceability, monitoring) and would formalize it against the real
  framework with their support — matches exactly what the JD says they'll help you grow into.
