# ForenSight AI

ForenSight AI is a local-first forensic evidence handling risk assessment and
recordkeeping foundation. It records handling conditions, highlights possible
risk indicators, and preserves assessments in SQLite.

> **Assessment limitation:** scores are preliminary, transparent rule-based
> screening only. They are not a trained AI model, do not establish that
> contamination occurred, and are not a substitute for forensic judgment or
> laboratory procedures.

## Requirements

- Python 3.11 or newer
- pip

## Setup

From the project directory, create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Generate a secret and place it in `.env` as `JWT_SECRET_KEY`:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

For local HTTP development, keep `COOKIE_SECURE=false`. Set it to `true` when
serving the application behind HTTPS. Then start the server:

```powershell
python -m uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). API documentation is
available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
SQLite data is created at `data/forensight.sqlite3` by default; override it
with `DATABASE_PATH`.

## Tests

```powershell
python -m pytest
```

## Security and data handling

- Passwords are hashed with Argon2id through `pwdlib`.
- Authentication uses expiring HS256 JWTs in `HttpOnly`, `SameSite=Strict`
  cookies. Protected endpoints also accept standard bearer tokens.
- Unsafe requests require a CSRF token and reject mismatched request origins.
- Dashboard and assessment queries are scoped to the authenticated user.
- Use HTTPS, rotate secrets through the environment, and follow organizational
  access-control, retention, backup, and evidence-handling policies in
  deployment. This foundation is not a validated forensic decision system.

## Experimental Logistic Regression evaluation

The Overview reports a separate Logistic Regression evaluation using the
purpose-built dataset in `data/synthetic_evidence_outcomes.csv` (1,500 records,
seed `20261113`). Regenerate the file with:

```powershell
python -m app.services.synthetic_dataset
```

Each record uses the evidence type, collection conditions, PPE compliance,
collection tool condition, packaging condition, storage condition, separate
packaging, and chain-of-custody documentation fields used by the assessment
form. The generator samples feature values from documented fixed categorical
distributions. It then calculates a simulated higher-risk probability from an
independent logistic equation with evidence-type effects, handling-condition
effects, and two interactions. It samples the binary outcome from that
probability and flips 8% of sampled outcomes to provide explicit label noise.
It does not call or reproduce the rule-based scoring engine; simulated labels
are not records of actual contamination.

Feature fields are sampled independently using these probabilities: evidence
type (Biological 22%, Trace 19%, Digital 16%, Firearms 14%, Document 13%,
Other 16%); collection conditions (controlled 70%, challenging 22%, unknown
8%); PPE compliant 84%; tool condition (clean 76%, compromised 18%, unknown
6%); packaging (intact 78%, compromised 16%, unknown 6%); storage (controlled
70%, uncontrolled 24%, unknown 6%); separate packaging 78%; custody
documentation complete 82%. For the higher-risk outcome, the generator uses
`p = 1 / (1 + exp(-η))`, where `η` starts at -2.0. Evidence-type offsets are
Biological +0.25, Trace +0.10, Digital -0.25, Firearms +0.15, Document -0.15,
and Other 0. Condition offsets are challenging collection +0.75 / unknown
+0.35; PPE non-compliance +0.70; compromised / unknown tool +0.80 / +0.30;
compromised / unknown packaging +0.75 / +0.30; uncontrolled / unknown storage
+0.70 / +0.25; no separate packaging +0.70; incomplete custody +0.65. Add
+0.30 for challenging collection with non-compliant PPE and +0.35 for
compromised packaging without separate packaging. The outcome is sampled
Bernoulli from `p`, then independently inverted with probability 0.08. All
values are simulation parameters, not empirically estimated effects.

The app one-hot encodes the categorical inputs and uses only scikit-learn
Logistic Regression. It evaluates a reproducible 75/25 stratified train/test
split (seed `42`) and displays accuracy, macro precision, macro recall, macro
F1, class-wise metrics, and the held-out confusion matrix. These are actual
metrics for this generated dataset and split, but they measure only performance
on synthetic data. They are not real-world forensic validation or
contamination-detection accuracy. The rule-based assessment remains the
primary screening mechanism and is not combined with the experimental model.

## Application structure

- `app/main.py` — FastAPI application, lifecycle, request protections, static UI
- `app/routes.py` — authentication, dashboard, and assessment API
- `app/auth.py` — password hashing and JWT authentication
- `app/database.py` — SQLite schema and connections
- `app/services/risk_assessment.py` — model interface and preliminary rules
- `app/services/logistic_regression.py` — isolated experimental Logistic Regression evaluation
- `app/services/synthetic_dataset.py` — reproducible synthetic feature and label generator
- `app/schemas.py` — request and response validation
- `frontend/` — responsive HTML, CSS, and vanilla JavaScript
- `tests/` — API workflow tests
