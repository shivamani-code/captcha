# SmartCAPTCHA

Local development project for SmartCAPTCHA.

## Project Structure

- `frontend/`: Static files for the CAPTCHA widget and demo page.
- `backend/`: FastAPI service and ML training/inference code.
- `dataset/`: CSV dataset used for model training.

## Local Setup

### 1) Backend: install dependencies

Create a virtual environment (recommended), then install:

```bash
pip install -r smartcaptcha/backend/requirements.txt
```

### 2) Dataset: add training rows

Edit:

`smartcaptcha/dataset/behavior_data.csv`

Add rows with the same columns as the header. Labels:

- `1` = human
- `0` = bot

### 3) Train the model

```bash
python smartcaptcha/backend/train_model.py
```

This creates:

`smartcaptcha/backend/smartcaptcha_model.joblib`

### 4) Run the API

Run from the `smartcaptcha/backend` folder:

```bash
uvicorn app:app --reload --port 8000
```

Health check:

`GET http://127.0.0.1:8000/health`

### 5) Run the frontend demo

Open:

`smartcaptcha/frontend/index.html`

in your browser, complete the slider, and observe the status result.

For more consistent local behavior (recommended), serve the frontend with a simple static server instead of opening the file directly.

### 6) Bot simulation (testing)

After training a model, run:

```bash
python smartcaptcha/backend/bot_simulation.py
```

It prints synthetic `bot_like` vs `human_like` confidence (`P(human)`).

## Dataset Collection Workflow (recommended)

### Export labeled rows from the demo

After completing the slider at least once, the demo enables:

- `Export as Human Row`
- `Export as Bot Row`

Each click downloads a small CSV containing **one labeled row** (header + row).

### Append exported rows into the training dataset

Use:

```bash
python smartcaptcha/dataset/append_rows.py path\\to\\smartcaptcha_row_human_*.csv path\\to\\smartcaptcha_row_bot_*.csv
```

This appends the rows into:

`smartcaptcha/dataset/behavior_data.csv`
