# 🛡️ SwipeTCHA – Smart Behaviour-Based CAPTCHA

SwipeTCHA is a behaviour-based, user-friendly security system designed to stop bots while keeping human verification fast and smooth. Instead of making people solve puzzles, it looks at **how** they drag a slider – mouse speed, path, timing and jitter – and uses supervised machine learning to decide *human* or *bot*. It also aims to improve accessibility and user experience compared with traditional puzzle CAPTCHAs.

🔗 **Live demo:** https://captcha-gray-sigma.vercel.app
🔐 **Demo integration:** [captcha_demo-site](https://github.com/shivamani-code/captcha_demo-site) – a full website locked behind this CAPTCHA

## ✨ Features

- **"Slide to verify" widget** – pointer events are recorded while dragging the handle
- **Rich behavioural features** computed in the browser, including:
  `avg_mouse_speed`, `mouse_path_entropy`, `click_delay`, `task_completion_time`, `idle_time`, `micro_jitter_variance`, `acceleration_curve`, `curvature_variance`, `overshoot_correction_ratio`, `timing_entropy`
- **Front-end bot heuristics** – flags drags that are too fast, too straight, too regular in timing or speed, or have too little jitter
- **Backend verification** – FastAPI `POST /verify` with a hard bot-rule gate, then a scikit-learn model (with a safe fallback when no model is loaded)
- **Embeddable widget** (`widget.js`) – drop a `<div id="smartcaptcha">` and the script on any site; it renders the CAPTCHA in an auto-resizing iframe and posts `{ type: "SMARTCAPTCHA_VERIFIED", success: true }` to the host page when the user passes
- **Feedback modal** – users can send feedback (email + message), stored in Firebase Firestore
- **Dataset tools** – synthetic data generator, model training script and a script to append exported labelled rows to the dataset

## 🛠️ Tech Stack

| Layer | Tech |
|-------|------|
| Frontend / widget | HTML, CSS, vanilla JavaScript (Pointer Events, iframe + `postMessage`) |
| Backend | Python, FastAPI, Uvicorn |
| ML | scikit-learn (Random Forest), pandas, NumPy, joblib |
| Feedback | Firebase Firestore |
| Hosting | Vercel (frontend) |

## 📁 Project Structure

```
captcha/
├── index.html / style.css         # Demo page (deployed site)
├── smartcaptcha.js                # Slider widget + feature extraction + verify call
├── widget.js                      # Embeddable loader (iframe + SMARTCAPTCHA_VERIFIED message)
├── firebaseFeedback.js            # Feedback → Firestore
├── feedbackModal.js               # Feedback modal UI
├── backend/                       # Minimal FastAPI verify service
└── smartcaptcha/
    ├── frontend/                  # Standalone widget demo
    ├── backend/app.py             # FastAPI: rule gate + ML model
    ├── ml/                        # generate_synthetic_data.py, train_model.py, merge_data.py
    └── dataset/                   # behavior_data.csv + append_rows.py
```

## 🚀 Run Locally

**Frontend**

```bash
git clone https://github.com/shivamani-code/captcha.git
cd captcha
npx serve .
```

The widget posts to `VERIFY_ENDPOINT` at the top of `smartcaptcha.js`; point it at your local API to use your own backend.

**Backend + model**

```bash
cd smartcaptcha
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt numpy

cd ml
python generate_synthetic_data.py
python train_model.py                 # creates captcha_model.pkl
cp captcha_model.pkl ../backend/

cd ../backend
uvicorn app:app --reload --port 8000
```

## 🧩 Embed on Your Site

```html
<div id="smartcaptcha"></div>
<script src="https://captcha-gray-sigma.vercel.app/widget.js"></script>
<script>
  window.addEventListener("message", (e) => {
    if (e.data?.type === "SMARTCAPTCHA_VERIFIED" && e.data.success === true) {
      // user verified – unlock your content
    }
  });
</script>
```
