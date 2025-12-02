# Breast Cancer Detection – Vercel Deployment Ready

This repository bundles the Streamlit/FastAPI deep-learning backend and the React frontend so they can be deployed together on [Vercel](https://vercel.com/).

## Repository Structure

```
BreastCancerDetect/
├── backend/               # FastAPI app + TensorFlow model + Grad-CAM utilities
├── frontend/              # React single-page UI (Create React App)
├── vercel.json            # Multi-build + routing config for Vercel
└── README.md
```

## Local Development

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Environment variables:

| Variable        | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `MODEL_PATH`    | Optional custom path to `breast_cancer_model.keras`. Defaults to `backend/models/...`. |
| `ALLOWED_ORIGINS` | CSV list of origins for CORS. Defaults to `*`.                          |

### Frontend

```bash
cd frontend
npm install
cp .env.example .env        # optional: override API base URL
npm start
```

`REACT_APP_API_BASE_URL` defaults to `http://127.0.0.1:8000` during local development and to `/api` in production.

## Deploying to Vercel

1. **Import the repo** into Vercel.
2. Ensure the project uses the root directory (the `vercel.json` file takes care of multi-build setup).
3. Configure environment variables (Project Settings → Environment Variables):
   - Backend: `MODEL_PATH` (if the model resides in Blob/S3) and `ALLOWED_ORIGINS` if you want to restrict CORS.
   - Frontend: `REACT_APP_API_BASE_URL` (set to `https://<your-domain>/api` if you need an explicit value).
4. The provided `vercel.json`:
   - Builds the React app via `@vercel/static-build`.
   - Packages the FastAPI app via `@vercel/python`.
   - Routes `/api/*` to the backend and every other route to the React `build/` output.
   - Increases the Python serverless function limits (duration, memory, bundle size) to handle TensorFlow + the 300 MB `.keras` model. You will need a Vercel Pro plan (or higher) because of the model’s size.
5. Trigger a deployment. Vercel will run both builds in parallel and apply the rewrites automatically.

### Notes on the TensorFlow Model

- The bundled `breast_cancer_model.keras` file is ~300 MB. Keep it inside `backend/models/` or host it externally and point `MODEL_PATH` to the download location (ensure the file is available inside the lambda at runtime).
- Cold starts can take several seconds because TensorFlow + the model have to be loaded into memory. Consider warming the endpoint with a cron ping if response latency matters.

## API Surface

| Method | Path        | Description                         |
|--------|-------------|-------------------------------------|
| `GET`  | `/api/health` | Health check + model load status.   |
| `POST` | `/api/analyze` | Multi-part file upload → JSON response containing probabilities, stats, and Grad-CAM visualisations (base64). |
| `POST` | `/api/report`  | Multi-part file upload → PDF download (generated via ReportLab). |

## Frontend Behaviour

- Automatically detects the correct API base URL (`REACT_APP_API_BASE_URL`, `localhost`, or `/api`).
- Provides status/error messaging, disables controls while the backend is processing, and surfaces all Grad-CAM assets returned by the API.
- The `.env.example` file documents the single environment variable that CRA exposes to the browser.

## Testing Checklist

1. `npm run build` in `frontend/` completes without warnings.
2. `uvicorn main:app --reload` works locally and `/docs` lists all endpoints.
3. `vercel build` (optional) succeeds locally if you have Vercel CLI installed; otherwise rely on the hosted pipeline.
4. After deployment, upload a sample image through the Vercel-hosted UI and verify:
   - `/api/health` returns `{ "status": "ok", "model_loaded": true }`.
   - `/api/analyze` returns the expected payload.
   - `/api/report` downloads a PDF.

## Troubleshooting

- **ModuleNotFoundError for TensorFlow** – ensure the version in `backend/requirements.txt` matches the lambda runtime (Python 3.11 at the time of writing).
- **Lambda bundle too large** – you must be on a Pro/Enterprise plan and may need to host the model externally plus download it at runtime.
- **CORS errors** – set `ALLOWED_ORIGINS` to the deployed frontend domain when splitting the stacks.

Feel free to adapt the deployment strategy (e.g., host the backend on Railway/Render) if Vercel’s limits become restrictive. The codebase now supports both single-origin Vercel deployments and traditional multi-service setups.

