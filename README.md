# ReefAugment AI

**ReefAugment AI** is a local, full-stack underwater species image augmentation and dataset management platform designed for marine biologists, researchers, students, and computer vision engineers.

Underwater imagery commonly suffers from poor visibility, blue-green color casts, low contrast, optical blur, sensor noise, uneven lighting, and scarce annotated samples. ReefAugment AI helps expand and diversify marine species image datasets locally on your machine—without paid APIs or external cloud services.

---

## Main Features

- **Dashboard**: Real-time statistics from SQLite (total uploads, augmented outputs, processing jobs, storage usage), daily activity timeline, technique breakdown chart, recent image grids, and quick navigation actions.
- **Image Upload**: Drag-and-drop or browse multi-image upload supporting `JPG`, `JPEG`, `PNG`, and `WEBP` (up to 25 MB per file), live client previews with dimensions and size, validation on both frontend and backend, collision-free naming, and safe storage of originals separate from outputs.
- **Augmentation Studio**: Interactive 3-panel workbench supporting **Single Technique** and **Sequential Pipeline** modes across all **15 image augmentation techniques**:
  1. **Horizontal Flip**
  2. **Vertical Flip**
  3. **Rotation** (configurable angle and border mode)
  4. **Zoom & Crop** (adjustable zoom factor, focus center, optional dimension restore)
  5. **Resize** (custom width/height with aspect ratio preservation toggle)
  6. **Brightness Adjustment**
  7. **Contrast Adjustment**
  8. **Saturation Adjustment**
  9. **Hue Adjustment**
  10. **Noise Addition** (Gaussian & speckle noise profiles)
  11. **Blur** (configurable Gaussian kernel radius)
  12. **Underwater Haze** (wavelength-dependent red attenuation, water-column scattering veil, teal/deep-blue/green-murk tints, depth gradient)
  13. **Underwater Particles** (marine snow, plankton specks, and micro-bubbles with density, brightness, and size controls)
  14. **Perspective Transformation** (controlled 4-point camera tilt distortion)
  15. **Elastic Transformation** (smooth water-refraction style deformation bounded to avoid tearing subjects)
- **AI Image Generation Studio (SDXL)**: Generate synthetic underwater species images via hosted **Stable Diffusion XL (SDXL)** inference across two modes:
  - **Text-to-Image**: Prompt + species category (`Fish`, `Shark`, `Turtle`, `Jellyfish`, `Dolphin`, `Coral`, `Other`), visual style (`Photorealistic`, `Scientific Photography`, `Natural Underwater`), SDXL resolution presets, steps, CFG scale, seed, and 1–4 outputs.
  - **Image-to-Image**: Upload a source image (preserved untouched) or pick from dataset uploads, add an optional modification prompt, tune transformation strength (`0.10–0.95`), compare side-by-side, and download variations.
  - **Dataset Integration**: Synthetic outputs are saved in `backend/outputs/`, labeled clearly as synthetic, displayed in **Results Gallery**, exportable in ZIP manifests, and can be sent directly into all **15 augmentation techniques**.
- **Batch Processing**: Select multiple source images, choose multiple techniques, specify variations per technique, tune base parameters, calculate expected total outputs (`images × techniques × variations`), track live background progress, inspect individual item outcomes, retry failed items, and download the entire batch as a ZIP.
- **Results Gallery**: Search by filename, filter by technique, sort newest/oldest, paginate large datasets, inspect lightbox previews, download individual results, multi-select for ZIP export, and safely delete generated results without ever deleting original uploaded images.
- **Before & After Comparison**: Compare original and augmented images side-by-side or with an interactive split-screen slider, inspect dimensions/file sizes/parameters, and download either image.
- **Dataset Summary**: Real SQLite analytics with storage breakdown, image dimension metrics, Recharts Bar/Pie/Line charts, and empty-state guidance.
- **Processing History**: Persistent SQLite audit trail across restarts with search, date range, technique, and status filters, job detail modal, direct downloads, and safe record deletion.
- **ZIP Export with Manifest**: Export all results, a specific batch job, or selected images into a clean ZIP containing `images/...` and a structured `manifest.json` (original filename, generated filename, technique, parameters, timestamp, and job ID).

---

## Technology Stack

| Layer | Technologies |
| --- | --- |
| **Frontend** | React 19, Vite 6, JavaScript, Tailwind CSS v4, React Router DOM, Axios, Lucide React, Recharts |
| **Backend** | Python (3.12+ / 3.14), FastAPI, Uvicorn, OpenCV (`opencv-python-headless`), Albumentations, Pillow, NumPy, python-multipart |
| **Database** | SQLite with SQLAlchemy ORM (`UploadedImage`, `AugmentationResult`, `ProcessingJob`, `ProcessingHistory`, `GeneratedImage`) |
| **Storage** | Local disk (`backend/uploads/` for originals, `backend/outputs/` for augmented outputs, `backend/database/` for SQLite) |

---

## Project Folder Structure

```text
ReefAugment-AI/
├── backend/
│   ├── database/               # SQLite database file (created automatically)
│   ├── uploads/                # Original uploaded images (preserved)
│   ├── outputs/                # Generated augmented images
│   ├── tests/
│   │   ├── test_api.py         # Automated backend augmentation & API test suite
│   │   └── test_generation.py  # Automated SDXL generation validation & integration tests
│   ├── augmentation.py         # 15 techniques, parameter validation & pipelines
│   ├── config.py               # Paths, upload limits, CORS & diffusion env config
│   ├── database.py             # SQLAlchemy engine & session setup
│   ├── generation.py           # Hosted SDXL diffusion provider client
│   ├── main.py                 # FastAPI app, static file mounts & startup
│   ├── models.py               # SQLAlchemy ORM tables
│   ├── routes.py               # REST API endpoints, batch/ZIP & SDXL generation routes
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── underwater_effects.py   # Underwater haze & suspended particles engines
│   ├── utils.py                # Validation, unique naming & path safety helpers
│   ├── requirements.txt        # Python dependencies
│   └── .env.example            # Sample environment configuration
├── frontend/
│   ├── src/
│   │   ├── components/         # ToastContext, ConfirmDialog, TechniqueControls, UiStates
│   │   ├── layouts/            # MainLayout (sidebar, top bar, health check, quick ZIP)
│   │   ├── pages/              # All 9 application pages (including AIGenerationStudio.jsx)
│   │   ├── services/           # Centralized Axios API client (api.js)
│   │   ├── App.jsx             # Router configuration
│   │   ├── main.jsx            # React entrypoint
│   │   └── index.css           # Tailwind CSS configuration
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── start_reefaugment.ps1       # One-click Windows PowerShell launcher
├── README.md
└── .gitignore
```

---

## Prerequisites

- **Windows 10/11** with PowerShell
- **Python 3.12, 3.13, or 3.14** (check exact path if Python is not on your PATH)
- **Node.js 20+** and **npm**

---

## Windows PowerShell Setup Instructions

### 1. Open the project folder

```powershell
cd C:\Users\maddi\ReefAugment-AI
```

### 2. Create and activate the Python virtual environment

If `python` works in PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **Note:** On some Windows systems, typing `python` triggers the Microsoft Store stub (`WindowsApps\python.exe`). If that happens, specify your installed Python executable explicitly, for example:
> ```powershell
> & "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" -m venv .venv
> .\.venv\Scripts\Activate.ps1
> ```

### 3. Install backend dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install frontend dependencies

Open a PowerShell terminal in `frontend`:

```powershell
cd C:\Users\maddi\ReefAugment-AI\frontend
npm install
```

---

## Running the Application Locally

### Option A: One-Click PowerShell Launcher (Recommended)

From the root `ReefAugment-AI` folder:

```powershell
.\start_reefaugment.ps1
```

This opens two PowerShell windows: one running the FastAPI backend and one running the Vite React frontend.

### Option B: Manual Startup (Two Terminals)

**Terminal 1 — Backend (FastAPI + Uvicorn):**

```powershell
cd C:\Users\maddi\ReefAugment-AI\backend
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Frontend (React + Vite):**

```powershell
cd C:\Users\maddi\ReefAugment-AI\frontend
npm run dev
```

---

## Local Application URLs

- **Frontend Web App:** [http://localhost:5173/-REEFAUGMENT-AI/](http://localhost:5173/-REEFAUGMENT-AI/)
- **Backend API Base URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Docs (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## GitHub Pages Deployment (Frontend)

The frontend is pre-configured for GitHub Pages deployment at `https://Nishu1270.github.io/-REEFAUGMENT-AI/`:

- `frontend/vite.config.js` sets `base: '/-REEFAUGMENT-AI/'`.
- `frontend/src/App.jsx` passes `basename={import.meta.env.BASE_URL}` to `<BrowserRouter>`.
- `frontend/src/services/api.js` reads the backend URL from `VITE_API_URL` (`import.meta.env.VITE_API_URL`) and never falls back to `localhost` or `127.0.0.1` in production builds.
- `.github/workflows/deploy.yml` builds `frontend/` and deploys `frontend/dist` (with `404.html` SPA fallback) via official GitHub Pages actions.

### GitHub Repository Settings to Enable

1. Open **Settings → Pages** in `Nishu1270/-REEFAUGMENT-AI` and set **Source** to **GitHub Actions**.
2. Open **Settings → Secrets and variables → Actions → Variables** and create a repository variable named `VITE_API_URL` pointing to your deployed HTTPS FastAPI backend URL (e.g. `https://your-backend-url.example.com`, with no trailing slash). Ensure your deployed backend `CORS_ORIGINS` includes `https://nishu1270.github.io`.
3. Push to `main` (or trigger the **Deploy Frontend to GitHub Pages** workflow manually from the **Actions** tab).

---

## Database & Storage Details

- The SQLite database (`backend/database/reefaugment.db`) is created automatically on backend startup.
- Five relational tables are maintained: `uploaded_images`, `augmentation_results`, `processing_jobs`, `processing_history`, and `generated_images`.
- Original uploads live in `backend/uploads/` and generated outputs live in `backend/outputs/`. Deleting an augmented or AI-generated image from the Results Gallery never deletes its source upload.

---

## AI Image Generation Studio (Hosted SDXL Diffusion)

### How the Diffusion Model Works

ReefAugment AI integrates **Stable Diffusion XL (SDXL)** through a configurable **hosted GPU inference provider** (`backend/generation.py`). Instead of downloading multi-gigabyte diffusion weights locally per request, the FastAPI backend constructs structured marine species prompts, submits inference calls securely with backend-only API tokens, validates returned PNG bytes with Pillow, saves outputs to `backend/outputs/`, and logs generation metadata to SQLite (`generated_images`).

> **Synthetic Data Notice:** All diffusion outputs are labeled `SYNTHETIC AI` (`is_synthetic: true`). They are designed to expand visual diversity for computer vision workflows and are **not** scientifically verified species photographs.

### Provider Configuration & Required Environment Variables

Create `backend/.env` (copy from `backend/.env.example`) and configure your provider credentials on the **backend only** (never expose tokens in `frontend/`):

| Variable | Required | Default / Example | Description |
| --- | --- | --- | --- |
| `DIFFUSION_PROVIDER` | Yes | `cloudflare` | Free-tier: `cloudflare`, `aihorde`; Paid/credits: `replicate`, `stability`, `huggingface`, `custom` |
| `DIFFUSION_API_KEY` | Yes | *(your provider token)* | Secret API key/token for the selected provider |
| `CLOUDFLARE_ACCOUNT_ID` | If `cloudflare` | *(your CF Account ID)* | Required when using Cloudflare Workers AI free tier |
| `DIFFUSION_MODEL` | Optional | `@cf/stabilityai/stable-diffusion-xl-base-1.0` | SDXL model identifier (`AlbedoBase XL (SDXL)` for AI Horde, `stability-ai/sdxl` for Replicate, `stable-diffusion-xl-1024-v1-0` for Stability) |
| `DIFFUSION_TIMEOUT_SECONDS` | Optional | `120` | Request timeout in seconds |
| `DIFFUSION_API_BASE_URL` | Optional | *(empty)* | Custom endpoint URL (required when `DIFFUSION_PROVIDER=custom`) |
| `DIFFUSION_IMG2IMG_URL` | Optional | *(empty)* | Dedicated endpoint URL for Image-to-Image (needed on `huggingface` if not using `cloudflare`, `aihorde`, `replicate`, or `stability`) |

**Provider Capability Notes:**
- **`cloudflare` (Free Tier — No Credit Card):** 10,000 free Neurons/day via Cloudflare Workers AI (`@cf/stabilityai/stable-diffusion-xl-base-1.0`); supports **both** Text-to-Image and Image-to-Image. Requires `CLOUDFLARE_ACCOUNT_ID` + `DIFFUSION_API_KEY`.
- **`aihorde` (Free Community Cluster — No Credit Card):** Crowdsourced SDXL GPUs (`https://stablehorde.net`); supports **both** Text-to-Image and Image-to-Image. Register a free API key at `stablehorde.net/register` (or test at lower resolution with anonymous `0000000000`).
- **`huggingface`**: Text-to-Image via `router.huggingface.co/hf-inference/models/...` (Inference Providers credits required for most models; Image-to-Image needs `DIFFUSION_IMG2IMG_URL`).
- **`replicate` / `stability`**: Paid/credit-based hosted SDXL supporting **both** Text-to-Image and Image-to-Image natively.
- **`custom`**: Calls your own hosted SDXL HTTP endpoint (`DIFFUSION_API_BASE_URL` / `DIFFUSION_IMG2IMG_URL`).

If credentials are missing, `/api/generation/status` returns `configured: false` with setup guidance, and generation requests return HTTP `503` without fabricating fake images.

### Generation API Endpoints

- `GET /api/generation/status` — Returns active provider, model, capability flags, allowed species/styles/dimensions, and configuration readiness (never exposes secrets).
- `POST /api/generation/text-to-image` — Validates prompt, species, visual style, SDXL resolution, steps, CFG scale, seed, and `num_images` (1–4); generates and persists PNG outputs.
- `POST /api/generation/image-to-image` — Accepts `multipart/form-data` with a newly uploaded image `file` or existing `image_id`, optional modification `prompt`, `strength` (`0.10–0.95`), and generation parameters.
- `GET /api/generation/history` — Queries persisted SQLite SDXL generation records (`mode`, `species`, `status`, `search`, `limit`, `offset`).
- `GET /api/generation/{gen_id}/download` — Downloads a generated PNG safely from `backend/outputs/`.
- `POST /api/generation/{gen_id}/send-to-augment` — Registers the synthetic image into `backend/uploads/` (`UploadedImage`) so it can be processed by all 15 augmentation techniques.
- `DELETE /api/generation/{gen_id}` — Deletes a generated output without affecting source uploaded images.

---

## Running Automated Backend Tests

```powershell
cd C:\Users\maddi\ReefAugment-AI\backend
.\.venv\Scripts\python.exe -m pytest -v
```

---

## How to Use Each Feature

1. **Image Upload**: Drag and drop one or multiple underwater species images (`JPG`, `JPEG`, `PNG`, `WEBP`). Verify dimensions and file size in the staging queue, then click **Upload**.
2. **Augmentation Studio**: Pick an uploaded image on the left, select a single technique or toggle **Sequential** mode to chain multiple operations in the center, adjust sliders, click **Preview Result** to inspect without saving or **Generate & Save** to persist to your dataset and gallery.
3. **Batch Processing**: Select multiple images (or add new ones inline), check the desired techniques, configure base parameters via the slider icons, choose variations per technique, inspect the expected image count, and click **Start Batch Processing**. Poll progress live, retry any failed items, or download the batch ZIP.
4. **Results Gallery**: Filter by technique or filename, sort newest/oldest, click thumbnails to zoom in, select multiple results to export a custom ZIP, or safely delete individual generated results.
5. **Before & After Comparison**: Select an original image and any of its augmented variants to view them side-by-side or scrub the interactive split-screen slider.
6. **Dataset Summary**: Review real storage utilization, dimension ranges, format pie chart, technique bar chart, and timeline line chart.
7. **Processing History**: Audit all historical single, pipeline, and batch jobs with filters, detailed parameter inspection modals, downloads, and safe history cleanup.

---

## Troubleshooting Common Issues

- **`python` opens Microsoft Store or fails with `The file cannot be accessed by the system`**:
  Use the full Python path (`$env:LOCALAPPDATA\Programs\Python\Python314\python.exe`) or run `.\backend\.venv\Scripts\python.exe` directly after creating the virtual environment.
- **PowerShell blocks `.ps1` activation (`running scripts is disabled`)**:
  Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once in PowerShell, or execute `.\.venv\Scripts\python.exe -m uvicorn ...` without calling `Activate.ps1`.
- **Frontend shows `Backend Offline`**:
  Ensure Uvicorn is running on `http://127.0.0.1:8000` and check `/api/health`.
- **Port 8000 or 5173 already in use**:
  Stop the conflicting process or change the port and update `VITE_API_URL` / `CORS_ORIGINS` accordingly.
- **Windows Application Control / Smart App Control blocks `stringzilla` (`albucore>=0.0.18`) on Python 3.14**:
  `backend/requirements.txt` intentionally pins `albumentations==1.4.18` and `albucore==0.0.17` (which do not depend on `stringzilla`), keeping the backend compatible with Python 3.14 on Windows without disabling system security policies. If you prefer newer Albumentations releases, create `backend\.venv` with Python 3.12 or 3.13 instead.
