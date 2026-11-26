"""Application configuration and storage paths for ReefAugment AI."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

DB_DIR = BASE_DIR / "database"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp_exports"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

_raw_cors = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = (
    [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]
    if _raw_cors
    else DEFAULT_CORS_ORIGINS
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{(DB_DIR / 'reefaugment.db').as_posix()}",
)

# Hosted SDXL Diffusion Provider Configuration (credentials remain backend-only)
DIFFUSION_PROVIDER = os.getenv("DIFFUSION_PROVIDER", "huggingface").strip().lower()
DIFFUSION_MODEL = os.getenv(
    "DIFFUSION_MODEL",
    "stabilityai/stable-diffusion-xl-base-1.0",
).strip()
DIFFUSION_API_BASE_URL = os.getenv("DIFFUSION_API_BASE_URL", "").strip()
DIFFUSION_IMG2IMG_URL = os.getenv("DIFFUSION_IMG2IMG_URL", "").strip()
DIFFUSION_REPLICATE_VERSION = os.getenv(
    "DIFFUSION_REPLICATE_VERSION",
    "7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
).strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
DIFFUSION_TIMEOUT_SECONDS = float(os.getenv("DIFFUSION_TIMEOUT_SECONDS", "120"))


def get_diffusion_api_key(provider: str | None = None) -> str:
    """Resolve diffusion API key from backend environment without hardcoding secrets."""
    active_provider = (provider or os.getenv("DIFFUSION_PROVIDER", DIFFUSION_PROVIDER)).strip().lower()
    generic_key = os.getenv("DIFFUSION_API_KEY", "").strip()
    if generic_key:
        return generic_key
    if active_provider == "huggingface":
        return (
            os.getenv("HF_API_TOKEN", "").strip()
            or os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
            or os.getenv("HF_TOKEN", "").strip()
        )
    if active_provider == "replicate":
        return os.getenv("REPLICATE_API_TOKEN", "").strip()
    if active_provider == "stability":
        return os.getenv("STABILITY_API_KEY", "").strip()
    if active_provider == "cloudflare":
        return os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if active_provider == "aihorde":
        return os.getenv("AI_HORDE_API_KEY", "").strip()
    return ""


def ensure_directories() -> None:
    """Ensure required local storage directories exist."""
    for directory in (DB_DIR, UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
