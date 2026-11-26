"""Automated tests for ReefAugment AI SDXL image generation studio."""

import io
import json
import sys
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
import database
import generation
import main as app_main
import routes


def _png_bytes(width=128, height=128, rgb=(20, 140, 180)) -> bytes:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = rgb
    arr[20:60, 20:80] = (240, 130, 50)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def isolated_gen_env(tmp_path, monkeypatch):
    """Provide isolated SQLite DB, uploads, outputs, and clean diffusion env."""
    db_dir = tmp_path / "database"
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    temp_dir = tmp_path / "temp_exports"
    for d in (db_dir, upload_dir, output_dir, temp_dir):
        d.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite:///{(db_dir / 'test_gen.db').as_posix()}"

    for mod in (config, routes, app_main):
        if hasattr(mod, "UPLOAD_DIR"):
            monkeypatch.setattr(mod, "UPLOAD_DIR", upload_dir)
        if hasattr(mod, "OUTPUT_DIR"):
            monkeypatch.setattr(mod, "OUTPUT_DIR", output_dir)
        if hasattr(mod, "TEMP_DIR"):
            monkeypatch.setattr(mod, "TEMP_DIR", temp_dir)

    monkeypatch.setattr(config, "DB_DIR", db_dir)
    monkeypatch.setattr(config, "DATABASE_URL", db_url)

    for env_key in (
        "DIFFUSION_PROVIDER",
        "DIFFUSION_API_KEY",
        "DIFFUSION_MODEL",
        "DIFFUSION_API_BASE_URL",
        "DIFFUSION_IMG2IMG_URL",
        "HF_API_TOKEN",
        "HUGGINGFACE_API_TOKEN",
        "HF_TOKEN",
        "REPLICATE_API_TOKEN",
        "STABILITY_API_KEY",
    ):
        monkeypatch.delenv(env_key, raising=False)

    test_engine = database.create_engine(
        db_url, connect_args={"check_same_thread": False}, future=True
    )

    @database.event.listens_for(test_engine, "connect")
    def _fk(conn, _):
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    TestSession = database.sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(database, "SessionLocal", TestSession)
    monkeypatch.setattr(routes, "SessionLocal", TestSession)

    import models  # noqa: F401
    database.Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app_main.app.dependency_overrides[database.get_db] = override_get_db

    with TestClient(app_main.app) as client:
        yield client, upload_dir, output_dir

    app_main.app.dependency_overrides.clear()
    test_engine.dispose()


def test_missing_credentials_and_validation(isolated_gen_env):
    client, *_ = isolated_gen_env

    # Status reports unconfigured cleanly without leaking keys
    st = client.get("/api/generation/status")
    assert st.status_code == 200
    body = st.json()
    assert body["configured"] is False
    assert "DIFFUSION_API_KEY" in body["required_env_vars"]

    # Missing credentials on text-to-image returns 503 (never fabricates fake images)
    r_unconf = client.post(
        "/api/generation/text-to-image",
        json={
            "prompt": "clownfish over coral reef",
            "species": "Fish",
            "style": "Photorealistic",
            "width": 1024,
            "height": 1024,
            "num_images": 1,
        },
    )
    assert r_unconf.status_code == 503

    # Validation errors on bad prompt, dimensions, species, style
    assert (
        client.post(
            "/api/generation/text-to-image",
            json={"prompt": "hi", "species": "Fish", "style": "Photorealistic"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/generation/text-to-image",
            json={"prompt": "reef shark swimming", "species": "InvalidCreature", "style": "Photorealistic"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/generation/text-to-image",
            json={"prompt": "reef shark swimming", "species": "Shark", "style": "Photorealistic", "width": 500, "height": 500},
        ).status_code
        == 400
    )

    # Invalid file upload on image-to-image rejected
    r_bad_file = client.post(
        "/api/generation/image-to-image",
        files={"file": ("script.exe", b"not-an-image", "application/octet-stream")},
        data={"prompt": "add reef lighting", "species": "Fish", "style": "Photorealistic"},
    )
    assert r_bad_file.status_code == 400

    # Missing source image on image-to-image rejected
    r_no_img = client.post(
        "/api/generation/image-to-image",
        data={"prompt": "add reef lighting"},
    )
    assert r_no_img.status_code == 400


def test_provider_errors_recorded_in_history(isolated_gen_env, monkeypatch):
    client, *_ = isolated_gen_env
    monkeypatch.setenv("DIFFUSION_PROVIDER", "huggingface")
    monkeypatch.setenv("DIFFUSION_API_KEY", "hf_test_secret_token_never_exposed")

    # Status never leaks actual key
    st = client.get("/api/generation/status").json()
    assert st["configured"] is True
    assert "hf_test_secret_token" not in json.dumps(st)

    def rate_limit_handler(_request):
        return httpx.Response(429, json={"error": "Rate limit exceeded"})

    transport = httpx.MockTransport(rate_limit_handler)
    monkeypatch.setattr(routes, "_get_sdxl_client", lambda: generation.HostedSDXLClient(transport=transport))

    res = client.post(
        "/api/generation/text-to-image",
        json={
            "prompt": "sea turtle gliding above seagrass",
            "species": "Turtle",
            "style": "Scientific Photography",
            "width": 1024,
            "height": 1024,
            "num_images": 1,
        },
    )
    assert res.status_code == 429

    hist = client.get("/api/generation/history?status=failed").json()
    assert hist["total"] == 1
    assert hist["items"][0]["status"] == "failed"
    assert "Rate limit" in hist["items"][0]["error_message"]


def test_text_to_image_storage_gallery_zip_and_augment_pipeline(isolated_gen_env, monkeypatch):
    client, upload_dir, output_dir = isolated_gen_env
    monkeypatch.setenv("DIFFUSION_PROVIDER", "huggingface")
    monkeypatch.setenv("DIFFUSION_API_KEY", "hf_dummy_key")

    png1 = _png_bytes(1024, 1024, (10, 120, 175))
    png2 = _png_bytes(1024, 1024, (25, 160, 150))
    call_counter = {"n": 0}

    def hf_handler(request):
        assert request.headers["Authorization"] == "Bearer hf_dummy_key"
        call_counter["n"] += 1
        return httpx.Response(200, content=png1 if call_counter["n"] == 1 else png2, headers={"content-type": "image/png"})

    transport = httpx.MockTransport(hf_handler)
    monkeypatch.setattr(routes, "_get_sdxl_client", lambda: generation.HostedSDXLClient(transport=transport))

    resp = client.post(
        "/api/generation/text-to-image",
        json={
            "prompt": "bioluminescent jellyfish in deep blue ocean",
            "species": "Jellyfish",
            "style": "Natural Underwater",
            "width": 1024,
            "height": 1024,
            "num_images": 2,
            "steps": 25,
            "guidance_scale": 7.0,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["mode"] == "text_to_image"
    assert payload["is_synthetic"] is True
    assert len(payload["images"]) == 2

    gen_item = payload["images"][0]
    assert (output_dir / gen_item["output_filename"]).exists()
    assert gen_item["is_synthetic"] is True

    # Download generated image
    dl = client.get(gen_item["download_url"])
    assert dl.status_code == 200

    # Integrated in Results Gallery
    gal = client.get("/api/results?technique=sdxl_text_to_image").json()
    assert gal["total"] == 2
    assert gal["items"][0]["origin_type"] == "ai_generated"
    assert gal["items"][0]["is_synthetic"] is True

    # Export in ZIP manifest
    zresp = client.get("/api/export/zip?technique=ai_generated")
    assert zresp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zresp.content)) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["total_images"] == 2
        assert manifest["items"][0]["is_synthetic"] is True
        assert manifest["items"][0]["mode"] == "text_to_image"

    # Send synthetic image into existing 15 augmentation techniques
    sent = client.post(f"/api/generation/{gen_item['id']}/send-to-augment")
    assert sent.status_code == 200
    imported_img = sent.json()
    assert (upload_dir / imported_img["stored_filename"]).exists()

    aug = client.post(
        "/api/augment",
        json={
            "image_id": imported_img["id"],
            "technique": "underwater_haze",
            "parameters": {"intensity": 0.4, "tint": "teal", "depth_gradient": True},
            "save_to_gallery": True,
        },
    )
    assert aug.status_code == 200
    assert (output_dir / aug.json()["output_filename"]).exists()


def test_image_to_image_replicate_preserves_source_and_safe_delete(isolated_gen_env, monkeypatch):
    client, upload_dir, output_dir = isolated_gen_env
    monkeypatch.setenv("DIFFUSION_PROVIDER", "replicate")
    monkeypatch.setenv("DIFFUSION_API_KEY", "r8_dummy_key")

    out_png = _png_bytes(512, 512, (40, 170, 190))

    def replicate_handler(request):
        if request.method == "POST" and str(request.url).endswith("/v1/predictions"):
            return httpx.Response(
                201,
                json={
                    "id": "pred_123",
                    "status": "succeeded",
                    "output": ["https://replicate.delivery/mock/var1.png"],
                },
            )
        if request.method == "GET" and "var1.png" in str(request.url):
            return httpx.Response(200, content=out_png, headers={"content-type": "image/png"})
        return httpx.Response(404)

    transport = httpx.MockTransport(replicate_handler)
    monkeypatch.setattr(routes, "_get_sdxl_client", lambda: generation.HostedSDXLClient(transport=transport))

    source_bytes = _png_bytes(256, 256, (15, 95, 145))
    res = client.post(
        "/api/generation/image-to-image",
        files={"file": ("dolphin_source.png", source_bytes, "image/png")},
        data={
            "prompt": "sunlit ocean surface rays",
            "species": "Dolphin",
            "style": "Photorealistic",
            "strength": "0.55",
            "num_images": "1",
            "steps": "25",
            "guidance_scale": "7.5",
        },
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["mode"] == "image_to_image"
    assert data["source_image"]["original_filename"] == "dolphin_source.png"

    orig_path = upload_dir / data["source_image"]["stored_filename"]
    gen_item = data["images"][0]
    gen_path = output_dir / gen_item["output_filename"]
    assert orig_path.exists()
    assert gen_path.exists()

    # Deleting generated AI image must preserve original uploaded source image
    del_res = client.delete(f"/api/generation/{gen_item['id']}")
    assert del_res.status_code == 200
    assert not gen_path.exists()
    assert orig_path.exists()
