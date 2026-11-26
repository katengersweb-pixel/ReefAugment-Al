"""Hosted Stable Diffusion XL (SDXL) generation client for ReefAugment AI."""

import base64
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

import config

ALLOWED_SPECIES = [
    "Fish",
    "Shark",
    "Turtle",
    "Jellyfish",
    "Dolphin",
    "Coral",
    "Other",
]

ALLOWED_STYLES = [
    "Photorealistic",
    "Scientific Photography",
    "Natural Underwater",
]

ALLOWED_DIMENSIONS: List[Tuple[int, int]] = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (768, 768),
]

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, distorted anatomy, extra fins, watermark, text, logo, oversaturated, "
    "cartoonish, low resolution, artifacts"
)

STYLE_PROMPTS = {
    "Photorealistic": (
        "photorealistic underwater marine photography, natural water-column lighting, "
        "crisp detail, realistic caustic light rays, subtle ocean turbidity"
    ),
    "Scientific Photography": (
        "clear marine biology field survey documentation photograph, natural coloration, "
        "accurate underwater optical illumination, high clarity specimen view"
    ),
    "Natural Underwater": (
        "authentic ambient underwater scuba camera capture, blue-green water depth attenuation, "
        "suspended micro-particles, realistic reef environment"
    ),
}


class GenerationProviderError(Exception):
    """Structured error from the hosted diffusion inference provider."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def get_runtime_diffusion_settings() -> Dict[str, Any]:
    """Read active diffusion settings dynamically from env/config without leaking secrets."""
    provider = (os.getenv("DIFFUSION_PROVIDER") or config.DIFFUSION_PROVIDER or "huggingface").strip().lower()
    default_model_map = {
        "huggingface": "stabilityai/stable-diffusion-xl-base-1.0",
        "replicate": "stability-ai/sdxl",
        "stability": "stable-diffusion-xl-1024-v1-0",
        "custom": "sdxl-custom",
    }
    env_model = os.getenv("DIFFUSION_MODEL", "").strip()
    if env_model:
        model = env_model
    elif provider == config.DIFFUSION_PROVIDER and config.DIFFUSION_MODEL:
        model = config.DIFFUSION_MODEL
    else:
        model = default_model_map.get(provider, "stabilityai/stable-diffusion-xl-base-1.0")

    api_key = config.get_diffusion_api_key(provider)
    base_url = (os.getenv("DIFFUSION_API_BASE_URL") or config.DIFFUSION_API_BASE_URL).strip()
    img2img_url = (os.getenv("DIFFUSION_IMG2IMG_URL") or config.DIFFUSION_IMG2IMG_URL).strip()
    replicate_version = (
        os.getenv("DIFFUSION_REPLICATE_VERSION") or config.DIFFUSION_REPLICATE_VERSION
    ).strip()
    timeout_s = float(os.getenv("DIFFUSION_TIMEOUT_SECONDS") or config.DIFFUSION_TIMEOUT_SECONDS)

    configured = bool(api_key)
    if provider == "custom" and not (base_url or img2img_url):
        configured = False

    supports_t2i = provider in {"huggingface", "replicate", "stability", "custom"}
    if provider in {"replicate", "stability"}:
        supports_i2i = True
    elif provider == "huggingface":
        supports_i2i = bool(img2img_url)
    elif provider == "custom":
        supports_i2i = bool(img2img_url or base_url)
    else:
        supports_i2i = False

    if not api_key:
        message = (
            f"Diffusion API credentials are not configured for provider '{provider}'. "
            "Set DIFFUSION_API_KEY (or provider-specific token) in backend/.env or backend environment variables."
        )
    elif provider == "custom" and not (base_url or img2img_url):
        message = "DIFFUSION_PROVIDER='custom' requires DIFFUSION_API_BASE_URL or DIFFUSION_IMG2IMG_URL."
    elif provider == "huggingface" and not img2img_url:
        message = (
            f"Provider '{provider}' ({model}) is ready for Text-to-Image. "
            "For Image-to-Image, set DIFFUSION_IMG2IMG_URL to a Hugging Face Dedicated Endpoint or switch DIFFUSION_PROVIDER to 'replicate' or 'stability'."
        )
    else:
        message = f"Provider '{provider}' ({model}) is configured and ready for Text-to-Image and Image-to-Image."

    return {
        "configured": configured,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "img2img_url": img2img_url,
        "replicate_version": replicate_version,
        "timeout_seconds": timeout_s,
        "supports_text_to_image": supports_t2i,
        "supports_image_to_image": supports_i2i,
        "required_env_vars": [
            "DIFFUSION_PROVIDER",
            "DIFFUSION_API_KEY",
            "DIFFUSION_MODEL",
        ],
        "message": message,
    }


def build_underwater_prompt(
    user_prompt: str,
    species: str = "Other",
    style: str = "Photorealistic",
    mode: str = "text_to_image",
) -> str:
    """Build a structured SDXL prompt incorporating marine species and visual style."""
    cleaned = (user_prompt or "").strip()
    if species not in ALLOWED_SPECIES:
        raise ValueError(f"Unsupported species '{species}'. Allowed: {ALLOWED_SPECIES}.")
    if style not in ALLOWED_STYLES:
        raise ValueError(f"Unsupported style '{style}'. Allowed: {ALLOWED_STYLES}.")

    style_desc = STYLE_PROMPTS[style]
    species_phrase = (
        f"underwater marine {species.lower()}"
        if species != "Other"
        else "underwater marine organism"
    )

    if mode == "image_to_image":
        base_desc = cleaned if cleaned else f"natural variation of {species_phrase} in marine habitat"
        return f"{base_desc}, {species_phrase}, {style_desc}"

    return f"{species_phrase}, {cleaned}, {style_desc}"


def validate_sdxl_dimensions(width: int, height: int) -> Tuple[int, int]:
    """Validate SDXL output resolution dimensions."""
    if (width, height) not in ALLOWED_DIMENSIONS:
        allowed_str = ", ".join(f"{w}x{h}" for w, h in ALLOWED_DIMENSIONS)
        raise ValueError(f"Unsupported SDXL dimensions {width}x{height}. Allowed: {allowed_str}.")
    return width, height


def _validate_image_bytes_to_png(raw_bytes: bytes) -> Tuple[bytes, int, int]:
    """Validate provider-returned image bytes and normalize to PNG bytes + dimensions."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")
            return buf.getvalue(), rgb.width, rgb.height
    except (UnidentifiedImageError, OSError) as exc:
        raise GenerationProviderError(
            "Inference provider returned invalid or unreadable image data.",
            status_code=502,
        ) from exc


def _raise_for_provider_http_error(resp: httpx.Response, provider: str) -> None:
    """Translate provider HTTP errors into clear user-facing errors without leaking secrets."""
    code = resp.status_code
    detail = ""
    try:
        payload = resp.json()
        detail = (
            payload.get("error")
            or payload.get("detail")
            or payload.get("message")
            or ""
        )
        if isinstance(detail, list):
            detail = "; ".join(str(x) for x in detail)
    except Exception:
        detail = resp.text[:250].strip()

    if code in (401, 403):
        raise GenerationProviderError(
            f"Authentication failed with '{provider}' ({code}). Verify DIFFUSION_API_KEY in backend environment.",
            status_code=502,
        )
    if code == 429:
        raise GenerationProviderError(
            f"Rate limit exceeded on '{provider}'. Please wait a moment and try again.",
            status_code=429,
        )
    if code == 503:
        raise GenerationProviderError(
            f"Hosted model on '{provider}' is warming up or temporarily unavailable ({detail or '503'}). Please retry shortly.",
            status_code=503,
        )
    raise GenerationProviderError(
        f"Diffusion provider '{provider}' returned HTTP {code}: {detail or 'Generation failed.'}",
        status_code=502,
    )


def _prepare_init_image_for_sdxl(source_path, max_dim: int = 1024) -> Tuple[bytes, int, int]:
    """Load source image, resize safely to SDXL-compatible multiple-of-64 dimensions, return PNG bytes."""
    with Image.open(source_path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        w, h = rgb.size
        scale = min(1.0, max_dim / float(max(w, h)))
        target_w = max(512, int(round((w * scale) / 64.0) * 64))
        target_h = max(512, int(round((h * scale) / 64.0) * 64))
        target_w = min(1280, target_w)
        target_h = min(1280, target_h)
        if (target_w, target_h) != (w, h):
            rgb = rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        return buf.getvalue(), target_w, target_h


class HostedSDXLClient:
    """Stateless HTTP client delegating SDXL generation to configured hosted GPU providers."""

    def __init__(self, transport: Optional[httpx.BaseTransport] = None):
        self.transport = transport

    def _client(self, timeout_seconds: float) -> httpx.Client:
        kwargs: Dict[str, Any] = {"timeout": httpx.Timeout(timeout_seconds)}
        if self.transport is not None:
            kwargs["transport"] = self.transport
        return httpx.Client(**kwargs)

    def generate_text_to_image(
        self,
        *,
        final_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_images: int,
        steps: int,
        guidance_scale: float,
        seed: Optional[int],
    ) -> Tuple[List[Tuple[bytes, int, int]], Dict[str, Any]]:
        cfg = get_runtime_diffusion_settings()
        if not cfg["configured"]:
            raise GenerationProviderError(cfg["message"], status_code=503)

        provider = cfg["provider"]
        model = cfg["model"]
        api_key = cfg["api_key"]
        timeout_s = cfg["timeout_seconds"]

        try:
            with self._client(timeout_s) as client:
                if provider == "huggingface":
                    images = self._hf_text_to_image(
                        client, cfg, final_prompt, negative_prompt, width, height, num_images, steps, guidance_scale, seed
                    )
                elif provider == "replicate":
                    images = self._replicate_generate(
                        client,
                        cfg,
                        input_payload={
                            "prompt": final_prompt,
                            "negative_prompt": negative_prompt,
                            "width": width,
                            "height": height,
                            "num_outputs": num_images,
                            "num_inference_steps": steps,
                            "guidance_scale": guidance_scale,
                            **({"seed": seed} if seed is not None else {}),
                        },
                    )
                elif provider == "stability":
                    images = self._stability_text_to_image(
                        client, cfg, final_prompt, negative_prompt, width, height, num_images, steps, guidance_scale, seed
                    )
                elif provider == "custom":
                    images = self._custom_generate(
                        client,
                        url=cfg["base_url"],
                        api_key=api_key,
                        payload={
                            "mode": "text_to_image",
                            "model": model,
                            "prompt": final_prompt,
                            "negative_prompt": negative_prompt,
                            "width": width,
                            "height": height,
                            "num_images": num_images,
                            "num_inference_steps": steps,
                            "guidance_scale": guidance_scale,
                            "seed": seed,
                        },
                    )
                else:
                    raise GenerationProviderError(
                        f"Unsupported DIFFUSION_PROVIDER '{provider}'.", status_code=400
                    )
        except httpx.TimeoutException as exc:
            raise GenerationProviderError(
                f"Diffusion request timed out after {int(timeout_s)}s. Retry or reduce steps/image count.",
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            raise GenerationProviderError(
                f"Network error contacting '{provider}': {exc}",
                status_code=502,
            ) from exc

        return images, {"provider": provider, "model": model}

    def generate_image_to_image(
        self,
        *,
        source_path,
        final_prompt: str,
        negative_prompt: str,
        strength: float,
        num_images: int,
        steps: int,
        guidance_scale: float,
        seed: Optional[int],
    ) -> Tuple[List[Tuple[bytes, int, int]], Dict[str, Any]]:
        cfg = get_runtime_diffusion_settings()
        if not cfg["configured"]:
            raise GenerationProviderError(cfg["message"], status_code=503)
        if not cfg["supports_image_to_image"]:
            raise GenerationProviderError(cfg["message"], status_code=503)

        provider = cfg["provider"]
        model = cfg["model"]
        api_key = cfg["api_key"]
        timeout_s = cfg["timeout_seconds"]
        init_png_bytes, init_w, init_h = _prepare_init_image_for_sdxl(source_path)
        init_b64 = base64.b64encode(init_png_bytes).decode("ascii")

        try:
            with self._client(timeout_s) as client:
                if provider == "replicate":
                    images = self._replicate_generate(
                        client,
                        cfg,
                        input_payload={
                            "prompt": final_prompt,
                            "negative_prompt": negative_prompt,
                            "image": f"data:image/png;base64,{init_b64}",
                            "prompt_strength": strength,
                            "width": init_w,
                            "height": init_h,
                            "num_outputs": num_images,
                            "num_inference_steps": steps,
                            "guidance_scale": guidance_scale,
                            **({"seed": seed} if seed is not None else {}),
                        },
                    )
                elif provider == "stability":
                    images = self._stability_image_to_image(
                        client,
                        cfg,
                        init_png_bytes,
                        final_prompt,
                        negative_prompt,
                        strength,
                        num_images,
                        steps,
                        guidance_scale,
                        seed,
                    )
                elif provider in {"huggingface", "custom"}:
                    target_url = cfg["img2img_url"] or cfg["base_url"]
                    images = self._custom_generate(
                        client,
                        url=target_url,
                        api_key=api_key,
                        payload={
                            "mode": "image_to_image",
                            "model": model,
                            "inputs": final_prompt,
                            "prompt": final_prompt,
                            "negative_prompt": negative_prompt,
                            "image_base64": init_b64,
                            "strength": strength,
                            "num_images": num_images,
                            "num_inference_steps": steps,
                            "guidance_scale": guidance_scale,
                            "seed": seed,
                        },
                    )
                else:
                    raise GenerationProviderError(
                        f"Unsupported DIFFUSION_PROVIDER '{provider}'.", status_code=400
                    )
        except httpx.TimeoutException as exc:
            raise GenerationProviderError(
                f"Image-to-image generation timed out after {int(timeout_s)}s.",
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            raise GenerationProviderError(
                f"Network error contacting '{provider}': {exc}",
                status_code=502,
            ) from exc

        return images, {"provider": provider, "model": model}

    def _hf_text_to_image(
        self,
        client: httpx.Client,
        cfg: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_images: int,
        steps: int,
        guidance_scale: float,
        seed: Optional[int],
    ) -> List[Tuple[bytes, int, int]]:
        url = cfg["base_url"] or f"https://router.huggingface.co/hf-inference/models/{cfg['model']}"
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
            "Accept": "image/png",
        }
        outputs: List[Tuple[bytes, int, int]] = []
        base_seed = seed if seed is not None else int(time.time()) % 1_000_000

        for idx in range(num_images):
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "seed": base_seed + idx,
                },
            }
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                _raise_for_provider_http_error(resp, "huggingface")
            outputs.append(_validate_image_bytes_to_png(resp.content))
        return outputs

    def _replicate_generate(
        self,
        client: httpx.Client,
        cfg: Dict[str, Any],
        input_payload: Dict[str, Any],
    ) -> List[Tuple[bytes, int, int]]:
        base = cfg["base_url"] or "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
            "Prefer": "wait=60",
        }
        resp = client.post(
            base,
            headers=headers,
            json={"version": cfg["replicate_version"], "input": input_payload},
        )
        if resp.status_code not in (200, 201, 202):
            _raise_for_provider_http_error(resp, "replicate")
        prediction = resp.json()

        poll_url = prediction.get("urls", {}).get("get")
        deadline = time.monotonic() + cfg["timeout_seconds"]
        while prediction.get("status") in {"starting", "processing"} and poll_url:
            if time.monotonic() > deadline:
                raise GenerationProviderError(
                    "Replicate SDXL prediction timed out before finishing.",
                    status_code=504,
                )
            time.sleep(1.5)
            poll_resp = client.get(poll_url, headers={"Authorization": f"Bearer {cfg['api_key']}"})
            if poll_resp.status_code != 200:
                _raise_for_provider_http_error(poll_resp, "replicate")
            prediction = poll_resp.json()

        if prediction.get("status") != "succeeded":
            err = prediction.get("error") or f"Prediction ended with status '{prediction.get('status')}'."
            raise GenerationProviderError(f"Replicate SDXL failed: {err}", status_code=502)

        output_urls = prediction.get("output") or []
        if isinstance(output_urls, str):
            output_urls = [output_urls]
        if not output_urls:
            raise GenerationProviderError("Replicate returned no generated images.", status_code=502)

        outputs: List[Tuple[bytes, int, int]] = []
        for img_url in output_urls:
            img_resp = client.get(img_url)
            if img_resp.status_code != 200:
                _raise_for_provider_http_error(img_resp, "replicate")
            outputs.append(_validate_image_bytes_to_png(img_resp.content))
        return outputs

    def _stability_text_to_image(
        self,
        client: httpx.Client,
        cfg: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_images: int,
        steps: int,
        guidance_scale: float,
        seed: Optional[int],
    ) -> List[Tuple[bytes, int, int]]:
        base = cfg["base_url"] or f"https://api.stability.ai/v1/generation/{cfg['model']}/text-to-image"
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: Dict[str, Any] = {
            "text_prompts": [
                {"text": prompt, "weight": 1.0},
                {"text": negative_prompt, "weight": -1.0},
            ],
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": guidance_scale,
            "samples": num_images,
        }
        if seed is not None:
            payload["seed"] = seed

        resp = client.post(base, headers=headers, json=payload)
        if resp.status_code != 200:
            _raise_for_provider_http_error(resp, "stability")
        artifacts = resp.json().get("artifacts") or []
        if not artifacts:
            raise GenerationProviderError("Stability AI returned no image artifacts.", status_code=502)

        outputs: List[Tuple[bytes, int, int]] = []
        for art in artifacts:
            raw = base64.b64decode(art["base64"])
            outputs.append(_validate_image_bytes_to_png(raw))
        return outputs

    def _stability_image_to_image(
        self,
        client: httpx.Client,
        cfg: Dict[str, Any],
        init_png_bytes: bytes,
        prompt: str,
        negative_prompt: str,
        strength: float,
        num_images: int,
        steps: int,
        guidance_scale: float,
        seed: Optional[int],
    ) -> List[Tuple[bytes, int, int]]:
        url = cfg["img2img_url"] or f"https://api.stability.ai/v1/generation/{cfg['model']}/image-to-image"
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Accept": "application/json",
        }
        # Stability image_strength represents source preservation (1.0 - transformation strength)
        image_strength = round(max(0.05, min(0.95, 1.0 - strength)), 2)
        data: Dict[str, Any] = {
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": str(image_strength),
            "text_prompts[0][text]": prompt,
            "text_prompts[0][weight]": "1",
            "text_prompts[1][text]": negative_prompt,
            "text_prompts[1][weight]": "-1",
            "steps": str(steps),
            "cfg_scale": str(guidance_scale),
            "samples": str(num_images),
        }
        if seed is not None:
            data["seed"] = str(seed)

        resp = client.post(
            url,
            headers=headers,
            data=data,
            files={"init_image": ("init.png", init_png_bytes, "image/png")},
        )
        if resp.status_code != 200:
            _raise_for_provider_http_error(resp, "stability")
        artifacts = resp.json().get("artifacts") or []
        if not artifacts:
            raise GenerationProviderError("Stability AI returned no image artifacts.", status_code=502)

        outputs: List[Tuple[bytes, int, int]] = []
        for art in artifacts:
            raw = base64.b64decode(art["base64"])
            outputs.append(_validate_image_bytes_to_png(raw))
        return outputs

    def _custom_generate(
        self,
        client: httpx.Client,
        url: str,
        api_key: str,
        payload: Dict[str, Any],
    ) -> List[Tuple[bytes, int, int]]:
        if not url:
            raise GenerationProviderError("No inference endpoint URL is configured.", status_code=503)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            _raise_for_provider_http_error(resp, "custom")

        content_type = (resp.headers.get("content-type") or "").lower()
        if content_type.startswith("image/"):
            return [_validate_image_bytes_to_png(resp.content)]

        data = resp.json()
        raw_items = data.get("images") or data.get("artifacts") or []
        if isinstance(raw_items, list) and raw_items:
            outputs: List[Tuple[bytes, int, int]] = []
            for item in raw_items:
                b64_str = item.get("base64") if isinstance(item, dict) else str(item)
                if "," in b64_str and b64_str.startswith("data:"):
                    b64_str = b64_str.split(",", 1)[1]
                outputs.append(_validate_image_bytes_to_png(base64.b64decode(b64_str)))
            return outputs

        raise GenerationProviderError(
            "Custom SDXL endpoint response did not contain image bytes or base64 images.",
            status_code=502,
        )
