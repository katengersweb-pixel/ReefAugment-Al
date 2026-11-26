"""Pydantic v2 request and response schemas for ReefAugment AI."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TechniqueParameterMeta(BaseModel):
    name: str
    label: str
    type: str  # "float" | "int" | "select" | "bool"
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    description: str = ""


class TechniqueMeta(BaseModel):
    key: str
    name: str
    category: str
    description: str
    parameters: List[TechniqueParameterMeta] = Field(default_factory=list)


class UploadedImageResponse(BaseModel):
    id: int
    original_filename: str
    stored_filename: str
    file_format: str
    file_size: int
    width: int
    height: int
    uploaded_at: datetime
    image_url: str
    augmentation_count: int = 0


class UploadBatchResponse(BaseModel):
    uploaded: List[UploadedImageResponse]
    errors: List[Dict[str, str]] = Field(default_factory=list)


class AugmentationStep(BaseModel):
    technique: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AugmentRequest(BaseModel):
    image_id: int
    technique: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    steps: Optional[List[AugmentationStep]] = None
    save_to_gallery: bool = True


class AugmentPreviewResponse(BaseModel):
    image_id: int
    technique: str
    parameters: Dict[str, Any]
    width: int
    height: int
    file_size: int
    preview_data_url: str


class AugmentationResultResponse(BaseModel):
    id: int
    original_image_id: int
    original_filename: Optional[str] = None
    original_image_url: Optional[str] = None
    job_id: Optional[int] = None
    output_filename: str
    technique: str
    technique_label: str
    parameters: Dict[str, Any]
    width: int
    height: int
    file_size: int
    created_at: datetime
    image_url: str
    download_url: str


class BatchRequest(BaseModel):
    image_ids: List[int] = Field(..., min_length=1)
    techniques: List[str] = Field(..., min_length=1)
    variations_per_technique: int = Field(1, ge=1, le=20)
    technique_parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class BatchItemResult(BaseModel):
    history_id: int
    original_image_id: int
    original_filename: str
    technique: str
    variation_index: int
    status: str
    result_id: Optional[int] = None
    output_filename: Optional[str] = None
    image_url: Optional[str] = None
    error_message: Optional[str] = None


class BatchJobStatusResponse(BaseModel):
    id: int
    job_type: str
    status: str
    total_images: int
    completed_images: int
    failed_images: int
    progress_percent: float
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    items: List[BatchItemResult] = Field(default_factory=list)
    download_zip_url: Optional[str] = None


class ComparisonResponse(BaseModel):
    original: UploadedImageResponse
    results: List[AugmentationResultResponse]


class HistoryEntryResponse(BaseModel):
    id: int
    job_id: Optional[int] = None
    original_image_id: Optional[int] = None
    result_id: Optional[int] = None
    original_filename: str
    output_filename: Optional[str] = None
    technique: str
    technique_label: str
    parameters: Dict[str, Any]
    status: str
    created_at: datetime
    error_message: Optional[str] = None
    result_image_url: Optional[str] = None
    download_url: Optional[str] = None


class DashboardResponse(BaseModel):
    total_uploaded_images: int
    total_augmented_images: int
    total_processing_jobs: int
    total_techniques: int
    total_storage_bytes: int
    recent_uploaded_images: List[UploadedImageResponse]
    recent_generated_images: List[AugmentationResultResponse]
    recent_activity: List[HistoryEntryResponse]
    activity_chart: List[Dict[str, Any]]
    technique_chart: List[Dict[str, Any]]


class SummaryResponse(BaseModel):
    total_original_images: int
    total_generated_images: int
    total_processing_jobs: int
    successful_jobs: int
    failed_jobs: int
    techniques_used_count: int
    total_techniques_available: int
    total_storage_bytes: int
    original_storage_bytes: int
    augmented_storage_bytes: int
    dimensions_overview: Dict[str, Any]
    format_distribution: List[Dict[str, Any]]
    technique_distribution: List[Dict[str, Any]]
    recent_activity: List[HistoryEntryResponse]


class SummaryChartsResponse(BaseModel):
    technique_distribution: List[Dict[str, Any]]
    format_distribution: List[Dict[str, Any]]
    activity_over_time: List[Dict[str, Any]]
    job_status_distribution: List[Dict[str, Any]]


class ZipExportRequest(BaseModel):
    result_ids: Optional[List[int]] = None
    generated_ids: Optional[List[int]] = None
    job_id: Optional[int] = None
    include_generated: bool = True



class GenerationStatusResponse(BaseModel):
    configured: bool
    provider: str
    model: str
    supports_text_to_image: bool
    supports_image_to_image: bool
    required_env_vars: List[str]
    allowed_species: List[str]
    allowed_styles: List[str]
    allowed_dimensions: List[str]
    message: str


class TextToImageRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1000)
    species: str = "Fish"
    style: str = "Photorealistic"
    width: int = 1024
    height: int = 1024
    num_images: int = Field(1, ge=1, le=4)
    negative_prompt: Optional[str] = Field(None, max_length=500)
    steps: int = Field(30, ge=15, le=50)
    guidance_scale: float = Field(7.5, ge=1.0, le=15.0)
    seed: Optional[int] = Field(None, ge=0, le=2147483647)


class GeneratedImageResponse(BaseModel):
    id: int
    job_id: Optional[int] = None
    source_image_id: Optional[int] = None
    source_filename: Optional[str] = None
    source_image_url: Optional[str] = None
    imported_image_id: Optional[int] = None
    mode: str
    species: str
    style: str
    prompt: str
    final_prompt: str
    negative_prompt: Optional[str] = None
    provider: str
    model: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output_filename: Optional[str] = None
    width: int = 0
    height: int = 0
    file_size: int = 0
    is_synthetic: bool = True
    synthetic_label: str = "Synthetic AI Generated (SDXL)"
    disclaimer: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    image_url: Optional[str] = None
    download_url: Optional[str] = None


class GenerationBatchResponse(BaseModel):
    job_id: int
    mode: str
    provider: str
    model: str
    is_synthetic: bool = True
    disclaimer: str
    source_image: Optional[UploadedImageResponse] = None
    images: List[GeneratedImageResponse]
