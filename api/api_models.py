"""
REST API Pydantic Modelleri — Request/Response schemalar.
Swagger dokumantasyonu icin tip guvenligi ve otomatik schema uretimi.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ═══════════════════════════════════════════════════════════
# REQUEST MODELLERİ
# ═══════════════════════════════════════════════════════════

class PredictURLRequest(BaseModel):
    """URL'den gorsel cekip analiz et."""
    url: str = Field(..., description="Analiz edilecek gorsel URL'si")
    tta_count: int = Field(default=5, ge=1, le=15, description="TTA augmentasyon sayisi")


class FeedbackRequest(BaseModel):
    """Kullanici geri bildirimi."""
    analysis_id: int = Field(..., description="Analiz ID")
    label: str = Field(..., pattern="^(REAL|FAKE)$", description="Kullanici etiketi")


class BatchPredictItem(BaseModel):
    """Toplu analiz icin tekil gorsel."""
    filename: str
    base64_data: str = Field(..., description="Base64 kodlu gorsel verisi")


class BatchPredictRequest(BaseModel):
    """Toplu analiz istegi."""
    images: List[BatchPredictItem]
    tta_count: int = Field(default=5, ge=1, le=15)


# ═══════════════════════════════════════════════════════════
# RESPONSE MODELLERİ
# ═══════════════════════════════════════════════════════════

class PredictResponse(BaseModel):
    """Tekil analiz sonucu."""
    verdict: str = Field(..., description="REAL veya FAKE")
    fake_prob: float = Field(..., description="Sahte olasiligi (0-1)")
    real_prob: float = Field(..., description="Gercek olasiligi (0-1)")
    confidence: float = Field(..., description="Guven skoru")
    analysis_id: Optional[int] = Field(None, description="DB kayit ID")
    fake_subtype: Optional[str] = Field(None, description="Alt-tip (digital/physical/ai_generated)")
    forensics: Optional[Dict[str, Any]] = Field(None, description="ELA ve Noise skorlari")
    xai_summary: Optional[Dict[str, float]] = Field(None, description="XAI skorlari")


class BatchPredictResponse(BaseModel):
    """Toplu analiz sonucu."""
    total: int
    results: List[PredictResponse]
    processing_time_sec: float


class ForensicsResponse(BaseModel):
    """Forensik analiz sonucu."""
    ela_score: float = Field(..., description="ELA yogunluk skoru (0-1)")
    noise_score: float = Field(..., description="Gurultu tutarsizlik skoru (0-1)")
    ela_map_base64: Optional[str] = Field(None, description="ELA haritasi (base64 PNG)")
    noise_map_base64: Optional[str] = Field(None, description="Noise haritasi (base64 PNG)")


class AnalyticsSummaryResponse(BaseModel):
    """Analiz istatistikleri."""
    total_analyses: int
    fake_count: int
    real_count: int
    fake_rate_pct: float
    period_days: int


class ROCDataResponse(BaseModel):
    """ROC verisi."""
    fpr: List[float]
    tpr: List[float]
    auc: float
    eer: float
    sample_count: int


class EmbeddingResponse(BaseModel):
    """Embedding verisi."""
    embeddings: List[List[float]]
    labels: List[str]
    count: int


class FeedbackPoolResponse(BaseModel):
    """Feedback havuz durumu."""
    total: int
    real_count: int
    fake_count: int
    ready_for_finetune: bool
    has_finetuned_model: bool


class HealthDetailedResponse(BaseModel):
    """Detayli saglik kontrolu."""
    status: str
    version: str
    system_name: str
    uptime_sec: float
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[float] = None
    feedback_pool_size: int
    total_analyses: int
    timestamp: str
