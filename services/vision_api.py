"""
Multi-provider Vision API servisi.
GPT-4o, Claude 3.5, Gemini 2.0 Flash destekli gorsel analiz.
"""
import base64
import json
import re
from pathlib import Path


def _encode_image(image_path: str) -> str:
    """Gorseli base64'e cevir."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _extract_json(text: str) -> dict | None:
    """LLM cevabindan JSON blogu cikar."""
    # ```json ... ``` bloku ara
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Duz JSON ara
    m = re.search(r"\{[^{}]*\"face_detected\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def analyze_with_openai(api_key: str, image_path: str, system_prompt: str) -> tuple[str, dict | None]:
    """OpenAI GPT-4o Vision API ile analiz."""
    import httpx
    b64 = _encode_image(image_path)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Bu yüz görüntüsünü analiz et. ANALYSIS_MODE: FULL, REGION_FOCUS: ALL. "
                                         "JSON formatında çıktı ver."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}", "detail": "high"
                }}
            ]}
        ],
        "max_tokens": 4096,
        "temperature": 0.2
    }
    r = httpx.post("https://api.openai.com/v1/chat/completions",
                   headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return text, _extract_json(text)


def analyze_with_anthropic(api_key: str, image_path: str, system_prompt: str) -> tuple[str, dict | None]:
    """Anthropic Claude 3.5 Vision API ile analiz."""
    import httpx
    b64 = _encode_image(image_path)
    ext = Path(image_path).suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
    media_type = media_map.get(ext, "image/jpeg")
    headers = {
        "x-api-key": api_key, "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": "Bu yüz görüntüsünü analiz et. ANALYSIS_MODE: FULL, REGION_FOCUS: ALL. "
                                     "JSON formatında çıktı ver."}
        ]}]
    }
    r = httpx.post("https://api.anthropic.com/v1/messages",
                   headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    text = r.json()["content"][0]["text"]
    return text, _extract_json(text)


def analyze_with_gemini(api_key: str, image_path: str, system_prompt: str) -> tuple[str, dict | None]:
    """Google Gemini Vision API ile analiz. Yeni google.genai SDK + model fallback."""
    import time
    from google import genai
    from google.genai import types
    import PIL.Image

    client = genai.Client(api_key=api_key)
    img = PIL.Image.open(image_path)

    # Model fallback zinciri
    models = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
    last_error = None

    for model_name in models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        "Bu yüz görüntüsünü analiz et. ANALYSIS_MODE: FULL, REGION_FOCUS: ALL. "
                        "JSON formatında çıktı ver.",
                        img
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2,
                        max_output_tokens=4096,
                    )
                )
                text = response.text
                return text, _extract_json(text)
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    wait = (attempt + 1) * 5
                    time.sleep(wait)
                    continue
                break

    return f"❌ Tüm Gemini modelleri başarısız: {last_error}", None


PROVIDERS = {
    "OpenAI (GPT-4o)": analyze_with_openai,
    "Anthropic (Claude)": analyze_with_anthropic,
    "Google (Gemini)": analyze_with_gemini,
}


def run_craniofacial_analysis(provider: str, api_key: str,
                               image_path: str, system_prompt: str) -> tuple[str, dict | None]:
    """Secilen provider ile kraniyofasiyal analiz calistir."""
    if not api_key or not api_key.strip():
        return "❌ API anahtarı girilmedi.", None
    if not image_path:
        return "❌ Görsel yüklenmedi.", None

    fn = PROVIDERS.get(provider)
    if not fn:
        return f"❌ Bilinmeyen provider: {provider}", None

    try:
        return fn(api_key.strip(), image_path, system_prompt)
    except Exception as e:
        return f"❌ API Hatası ({provider}): {e}", None

