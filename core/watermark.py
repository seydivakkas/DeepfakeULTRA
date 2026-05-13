"""
Deepfake Detection System v3.0 — Görünmez Watermark
LSB steganografi ile görsel watermark enjeksiyonu.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _text_to_bits(text: str) -> str:
    """Metni binary string'e dönüştür."""
    return ''.join(format(ord(c), '08b') for c in text) + '00000000'  # NULL terminator


def _bits_to_text(bits: str) -> str:
    """Binary string'i metne dönüştür."""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        val = int(byte, 2)
        if val == 0:  # NULL terminator
            break
        chars.append(chr(val))
    return ''.join(chars)


def apply_invisible_watermark(image: Image.Image, text: str = "ANALYZED") -> Image.Image:
    """
    LSB steganografi ile görünmez watermark enjekte et.

    Args:
        image: Kaynak PIL Image
        text: Gömülecek metin

    Returns:
        Watermark eklenmiş PIL Image
    """
    img = image.copy().convert("RGB")
    pixels = np.array(img)
    flat = pixels.flatten()
    bits = _text_to_bits(text)

    if len(bits) > len(flat):
        # Metin çok uzun, kırp
        bits = bits[:len(flat)]

    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)

    pixels = flat.reshape(pixels.shape)
    return Image.fromarray(pixels)


def detect_watermark(image: Image.Image, max_chars: int = 100) -> str:
    """
    LSB watermark'ı oku.

    Args:
        image: Watermarklı PIL Image
        max_chars: Okunacak maksimum karakter

    Returns:
        Çıkarılan metin veya boş string
    """
    img = image.convert("RGB")
    flat = np.array(img).flatten()
    bits = ''.join(str(b & 1) for b in flat[:max_chars * 8 + 8])
    return _bits_to_text(bits)


def apply_visible_watermark(
    image: Image.Image,
    text: str = "DEEPFAKE ANALYZED",
    opacity: float = 0.15,
) -> Image.Image:
    """
    Köşeye yarı saydam görünür watermark ekle.

    Args:
        image: Kaynak PIL Image
        text: Watermark metni
        opacity: Saydamlık (0.0-1.0)

    Returns:
        Watermark eklenmiş PIL Image
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Font boyutu: görsel genişliğinin ~1/15'i
    font_size = max(12, img.width // 15)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Sağ alt köşeye yerleştir
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = img.width - text_w - 10
    y = img.height - text_h - 10

    alpha = int(255 * opacity)
    draw.text((x, y), text, fill=(255, 255, 255, alpha), font=font)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")
