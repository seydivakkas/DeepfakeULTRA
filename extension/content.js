/**
 * DeepfakeULTRA Chrome Extension — Content Script
 * Canvas fallback: CORS engelli görselleri base64'e çevirir.
 */

// Background script'ten gelen mesajları dinle
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "CAPTURE_IMAGE") {
    captureImageAsBase64(message.imageUrl)
      .then((base64) => sendResponse({ success: true, base64 }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true; // async
  }
});

/**
 * Görseli canvas ile base64'e çevir (CORS fallback).
 */
async function captureImageAsBase64(imageUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const base64 = canvas.toDataURL("image/jpeg", 0.95);
        resolve(base64);
      } catch (err) {
        reject(new Error("Canvas çevirimi başarısız: " + err.message));
      }
    };
    img.onerror = () => reject(new Error("Görsel yüklenemedi"));
    img.src = imageUrl;
  });
}
