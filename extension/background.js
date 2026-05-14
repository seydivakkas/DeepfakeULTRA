/**
 * DeepfakeULTRA Chrome Extension — Background Service Worker
 * Sağ tık context menü oluşturur ve API ile iletişim kurar.
 */

const API_BASE = "http://localhost:8000";

// ── Context Menü Oluştur ──
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "deepfake-analyze",
    title: "🔍 DeepfakeULTRA — Görüntüyü Analiz Et",
    contexts: ["image"],
  });
});

// ── Context Menü Tıklama ──
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "deepfake-analyze") return;

  const imageUrl = info.srcUrl;
  if (!imageUrl) return;

  // Popup'a "analiz başladı" mesajı gönder
  await chrome.storage.local.set({
    analysisState: "loading",
    analysisResult: null,
    imageUrl: imageUrl,
  });

  // Popup'ı aç
  // Not: Service worker'dan popup açılamaz, badge ile bilgilendir
  chrome.action.setBadgeText({ text: "..." });
  chrome.action.setBadgeBackgroundColor({ color: "#F59E0B" });

  try {
    const result = await analyzeImage(imageUrl);

    await chrome.storage.local.set({
      analysisState: "done",
      analysisResult: result,
      imageUrl: imageUrl,
      analysisTime: new Date().toISOString(),
    });

    // Badge güncelle
    const label = result.label || result.prediction || "?";
    const isFake = label.toUpperCase().includes("FAKE");
    chrome.action.setBadgeText({ text: isFake ? "FAKE" : "REAL" });
    chrome.action.setBadgeBackgroundColor({
      color: isFake ? "#EF4444" : "#22C55E",
    });
  } catch (error) {
    await chrome.storage.local.set({
      analysisState: "error",
      analysisError: error.message,
      imageUrl: imageUrl,
    });
    chrome.action.setBadgeText({ text: "ERR" });
    chrome.action.setBadgeBackgroundColor({ color: "#EF4444" });
  }
});

/**
 * Görseli analiz et — önce URL endpoint, fallback olarak base64 upload.
 */
async function analyzeImage(imageUrl) {
  // Önce sunucu durumunu kontrol et
  try {
    const healthResp = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (!healthResp.ok) throw new Error("Sunucu yanıt vermiyor");
  } catch {
    throw new Error(
      "DeepfakeULTRA sunucusu çalışmıyor. python main.py api komutunu çalıştırın."
    );
  }

  // Yöntem 1: URL ile analiz (hızlı, sunucu görseli indirir)
  try {
    const resp = await fetch(
      `${API_BASE}/predict/url?url=${encodeURIComponent(imageUrl)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }
    );
    if (resp.ok) {
      return await resp.json();
    }
  } catch {
    // URL erişilemiyorsa fallback'e geç
  }

  // Yöntem 2: Görseli indir → base64 → upload (fallback)
  try {
    const imageResp = await fetch(imageUrl);
    const blob = await imageResp.blob();

    const formData = new FormData();
    formData.append("file", blob, "image.jpg");

    const resp = await fetch(`${API_BASE}/predict/image`, {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) throw new Error(`API Hatası: ${resp.status}`);
    return await resp.json();
  } catch (err) {
    throw new Error(`Analiz başarısız: ${err.message}`);
  }
}

// ── Content script mesajlarını dinle (canvas fallback) ──
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ANALYZE_BASE64") {
    // Base64 verisi content script'ten geldi
    fetch(`${API_BASE}/predict/image`, {
      method: "POST",
      body: base64ToFormData(message.base64, message.filename),
    })
      .then((r) => r.json())
      .then((result) => {
        chrome.storage.local.set({
          analysisState: "done",
          analysisResult: result,
          imageUrl: message.imageUrl,
          analysisTime: new Date().toISOString(),
        });
        sendResponse({ success: true, result });
      })
      .catch((err) => {
        sendResponse({ success: false, error: err.message });
      });
    return true; // async response
  }
});

function base64ToFormData(base64, filename) {
  const byteString = atob(base64.split(",")[1]);
  const mimeString = base64.split(",")[0].split(":")[1].split(";")[0];
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  const blob = new Blob([ab], { type: mimeString });
  const formData = new FormData();
  formData.append("file", blob, filename || "image.jpg");
  return formData;
}
