/**
 * DeepfakeULTRA Chrome Extension — Background Service Worker
 * Sağ tık → localhost:7860'a yönlendirir ve görseli otomatik analiz ettirir.
 */

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

  // Görseli storage'a kaydet (content script okuyacak)
  await chrome.storage.local.set({
    pendingImageUrl: imageUrl,
    pendingTimestamp: Date.now(),
  });

  // localhost:7860'ı aç (query param ile)
  const gradioUrl = `http://localhost:7860?deepfake_analyze=${encodeURIComponent(imageUrl)}`;

  // Zaten açık bir Gradio sekmesi var mı?
  const tabs = await chrome.tabs.query({ url: "http://localhost:7860/*" });

  if (tabs.length > 0) {
    // Mevcut sekmeyi güncelle ve öne getir
    await chrome.tabs.update(tabs[0].id, { url: gradioUrl, active: true });
    await chrome.windows.update(tabs[0].windowId, { focused: true });
  } else {
    // Yeni sekme aç
    await chrome.tabs.create({ url: gradioUrl });
  }
});
