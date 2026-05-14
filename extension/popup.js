/**
 * DeepfakeULTRA Chrome Extension — Popup Logic
 * Storage'dan analiz sonuçlarını okur ve UI'ı günceller.
 */

const API_BASE = "http://localhost:8000";
const GRADIO_URL = "http://localhost:7860";

// ── DOM Elementleri ──
const $ = (id) => document.getElementById(id);

const states = {
  loading: $("loadingState"),
  empty: $("emptyState"),
  error: $("errorState"),
  result: $("resultState"),
};

// ── Sayfa Yüklendiğinde ──
document.addEventListener("DOMContentLoaded", async () => {
  // Sunucu durumunu kontrol et
  await checkServerStatus();

  // Storage'dan son analiz sonucunu oku
  const data = await chrome.storage.local.get([
    "analysisState",
    "analysisResult",
    "analysisError",
    "imageUrl",
    "analysisTime",
  ]);

  showState(data.analysisState || "empty", data);

  // Buton eventleri
  $("btnDetailed").addEventListener("click", () => {
    chrome.tabs.create({ url: GRADIO_URL });
  });

  $("btnCopy").addEventListener("click", () => {
    copyResult(data.analysisResult);
  });
});

// ── Storage değişikliklerini dinle (canlı güncelleme) ──
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;
  if (changes.analysisState) {
    chrome.storage.local
      .get(["analysisState", "analysisResult", "analysisError", "imageUrl", "analysisTime"])
      .then((data) => showState(data.analysisState, data));
  }
});

// ── Sunucu Durumu ──
async function checkServerStatus() {
  const dot = $("serverStatus");
  try {
    const resp = await fetch(`${API_BASE}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(3000),
    });
    if (resp.ok) {
      dot.classList.remove("offline");
      dot.classList.add("online");
      dot.title = "Sunucu aktif ✅";
    } else {
      throw new Error();
    }
  } catch {
    dot.classList.remove("online");
    dot.classList.add("offline");
    dot.title = "Sunucu çevrimdışı ❌";
  }
}

// ── State Gösterimi ──
function showState(state, data) {
  // Tümünü gizle
  Object.values(states).forEach((el) => el.classList.add("hidden"));

  switch (state) {
    case "loading":
      states.loading.classList.remove("hidden");
      break;

    case "done":
      renderResult(data.analysisResult, data.imageUrl, data.analysisTime);
      states.result.classList.remove("hidden");
      break;

    case "error":
      $("errorMessage").textContent = data.analysisError || "Bilinmeyen hata";
      states.error.classList.remove("hidden");
      break;

    default:
      states.empty.classList.remove("hidden");
  }
}

// ── Sonuç Render ──
function renderResult(result, imageUrl, analysisTime) {
  if (!result) return;

  // Thumbnail
  if (imageUrl) {
    $("thumbnailImg").src = imageUrl;
  }

  // Label
  const label = (result.label || result.prediction || "UNKNOWN").toUpperCase();
  const isFake = label.includes("FAKE");
  const isReal = label.includes("REAL");
  const type = isFake ? "fake" : isReal ? "real" : "uncertain";

  const badge = $("resultBadge");
  badge.className = `result-badge ${type}`;
  $("resultLabel").textContent = isFake ? "🔴 FAKE" : isReal ? "🟢 REAL" : "🟡 UNCERTAIN";

  // Confidence
  const fakeProb = result.fake_prob ?? result.fake_probability ?? 0.5;
  const realProb = 1 - fakeProb;
  const confidence = Math.max(fakeProb, realProb) * 100;

  $("confidenceValue").textContent = `${confidence.toFixed(1)}%`;

  const fill = $("confidenceFill");
  fill.style.width = `${confidence}%`;
  fill.className = `confidence-fill ${type}`;

  // Details
  $("fakeProb").textContent = fakeProb.toFixed(4);
  $("realProb").textContent = realProb.toFixed(4);
  $("platform").textContent = result.platform || result.source_platform || "—";

  if (analysisTime) {
    const date = new Date(analysisTime);
    $("analysisTime").textContent = date.toLocaleTimeString("tr-TR");
  }
}

// ── Sonucu Kopyala ──
async function copyResult(result) {
  if (!result) return;

  const label = result.label || result.prediction || "?";
  const fakeProb = result.fake_prob ?? result.fake_probability ?? 0;
  const text = `DeepfakeULTRA Analiz Sonucu\nSonuç: ${label}\nSahte Olasılığı: ${(fakeProb * 100).toFixed(1)}%\nGerçek Olasılığı: ${((1 - fakeProb) * 100).toFixed(1)}%`;

  try {
    await navigator.clipboard.writeText(text);
    $("btnCopy").textContent = "✅ Kopyalandı!";
    setTimeout(() => {
      $("btnCopy").textContent = "📋 Sonucu Kopyala";
    }, 2000);
  } catch {
    // Fallback
    $("btnCopy").textContent = "❌ Kopyalanamadı";
  }
}
