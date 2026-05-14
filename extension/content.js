/**
 * DeepfakeULTRA Chrome Extension — Content Script (localhost:7860)
 * URL parametresinden görseli alır → Gradio upload'a enjekte eder → Analiz Et'e basar.
 */

(function () {
  // Sadece localhost:7860'da çalış
  if (!window.location.hostname.match(/^(localhost|127\.0\.0\.1)$/)) return;
  if (!window.location.port && !window.location.href.includes(":7860")) return;

  // URL'den analiz parametresini al
  const params = new URLSearchParams(window.location.search);
  const imageUrl = params.get("deepfake_analyze");

  if (!imageUrl) return;

  console.log("[DeepfakeULTRA Extension] Görsel analizi başlatılıyor:", imageUrl);

  // Gradio'nun tamamen yüklenmesini bekle
  waitForGradio().then(() => injectAndAnalyze(imageUrl));

  /**
   * Gradio bileşenlerinin DOM'a yüklenmesini bekle.
   */
  function waitForGradio() {
    return new Promise((resolve) => {
      let attempts = 0;
      const maxAttempts = 50; // 10 saniye

      const check = () => {
        attempts++;
        // Gradio file upload alanını ara
        const uploadArea =
          document.querySelector('input[type="file"]') ||
          document.querySelector(".upload-button") ||
          document.querySelector('[data-testid="image"]');

        if (uploadArea || attempts >= maxAttempts) {
          // Ekstra 1.5s bekle — Gradio eventlerinin bağlanması için
          setTimeout(resolve, 1500);
        } else {
          setTimeout(check, 200);
        }
      };
      check();
    });
  }

  /**
   * Görseli indir → Gradio input'una enjekte et → Analiz butonuna bas.
   */
  async function injectAndAnalyze(url) {
    try {
      showNotification("🔄 Görsel indiriliyor...");

      // 1. Görseli indir
      const blob = await fetchImageAsBlob(url);
      const fileName = extractFileName(url);
      const file = new File([blob], fileName, { type: blob.type || "image/jpeg" });

      // 2. Gradio file input'unu bul
      const fileInput = findFileInput();
      if (!fileInput) {
        showNotification("❌ Gradio yükleme alanı bulunamadı", "error");
        return;
      }

      // 3. Dosyayı input'a enjekte et
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      fileInput.files = dataTransfer.files;

      // Change event'i tetikle (Gradio bunu dinler)
      fileInput.dispatchEvent(new Event("change", { bubbles: true }));

      showNotification("📷 Görsel yüklendi, analiz başlatılıyor...");

      // 4. Kısa bekle → "Analiz Et" butonuna bas
      setTimeout(() => {
        clickAnalyzeButton();
        // URL parametresini temizle (tekrar yükleme olmasın)
        cleanUrl();
      }, 2000);
    } catch (err) {
      console.error("[DeepfakeULTRA Extension] Hata:", err);
      showNotification(`❌ Hata: ${err.message}`, "error");
    }
  }

  /**
   * Görseli blob olarak indir.
   */
  async function fetchImageAsBlob(url) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.blob();
    } catch {
      // CORS hatası durumunda proxy dene
      // Doğrudan görseli base64 olarak almaya çalış
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          const canvas = document.createElement("canvas");
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          canvas.getContext("2d").drawImage(img, 0, 0);
          canvas.toBlob(
            (blob) => (blob ? resolve(blob) : reject(new Error("Canvas blob boş"))),
            "image/jpeg",
            0.95
          );
        };
        img.onerror = () => reject(new Error("Görsel yüklenemedi (CORS)"));
        img.src = url;
      });
    }
  }

  /**
   * Gradio file input elementini bul.
   */
  function findFileInput() {
    // Yöntem 1: İlk sekmedeki (Single Image) file input
    const inputs = document.querySelectorAll('input[type="file"]');
    for (const input of inputs) {
      if (input.accept && input.accept.includes("image")) return input;
    }
    // Yöntem 2: Herhangi bir file input
    if (inputs.length > 0) return inputs[0];

    return null;
  }

  /**
   * "Analiz Et" butonuna tıkla.
   */
  function clickAnalyzeButton() {
    // Buton metnine göre ara
    const buttons = document.querySelectorAll("button");
    for (const btn of buttons) {
      const text = btn.textContent.trim().toLowerCase();
      if (
        text.includes("analiz et") ||
        text.includes("analyze") ||
        text.includes("🔬")
      ) {
        console.log("[DeepfakeULTRA Extension] Analiz butonu bulundu, tıklanıyor...");
        btn.click();
        showNotification("🔬 Analiz başlatıldı!", "success");
        return;
      }
    }
    showNotification("⚠️ Analiz butonu bulunamadı — görseli yükledik, manuel olarak tıklayın", "warning");
  }

  /**
   * URL'den dosya adı çıkar.
   */
  function extractFileName(url) {
    try {
      const pathname = new URL(url).pathname;
      const name = pathname.split("/").pop();
      return name && name.includes(".") ? name : "image.jpg";
    } catch {
      return "image.jpg";
    }
  }

  /**
   * URL'den query parametresini temizle.
   */
  function cleanUrl() {
    const url = new URL(window.location.href);
    url.searchParams.delete("deepfake_analyze");
    window.history.replaceState({}, "", url.toString());
  }

  /**
   * Sayfada bildirim göster.
   */
  function showNotification(message, type = "info") {
    // Mevcut bildirimi kaldır
    const existing = document.getElementById("dfx-notification");
    if (existing) existing.remove();

    const div = document.createElement("div");
    div.id = "dfx-notification";
    div.textContent = message;

    const colors = {
      info: { bg: "#1e40af", border: "#3b82f6" },
      success: { bg: "#166534", border: "#22c55e" },
      error: { bg: "#991b1b", border: "#ef4444" },
      warning: { bg: "#92400e", border: "#f59e0b" },
    };
    const c = colors[type] || colors.info;

    Object.assign(div.style, {
      position: "fixed",
      top: "16px",
      right: "16px",
      zIndex: "99999",
      padding: "12px 20px",
      borderRadius: "8px",
      background: c.bg,
      border: `1px solid ${c.border}`,
      color: "#fff",
      fontSize: "14px",
      fontFamily: "system-ui, sans-serif",
      fontWeight: "600",
      boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
      transition: "opacity 0.3s",
    });

    document.body.appendChild(div);

    // 4 saniye sonra kaldır
    setTimeout(() => {
      div.style.opacity = "0";
      setTimeout(() => div.remove(), 300);
    }, 4000);
  }
})();
