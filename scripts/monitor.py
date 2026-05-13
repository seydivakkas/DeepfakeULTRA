"""
Pipeline İlerleme Kontrol Paneli
Gerçek zamanlı olarak veri hazırlama pipeline'ının durumunu gösterir.

Kullanım:
    python scripts/monitor.py          # Terminal modu (varsayılan)
    python scripts/monitor.py --web    # Tarayıcı paneli (localhost:8787)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov"}


def count_files(directory: Path, extensions: set) -> int:
    if not directory.exists():
        return 0
    count = 0
    for ext in extensions:
        count += len(list(directory.rglob(f"*{ext}")))
    return count


def dir_size_gb(directory: Path) -> float:
    if not directory.exists():
        return 0
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def get_status() -> dict:
    """Pipeline durumunu topla."""
    status = {
        "timestamp": time.strftime("%H:%M:%S"),
        "stages": {},
        "overall_progress": 0,
        "current_stage": "Bilinmiyor",
    }

    # 1. Raw videolar
    raw_ffpp = DATASET_DIR / "raw" / "ffpp"
    raw_antispoof = DATASET_DIR / "raw" / "antispoof"
    ffpp_videos = count_files(raw_ffpp, VIDEO_EXTS)
    antispoof_files = count_files(raw_antispoof, VIDEO_EXTS | IMAGE_EXTS)

    status["stages"]["download"] = {
        "name": "1. Indirme",
        "ffpp_videos": ffpp_videos,
        "ffpp_target": 6000,
        "ffpp_size_gb": round(dir_size_gb(raw_ffpp), 2),
        "antispoof_files": antispoof_files,
        "done": ffpp_videos >= 6000 and antispoof_files > 0,
    }

    # 2. Frame cikarma
    frames_ffpp = DATASET_DIR / "_raw_frames" / "ffpp"
    frames_antispoof = DATASET_DIR / "_raw_frames" / "antispoof"
    ffpp_frames = count_files(frames_ffpp, IMAGE_EXTS)
    antispoof_frames = count_files(frames_antispoof, IMAGE_EXTS)
    ffpp_frame_target = ffpp_videos * 30

    status["stages"]["extract"] = {
        "name": "2. Frame Cikarma",
        "ffpp_frames": ffpp_frames,
        "ffpp_target": ffpp_frame_target,
        "antispoof_frames": antispoof_frames,
        "done": ffpp_frames >= ffpp_frame_target * 0.95 if ffpp_frame_target > 0 else False,
    }

    # 3. Yuz kirpma
    cropped_ffpp = DATASET_DIR / "_cropped_faces" / "ffpp"
    cropped_antispoof = DATASET_DIR / "_cropped_faces" / "antispoof"
    ffpp_cropped = count_files(cropped_ffpp, IMAGE_EXTS)
    antispoof_cropped = count_files(cropped_antispoof, IMAGE_EXTS)

    status["stages"]["crop"] = {
        "name": "3. Yuz Kirpma",
        "ffpp_cropped": ffpp_cropped,
        "ffpp_source": ffpp_frames,
        "antispoof_cropped": antispoof_cropped,
        "done": ffpp_cropped > 0 and ffpp_cropped >= ffpp_frames * 0.9 if ffpp_frames > 0 else False,
    }

    # 4. Split
    split_ffpp = DATASET_DIR / "deepfake" / "ff++"
    split_antispoof = DATASET_DIR / "liveness" / "casia-fasd"

    split_counts = {}
    for name, base in [("ffpp", split_ffpp), ("antispoof", split_antispoof)]:
        for split in ["train", "val", "test"]:
            split_dir = base / split
            if split_dir.exists():
                for label_dir in split_dir.iterdir():
                    if label_dir.is_dir():
                        key = f"{name}_{split}_{label_dir.name}"
                        split_counts[key] = len([f for f in label_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])

    total_split = sum(split_counts.values())
    status["stages"]["split"] = {
        "name": "4. Train/Val/Test Bolme",
        "total_files": total_split,
        "details": split_counts,
        "done": total_split > 1000,
    }

    # Genel ilerleme
    stages_done = sum(1 for s in status["stages"].values() if s["done"])
    status["overall_progress"] = round(stages_done / 4 * 100)

    # Aktif asama
    if not status["stages"]["download"]["done"]:
        status["current_stage"] = "Indirme"
    elif not status["stages"]["extract"]["done"]:
        pct = round(ffpp_frames / max(ffpp_frame_target, 1) * 100, 1)
        status["current_stage"] = f"Frame Cikarma (%{pct})"
    elif not status["stages"]["crop"]["done"]:
        pct = round(ffpp_cropped / max(ffpp_frames, 1) * 100, 1)
        status["current_stage"] = f"Yuz Kirpma (%{pct})"
    elif not status["stages"]["split"]["done"]:
        status["current_stage"] = "Split"
    else:
        status["current_stage"] = "TAMAMLANDI"
        status["overall_progress"] = 100

    return status


def print_terminal(status: dict):
    """Terminal paneli."""
    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 60)
    print("  DEEPFAKE ULTRA — Pipeline Kontrol Paneli")
    print(f"  Son guncelleme: {status['timestamp']}")
    print("=" * 60)

    # Genel ilerleme cubugu
    pct = status["overall_progress"]
    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  Genel: [{bar}] %{pct}")
    print(f"  Aktif: {status['current_stage']}")

    # Asamalar
    dl = status["stages"]["download"]
    ext = status["stages"]["extract"]
    crop = status["stages"]["crop"]
    split = status["stages"]["split"]

    print(f"\n  {'─' * 56}")

    # 1. Indirme
    icon = "✅" if dl["done"] else "⏳"
    print(f"  {icon} {dl['name']}")
    print(f"     FF++: {dl['ffpp_videos']}/{dl['ffpp_target']} video ({dl['ffpp_size_gb']} GB)")
    print(f"     Anti-Spoofing: {dl['antispoof_files']} dosya")

    # 2. Frame cikarma
    icon = "✅" if ext["done"] else ("⏳" if ext["ffpp_frames"] > 0 else "⬚")
    pct_ext = round(ext["ffpp_frames"] / max(ext["ffpp_target"], 1) * 100, 1)
    print(f"\n  {icon} {ext['name']}")
    bar_f = int(20 * pct_ext / 100)
    bar_str = "█" * bar_f + "░" * (20 - bar_f)
    print(f"     FF++: [{bar_str}] {ext['ffpp_frames']}/{ext['ffpp_target']} (%{pct_ext})")
    print(f"     Anti-Spoofing: {ext['antispoof_frames']} frame")

    # 3. Yuz kirpma
    icon = "✅" if crop["done"] else ("⏳" if crop["ffpp_cropped"] > 0 else "⬚")
    pct_crop = round(crop["ffpp_cropped"] / max(crop["ffpp_source"], 1) * 100, 1) if crop["ffpp_source"] > 0 else 0
    print(f"\n  {icon} {crop['name']}")
    bar_c = int(20 * pct_crop / 100)
    bar_str = "█" * bar_c + "░" * (20 - bar_c)
    print(f"     FF++: [{bar_str}] {crop['ffpp_cropped']}/{crop['ffpp_source']} (%{pct_crop})")
    print(f"     Anti-Spoofing: {crop['antispoof_cropped']} yuz")

    # 4. Split
    icon = "✅" if split["done"] else "⬚"
    print(f"\n  {icon} {split['name']}")
    print(f"     Toplam: {split['total_files']} dosya")

    print(f"\n  {'─' * 56}")
    print(f"  Yenilemek icin bekleyin (5 sn)... Cikmak: Ctrl+C")


def get_web_html() -> str:
    """Web paneli HTML."""
    return """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepfakeULTRA Pipeline Monitor</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0a0a1a;
    color: #e0e0e0;
    min-height: 100vh;
    padding: 2rem;
}
.container { max-width: 800px; margin: 0 auto; }
h1 {
    text-align: center;
    font-size: 1.8rem;
    background: linear-gradient(135deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.subtitle { text-align: center; color: #888; font-size: 0.9rem; margin-bottom: 2rem; }
.overall {
    background: #111133;
    border: 1px solid #2a2a5a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.overall-label { font-size: 0.9rem; color: #aaa; margin-bottom: 0.5rem; }
.overall-bar {
    height: 28px;
    background: #1a1a3a;
    border-radius: 14px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.overall-fill {
    height: 100%;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7);
    border-radius: 14px;
    transition: width 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    color: #fff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
}
.current-stage {
    font-size: 1.1rem;
    color: #00d4ff;
    margin-top: 0.5rem;
}
.stage {
    background: #111128;
    border: 1px solid #222250;
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s;
}
.stage.active { border-color: #00d4ff; box-shadow: 0 0 15px rgba(0,212,255,0.1); }
.stage.done { border-color: #00cc66; }
.stage-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
}
.stage-icon { font-size: 1.3rem; }
.stage-name { font-weight: 600; font-size: 1rem; }
.stage-bar {
    height: 8px;
    background: #1a1a3a;
    border-radius: 4px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.stage-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}
.stage-fill.blue { background: linear-gradient(90deg, #00d4ff, #0099cc); }
.stage-fill.green { background: linear-gradient(90deg, #00cc66, #00aa55); }
.stage-fill.purple { background: linear-gradient(90deg, #7b2ff7, #5a1fd4); }
.stage-fill.orange { background: linear-gradient(90deg, #ff9500, #ff6600); }
.stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #aaa;
    margin-top: 0.3rem;
}
.stat-value { color: #fff; font-weight: 500; }
.refresh-note { text-align: center; color: #555; font-size: 0.8rem; margin-top: 1.5rem; }
</style>
</head>
<body>
<div class="container">
    <h1>🧠 DeepfakeULTRA</h1>
    <p class="subtitle">Pipeline Kontrol Paneli — <span id="time">--:--:--</span></p>

    <div class="overall">
        <div class="overall-label">Genel İlerleme</div>
        <div class="overall-bar">
            <div class="overall-fill" id="overall-fill" style="width: 0%">0%</div>
        </div>
        <div class="current-stage" id="current-stage">Yükleniyor...</div>
    </div>

    <div class="stage" id="stage-download">
        <div class="stage-header">
            <span class="stage-icon" id="icon-download">⬚</span>
            <span class="stage-name">1. İndirme</span>
        </div>
        <div class="stage-bar"><div class="stage-fill blue" id="bar-download" style="width: 0%"></div></div>
        <div class="stat-row"><span>FF++ Video</span><span class="stat-value" id="dl-ffpp">0 / 6000</span></div>
        <div class="stat-row"><span>FF++ Boyut</span><span class="stat-value" id="dl-size">0 GB</span></div>
        <div class="stat-row"><span>Anti-Spoofing</span><span class="stat-value" id="dl-antispoof">0 dosya</span></div>
    </div>

    <div class="stage" id="stage-extract">
        <div class="stage-header">
            <span class="stage-icon" id="icon-extract">⬚</span>
            <span class="stage-name">2. Frame Çıkarma</span>
        </div>
        <div class="stage-bar"><div class="stage-fill purple" id="bar-extract" style="width: 0%"></div></div>
        <div class="stat-row"><span>FF++ Frame</span><span class="stat-value" id="ext-ffpp">0 / 0</span></div>
        <div class="stat-row"><span>Anti-Spoofing</span><span class="stat-value" id="ext-antispoof">0 frame</span></div>
    </div>

    <div class="stage" id="stage-crop">
        <div class="stage-header">
            <span class="stage-icon" id="icon-crop">⬚</span>
            <span class="stage-name">3. Yüz Kırpma</span>
        </div>
        <div class="stage-bar"><div class="stage-fill orange" id="bar-crop" style="width: 0%"></div></div>
        <div class="stat-row"><span>FF++ Yüz</span><span class="stat-value" id="crop-ffpp">0 / 0</span></div>
        <div class="stat-row"><span>Anti-Spoofing</span><span class="stat-value" id="crop-antispoof">0 yüz</span></div>
    </div>

    <div class="stage" id="stage-split">
        <div class="stage-header">
            <span class="stage-icon" id="icon-split">⬚</span>
            <span class="stage-name">4. Train / Val / Test Bölme</span>
        </div>
        <div class="stage-bar"><div class="stage-fill green" id="bar-split" style="width: 0%"></div></div>
        <div class="stat-row"><span>Toplam</span><span class="stat-value" id="split-total">0 dosya</span></div>
    </div>

    <p class="refresh-note">Her 5 saniyede otomatik güncellenir</p>
</div>

<script>
async function refresh() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        document.getElementById('time').textContent = data.timestamp;
        document.getElementById('overall-fill').style.width = data.overall_progress + '%';
        document.getElementById('overall-fill').textContent = data.overall_progress + '%';
        document.getElementById('current-stage').textContent = '▶ ' + data.current_stage;

        const dl = data.stages.download;
        const dlPct = Math.round(dl.ffpp_videos / dl.ffpp_target * 100);
        document.getElementById('bar-download').style.width = dlPct + '%';
        document.getElementById('dl-ffpp').textContent = dl.ffpp_videos + ' / ' + dl.ffpp_target;
        document.getElementById('dl-size').textContent = dl.ffpp_size_gb + ' GB';
        document.getElementById('dl-antispoof').textContent = dl.antispoof_files + ' dosya';
        setStageState('download', dl.done, dl.ffpp_videos > 0);

        const ext = data.stages.extract;
        const extPct = ext.ffpp_target > 0 ? Math.round(ext.ffpp_frames / ext.ffpp_target * 100) : 0;
        document.getElementById('bar-extract').style.width = Math.min(extPct, 100) + '%';
        document.getElementById('ext-ffpp').textContent = ext.ffpp_frames.toLocaleString() + ' / ' + ext.ffpp_target.toLocaleString();
        document.getElementById('ext-antispoof').textContent = ext.antispoof_frames.toLocaleString() + ' frame';
        setStageState('extract', ext.done, ext.ffpp_frames > 0);

        const crop = data.stages.crop;
        const cropPct = crop.ffpp_source > 0 ? Math.round(crop.ffpp_cropped / crop.ffpp_source * 100) : 0;
        document.getElementById('bar-crop').style.width = Math.min(cropPct, 100) + '%';
        document.getElementById('crop-ffpp').textContent = crop.ffpp_cropped.toLocaleString() + ' / ' + crop.ffpp_source.toLocaleString();
        document.getElementById('crop-antispoof').textContent = crop.antispoof_cropped.toLocaleString() + ' yüz';
        setStageState('crop', crop.done, crop.ffpp_cropped > 0);

        const split = data.stages.split;
        const splitPct = split.done ? 100 : 0;
        document.getElementById('bar-split').style.width = splitPct + '%';
        document.getElementById('split-total').textContent = split.total_files.toLocaleString() + ' dosya';
        setStageState('split', split.done, split.total_files > 0);

    } catch (e) {
        document.getElementById('current-stage').textContent = '⚠ Bağlantı hatası';
    }
}

function setStageState(id, done, active) {
    const el = document.getElementById('stage-' + id);
    const icon = document.getElementById('icon-' + id);
    el.classList.remove('done', 'active');
    if (done) { el.classList.add('done'); icon.textContent = '✅'; }
    else if (active) { el.classList.add('active'); icon.textContent = '⏳'; }
    else { icon.textContent = '⬚'; }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


class MonitorHandler(SimpleHTTPRequestHandler):
    """Web sunucu handler."""

    def do_GET(self):
        if self.path == "/api/status":
            status = get_status()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(status, ensure_ascii=False).encode("utf-8"))
        elif self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(get_web_html().encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Sessiz mod


def run_terminal_mode():
    """Terminal modunda canlı izleme."""
    print("Pipeline izleme baslatiliyor... (Ctrl+C ile cik)")
    try:
        while True:
            status = get_status()
            print_terminal(status)
            if status["current_stage"] == "TAMAMLANDI":
                print("\n  🎉 TUM ASAMALAR TAMAMLANDI!")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n  Izleme durduruldu.")


def run_web_mode(port: int = 8787):
    """Web paneli modunda canlı izleme."""
    server = HTTPServer(("0.0.0.0", port), MonitorHandler)
    print(f"Pipeline kontrol paneli baslatildi!")
    print(f"  Tarayicida ac: http://localhost:{port}")
    print(f"  Durdurmak icin: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        print("\n  Sunucu durduruldu.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Ilerleme Kontrol Paneli")
    parser.add_argument("--web", action="store_true", help="Tarayici paneli (localhost:8787)")
    parser.add_argument("--port", type=int, default=8787, help="Web sunucu portu")
    parser.add_argument("--json", action="store_true", help="JSON cikti (tek seferlik)")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(get_status(), indent=2, ensure_ascii=False))
    elif args.web:
        run_web_mode(args.port)
    else:
        run_terminal_mode()


if __name__ == "__main__":
    main()
