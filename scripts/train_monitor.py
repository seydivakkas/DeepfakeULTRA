"""
Eğitim İlerleme Kontrol Paneli
MLflow metriklerini ve GPU durumunu canlı izler.

Kullanım:
    python scripts/train_monitor.py          # Web paneli (localhost:8788)
    python scripts/train_monitor.py --json   # JSON çıktı
"""

import json
import os
import subprocess
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MODELS_DIR = PROJECT_ROOT / "models"


def get_gpu_info() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "gpu_util": int(parts[0]),
            "mem_used": int(parts[1]),
            "mem_total": int(parts[2]),
            "temp": int(parts[3]),
            "power": float(parts[4]),
        }
    except Exception:
        return {"gpu_util": 0, "mem_used": 0, "mem_total": 0, "temp": 0, "power": 0}


def read_mlflow_metrics() -> dict:
    """MLflow metrics klasöründen epoch metriklerini oku."""
    metrics = {}
    experiment_dirs = sorted(MLRUNS_DIR.glob("*"), key=lambda p: p.stat().st_mtime if p.is_dir() else 0, reverse=True)

    for exp_dir in experiment_dirs:
        if not exp_dir.is_dir() or exp_dir.name == ".trash":
            continue
        run_dirs = sorted(exp_dir.glob("*"), key=lambda p: p.stat().st_mtime if p.is_dir() else 0, reverse=True)
        for run_dir in run_dirs:
            metrics_dir = run_dir / "metrics"
            if not metrics_dir.exists():
                continue
            for metric_file in metrics_dir.iterdir():
                if not metric_file.is_file():
                    continue
                name = metric_file.stem
                values = []
                for line in metric_file.read_text().strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 3:
                        values.append({"step": int(parts[2]), "value": float(parts[1])})
                metrics[name] = sorted(values, key=lambda x: x["step"])
            if metrics:
                return metrics

    return metrics


def get_training_status() -> dict:
    metrics = read_mlflow_metrics()
    gpu = get_gpu_info()

    # Model checkpoint bilgisi
    model_path = MODELS_DIR / "best_model.pth"
    model_info = {}
    if model_path.exists():
        stat = model_path.stat()
        model_info = {
            "size_mb": round(stat.st_size / 1024 / 1024, 1),
            "last_updated": time.strftime("%H:%M:%S", time.localtime(stat.st_mtime)),
        }

    # Son epoch bilgileri
    epochs_completed = 0
    latest = {}
    for key in ["train_loss", "train_acc", "val_loss", "val_acc", "val_auc", "lr"]:
        if key in metrics and metrics[key]:
            vals = metrics[key]
            epochs_completed = max(epochs_completed, vals[-1]["step"] + 1)
            latest[key] = vals[-1]["value"]

    return {
        "timestamp": time.strftime("%H:%M:%S"),
        "gpu": gpu,
        "model": model_info,
        "epochs_completed": epochs_completed,
        "epochs_total": 30,
        "latest": latest,
        "history": metrics,
        "is_running": gpu["gpu_util"] > 10 or gpu["mem_used"] > 1000,
    }


def get_html() -> str:
    return """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>DeepfakeULTRA — Eğitim Monitörü</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a1a; color: #e0e0e0; padding: 1.5rem; }
.container { max-width: 900px; margin: 0 auto; }
h1 { text-align: center; font-size: 1.6rem; background: linear-gradient(135deg, #ff6b35, #f7c948); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { text-align: center; color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
.card { background: #111128; border: 1px solid #222250; border-radius: 10px; padding: 1rem; }
.card.wide { grid-column: 1 / -1; }
.card-title { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem; }
.big-number { font-size: 2.5rem; font-weight: 700; }
.big-number.green { color: #00cc66; }
.big-number.blue { color: #00d4ff; }
.big-number.orange { color: #ff9500; }
.big-number.red { color: #ff4444; }
.stat-row { display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid #1a1a3a; font-size: 0.85rem; }
.stat-label { color: #888; }
.stat-value { font-weight: 600; }
.progress-bar { height: 10px; background: #1a1a3a; border-radius: 5px; overflow: hidden; margin: 0.5rem 0; }
.progress-fill { height: 100%; border-radius: 5px; transition: width 0.5s; background: linear-gradient(90deg, #ff6b35, #f7c948); }
.status-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
.status-badge.running { background: #00cc6633; color: #00cc66; border: 1px solid #00cc6655; }
.status-badge.stopped { background: #ff444433; color: #ff4444; border: 1px solid #ff444455; }
canvas { width: 100% !important; height: 200px !important; }
.chart-container { position: relative; height: 200px; margin-top: 0.5rem; }
.refresh { text-align: center; color: #555; font-size: 0.75rem; margin-top: 1rem; }
.metric-cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-bottom: 1rem; }
.metric-card { background: #111133; border: 1px solid #2a2a5a; border-radius: 8px; padding: 0.8rem; text-align: center; }
.metric-card .label { font-size: 0.7rem; color: #888; text-transform: uppercase; }
.metric-card .value { font-size: 1.3rem; font-weight: 700; margin-top: 0.2rem; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="container">
    <h1>🔥 Eğitim Monitörü</h1>
    <p class="subtitle">DeepfakeULTRA v3 — <span id="time">--:--:--</span> <span class="status-badge running" id="status-badge">ÇALIŞIYOR</span></p>

    <div class="metric-cards">
        <div class="metric-card"><div class="label">Epoch</div><div class="value blue" id="epoch">0/30</div></div>
        <div class="metric-card"><div class="label">Val AUC</div><div class="value green" id="val-auc">—</div></div>
        <div class="metric-card"><div class="label">Train Loss</div><div class="value orange" id="train-loss">—</div></div>
        <div class="metric-card"><div class="label">Val Acc</div><div class="value" id="val-acc">—</div></div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">🎮 GPU Kullanımı</div>
            <div class="big-number blue" id="gpu-util">—%</div>
            <div class="progress-bar"><div class="progress-fill" id="gpu-bar" style="width:0%"></div></div>
            <div class="stat-row"><span class="stat-label">VRAM</span><span class="stat-value" id="vram">— / — MiB</span></div>
            <div class="stat-row"><span class="stat-label">Sıcaklık</span><span class="stat-value" id="temp">—°C</span></div>
            <div class="stat-row"><span class="stat-label">Güç</span><span class="stat-value" id="power">— W</span></div>
        </div>
        <div class="card">
            <div class="card-title">📊 Eğitim İlerlemesi</div>
            <div class="big-number orange" id="progress-pct">0%</div>
            <div class="progress-bar"><div class="progress-fill" id="epoch-bar" style="width:0%"></div></div>
            <div class="stat-row"><span class="stat-label">Model Boyutu</span><span class="stat-value" id="model-size">—</span></div>
            <div class="stat-row"><span class="stat-label">Son Kayıt</span><span class="stat-value" id="model-time">—</span></div>
            <div class="stat-row"><span class="stat-label">LR</span><span class="stat-value" id="lr">—</span></div>
        </div>
        <div class="card wide">
            <div class="card-title">📈 Loss Grafiği</div>
            <div class="chart-container"><canvas id="loss-chart"></canvas></div>
        </div>
        <div class="card wide">
            <div class="card-title">📈 AUC & Accuracy</div>
            <div class="chart-container"><canvas id="auc-chart"></canvas></div>
        </div>
    </div>
    <p class="refresh">Her 10 saniyede otomatik güncellenir</p>
</div>

<script>
const chartOpts = {
    responsive: true, maintainAspectRatio: false,
    scales: { x: { grid: { color: '#1a1a3a' }, ticks: { color: '#888' } },
              y: { grid: { color: '#1a1a3a' }, ticks: { color: '#888' } } },
    plugins: { legend: { labels: { color: '#ccc', font: { size: 11 } } } },
    animation: { duration: 300 }
};

const lossChart = new Chart(document.getElementById('loss-chart'), {
    type: 'line', data: { labels: [], datasets: [
        { label: 'Train Loss', data: [], borderColor: '#ff6b35', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 3 },
        { label: 'Val Loss', data: [], borderColor: '#00d4ff', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 3 }
    ]}, options: chartOpts
});

const aucChart = new Chart(document.getElementById('auc-chart'), {
    type: 'line', data: { labels: [], datasets: [
        { label: 'Val AUC', data: [], borderColor: '#00cc66', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 3 },
        { label: 'Val Acc', data: [], borderColor: '#f7c948', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 3 },
        { label: 'Train Acc', data: [], borderColor: '#ff6b3588', borderWidth: 1, borderDash: [5,5], fill: false, tension: 0.3, pointRadius: 2 }
    ]}, options: chartOpts
});

async function refresh() {
    try {
        const res = await fetch('/api/status');
        const d = await res.json();
        document.getElementById('time').textContent = d.timestamp;
        const badge = document.getElementById('status-badge');
        badge.textContent = d.is_running ? 'ÇALIŞIYOR' : 'DURDU';
        badge.className = 'status-badge ' + (d.is_running ? 'running' : 'stopped');

        document.getElementById('epoch').textContent = d.epochs_completed + '/' + d.epochs_total;
        document.getElementById('val-auc').textContent = d.latest.val_auc ? d.latest.val_auc.toFixed(4) : '—';
        document.getElementById('train-loss').textContent = d.latest.train_loss ? d.latest.train_loss.toFixed(4) : '—';
        document.getElementById('val-acc').textContent = d.latest.val_acc ? (d.latest.val_acc * 100).toFixed(1) + '%' : '—';

        document.getElementById('gpu-util').textContent = d.gpu.gpu_util + '%';
        document.getElementById('gpu-bar').style.width = d.gpu.gpu_util + '%';
        document.getElementById('vram').textContent = d.gpu.mem_used + ' / ' + d.gpu.mem_total + ' MiB';
        document.getElementById('temp').textContent = d.gpu.temp + '°C';
        document.getElementById('power').textContent = d.gpu.power.toFixed(0) + ' W';

        const pct = Math.round(d.epochs_completed / d.epochs_total * 100);
        document.getElementById('progress-pct').textContent = pct + '%';
        document.getElementById('epoch-bar').style.width = pct + '%';
        document.getElementById('model-size').textContent = d.model.size_mb ? d.model.size_mb + ' MB' : '—';
        document.getElementById('model-time').textContent = d.model.last_updated || '—';
        document.getElementById('lr').textContent = d.latest.lr ? d.latest.lr.toExponential(2) : '—';

        // Charts
        const h = d.history;
        if (h.train_loss) {
            const labels = h.train_loss.map(v => 'E' + (v.step + 1));
            lossChart.data.labels = labels;
            lossChart.data.datasets[0].data = h.train_loss.map(v => v.value);
            lossChart.data.datasets[1].data = (h.val_loss || []).map(v => v.value);
            lossChart.update('none');

            aucChart.data.labels = labels;
            aucChart.data.datasets[0].data = (h.val_auc || []).map(v => v.value);
            aucChart.data.datasets[1].data = (h.val_acc || []).map(v => v.value);
            aucChart.data.datasets[2].data = (h.train_acc || []).map(v => v.value);
            aucChart.update('none');
        }
    } catch (e) { console.error(e); }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""


class TrainMonitorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            status = get_training_status()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status, ensure_ascii=False).encode("utf-8"))
        elif self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(get_html().encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Eğitim İlerleme Monitörü")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(get_training_status(), indent=2, ensure_ascii=False))
    else:
        server = HTTPServer(("0.0.0.0", args.port), TrainMonitorHandler)
        print(f"Egitim monitoru baslatildi: http://localhost:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()


if __name__ == "__main__":
    main()
