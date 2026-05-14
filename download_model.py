"""
DeepfakeULTRA -- Model Indirme Scripti
GitHub Releases'tan pretrained model dosyalarini indirir.

Kullanim:
    python download_model.py
    python download_model.py --model best_run5_forensic.pth
    python download_model.py --list
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

# Konsol encoding guvenli cikti
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ======================================================
# Konfigurasyon
# ======================================================
REPO = "seydivakkas/DeepfakeULTRA"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
MODELS_DIR = Path(__file__).parent / "models"


def get_latest_release():
    """GitHub API'den son release bilgilerini al."""
    try:
        req = urllib.request.Request(API_URL, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("[X] Henuz bir GitHub Release bulunamadi.")
            print(f"    Repo: https://github.com/{REPO}/releases")
            sys.exit(1)
        raise


def list_assets(release):
    """Release'deki dosyalari listele."""
    assets = release.get("assets", [])
    if not assets:
        print("[!] Release'te dosya yok.")
        return []

    print(f"\n[i] Release: {release['tag_name']} -- {release['name']}")
    print(f"    Tarih: {release['published_at'][:10]}")
    print(f"\n{'Dosya':<40} {'Boyut':>10}  {'Durum':>10}")
    print("-" * 64)

    for asset in assets:
        name = asset["name"]
        size_mb = asset["size"] / (1024 * 1024)
        local_path = MODELS_DIR / name
        status = "[OK] Mevcut" if local_path.exists() else "[>>] Indir"
        print(f"{name:<40} {size_mb:>7.1f} MB  {status:>10}")

    return assets


def download_asset(asset, force=False):
    """Tek bir asset'i indir."""
    name = asset["name"]
    url = asset["browser_download_url"]
    size = asset["size"]
    dest = MODELS_DIR / name

    if dest.exists() and not force:
        print(f"[OK] {name} zaten mevcut ({dest.stat().st_size / 1024 / 1024:.1f} MB). Atlandi.")
        return True

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[>>] Indiriliyor: {name} ({size / 1024 / 1024:.1f} MB)")
    print(f"     Kaynak: {url}")
    print(f"     Hedef:  {dest}")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", size))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB

            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = downloaded / total * 100
                    bar_len = 30
                    filled = int(bar_len * downloaded / total)
                    bar = "#" * filled + "." * (bar_len - filled)
                    sys.stdout.write(f"\r     [{bar}] {pct:5.1f}%  {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB")
                    sys.stdout.flush()

        print(f"\n[OK] {name} indirildi.")
        return True

    except Exception as e:
        print(f"\n[X] Indirme hatasi: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DeepfakeULTRA model indirici")
    parser.add_argument("--model", type=str, default=None,
                        help="Indirilecek model dosyasi (orn: best_run5_forensic.pth)")
    parser.add_argument("--list", action="store_true",
                        help="Mevcut dosyalari listele")
    parser.add_argument("--force", action="store_true",
                        help="Mevcut dosyalari yeniden indir")
    parser.add_argument("--all", action="store_true",
                        help="Tum dosyalari indir")
    args = parser.parse_args()

    print("DeepfakeULTRA -- Model Indirici")
    print(f"   Repo: https://github.com/{REPO}")
    print(f"   Hedef: {MODELS_DIR}/")

    release = get_latest_release()
    assets = list_assets(release)

    if not assets:
        return

    if args.list:
        return

    # Indirilecek dosyalari belirle
    if args.model:
        targets = [a for a in assets if a["name"] == args.model]
        if not targets:
            print(f"\n[X] '{args.model}' bulunamadi. --list ile mevcut dosyalari gorun.")
            return
    elif args.all:
        targets = assets
    else:
        # Varsayilan: sadece .pth dosyalari
        targets = [a for a in assets if a["name"].endswith(".pth")]
        if not targets:
            targets = assets

    print(f"\n[>>] {len(targets)} dosya indirilecek...")

    success = 0
    for asset in targets:
        if download_asset(asset, force=args.force):
            success += 1

    print(f"\n{'='*50}")
    print(f"[OK] {success}/{len(targets)} dosya basariyla indirildi.")
    print(f"[*] Simdi uygulamayi baslatabilirsiniz: python app.py")


if __name__ == "__main__":
    main()
