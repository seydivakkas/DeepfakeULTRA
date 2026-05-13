"""
Deepfake Detection System v3 — Ana Giriş Noktası (CLI)
Kullanım:
    python main.py demo              → Gradio UI başlat
    python main.py train             → Model eğitimi
    python main.py eval              → Model değerlendirmesi
    python main.py api               → FastAPI sunucusu
    python main.py predict -i img.jpg → Tek görüntü tahmini
    python main.py --test-random     → Rastgele girdi ile model testi
"""

import argparse
import sys
from pathlib import Path


def cmd_demo(args):
    """Gradio demo arayüzünü başlat."""
    from app import create_app
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share)


def cmd_train(args):
    """Model eğitimini başlat."""
    from core.trainer import train_and_evaluate
    train_and_evaluate(
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=args.resume,
    )


def cmd_eval(args):
    """Eğitilmiş modeli değerlendir."""
    from core.evaluation import run_evaluation
    run_evaluation(model_path=args.model)


def cmd_api(args):
    """FastAPI sunucusunu başlat."""
    import uvicorn
    from config import api_cfg
    uvicorn.run(
        "api.server:app",
        host=api_cfg.HOST,
        port=args.port or api_cfg.PORT,
        reload=args.reload,
    )


def cmd_predict(args):
    """Tek bir görüntü üzerinde tahmin yap."""
    from inference.predictor import DeepfakePredictor

    predictor = DeepfakePredictor()
    result = predictor.predict(args.image)

    print(f"\n{'='*50}")
    print(f"📊 Sonuç: {result['label']}")
    print(f"   Sahte Olasılığı: {result['fake_prob']:.4f}")
    print(f"   Gerçek Olasılığı: {result['real_prob']:.4f}")
    print(f"   GradCAM++ Skoru: {result.get('cam_score', 'N/A')}")
    print(f"{'='*50}\n")

    if args.xai:
        print("🔍 XAI haritaları oluşturuluyor...")
        from inference.xai_module import generate_xai_maps
        generate_xai_maps(args.image, result)

    if args.report:
        print("📄 PDF rapor oluşturuluyor...")
        from services.pdf_report import generate_report
        generate_report(args.image, result)


def cmd_test_random(args):
    """Rastgele girdi ile model mimarisini test et."""
    import torch
    from config import DEVICE, model_cfg

    print(f"🔧 Cihaz: {DEVICE}")
    print(f"🧪 Rastgele girdi ile model testi başlatılıyor...\n")

    try:
        from core.dual_mobilenetv3 import DualPathDeepfakeDetector
        model = DualPathDeepfakeDetector().to(DEVICE)

        # Rastgele girdiler
        batch = 2
        rgb = torch.randn(batch, 3, model_cfg.IMG_SIZE, model_cfg.IMG_SIZE).to(DEVICE)
        freq = torch.randn(batch, model_cfg.DWT_CHANNELS, model_cfg.IMG_SIZE, model_cfg.IMG_SIZE).to(DEVICE)
        mesh = torch.randn(batch, model_cfg.MESH_INPUT_DIM).to(DEVICE)

        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(rgb, freq, mesh)

        probs = torch.softmax(logits, dim=1)
        print(f"✅ Model forward pass başarılı!")
        print(f"   Çıktı boyutu: {logits.shape}")
        print(f"   Olasılıklar: {probs.cpu().numpy()}")

        # Parametre sayısı
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 Parametre sayısı: {total:,} (eğitilebilir: {trainable:,})")

    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="🔬 Deepfake Detection System v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Komut seçin")

    # demo
    p_demo = subparsers.add_parser("demo", help="Gradio UI başlat")
    p_demo.add_argument("--share", action="store_true", help="Public link oluştur")
    p_demo.set_defaults(func=cmd_demo)

    # train
    p_train = subparsers.add_parser("train", help="Model eğit")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--resume", type=str, default=None, help="Checkpoint yolu")
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = subparsers.add_parser("eval", help="Model değerlendir")
    p_eval.add_argument("--model", type=str, default=None, help="Model dosya yolu")
    p_eval.set_defaults(func=cmd_eval)

    # api
    p_api = subparsers.add_parser("api", help="FastAPI sunucusu")
    p_api.add_argument("--port", type=int, default=None)
    p_api.add_argument("--reload", action="store_true")
    p_api.set_defaults(func=cmd_api)

    # predict
    p_pred = subparsers.add_parser("predict", help="Tek görüntü tahmini")
    p_pred.add_argument("-i", "--image", required=True, help="Görüntü yolu")
    p_pred.add_argument("--xai", action="store_true", help="XAI haritaları oluştur")
    p_pred.add_argument("--report", action="store_true", help="PDF rapor oluştur")
    p_pred.set_defaults(func=cmd_predict)

    # test-random
    parser.add_argument("--test-random", action="store_true",
                        help="Rastgele girdi ile model testi")

    args = parser.parse_args()

    if args.test_random:
        cmd_test_random(args)
    elif hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
