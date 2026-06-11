# DeepfakeULTRA V5 — Cross-Dataset Benchmark Raporu

**Model:** DeepfakeULTRA V5 (Dual-MobileNetV3 + Transformer)  
**Egitim Verisi:** DF40 Dataset  
**Tarih:** 2026-05-13

---

## 1. Genel Bakis

DeepfakeULTRA V5 modeli **6 farkli dataset** uzerinde degerlendirildi:

1. **Ic Test (DF40)** — Egitim verisiyle ayni dagitim
2. **Celeb-DF v2** — Autoencoder face-swap (CVPR 2020)
3. **FaceForensics++ (c23)** — Deepfakes + Face2Face (ICCV 2019)
4. **DFDC** — Meta/Facebook Deepfake Detection Challenge
5. **Deepfake-vs-Real-20K** — Cesitli modern deepfake yontemleri
6. **DeepFakeFace (DFF)** — Stable Diffusion (text2img + inpainting)

---

## 2. Karsilastirmali Performans Tablosu

| Metrik | Ic Test | Celeb-DF v2 | FF++ | DFDC | Deepfake20K | DFF (SD) |
|--------|:-------:|:-----------:|:----:|:----:|:-----------:|:--------:|
| **Gorsel** | 59,655 | 1,890 | 750 | 2,000 | 2,000 | 2,000 |
| **REAL/FAKE** | 29.7K/29.9K | 1K/890 | 250/500 | 1K/1K | 1K/1K | 1K/1K |
| | | | | | | |
| **ROC-AUC** | **0.9606** | 0.5439 | 0.5121 | 0.5361 | **0.6231** | 0.4745 |
| **EER** | 0.1036 | 0.4651 | 0.4920 | 0.4745 | 0.3990 | 0.5205 |
| **Accuracy** | 0.8990 | 0.5566 | 0.4640 | 0.5315 | **0.6290** | 0.5065 |
| **Macro F1** | 0.8989 | 0.5162 | 0.4638 | 0.5294 | **0.6164** | 0.3662 |
| **Youden J** | 0.7980 | 0.0833 | 0.0580 | 0.0630 | **0.2580** | 0.0130 |
| | | | | | | |
| **Threshold** | 0.4423 | 0.3209 | 0.4076 | 0.2930 | 0.2904 | 0.2564 |
| **ECE** | 0.2705 | 0.1694 | 0.2911 | 0.1811 | 0.1900 | 0.1909 |
| **FPR@95** | 0.2463 | 0.9340 | 0.9360 | 0.9340 | 0.8410 | 0.9560 |
| | | | | | | |
| **REAL prob** | 0.3233 | 0.3107 | 0.3732 | 0.3146 | 0.3126 | 0.3190 |
| **FAKE prob** | 0.5509 | 0.3148 | 0.3767 | 0.3232 | 0.3261 | 0.3142 |
| **Fark** | **0.2276** | 0.0041 | 0.0035 | 0.0086 | **0.0135** | -0.0048 |

---

## 3. Performans Siralamasii (AUC)

```
1. Ic Test (DF40)       ||||||||||||||||||||||||||||||||||||||||||||||||  0.9606
2. Deepfake-vs-Real-20K |||||||||||||||||||||||||||||||                  0.6231
3. Celeb-DF v2          |||||||||||||||||||||||||||                      0.5439
4. DFDC                 ||||||||||||||||||||||||||                       0.5361
5. FaceForensics++      |||||||||||||||||||||||||                        0.5121
6. DeepFakeFace (SD)    |||||||||||||||||||||||                          0.4745
```

---

## 4. Dataset Detaylari

| Dataset | Kaynak | Yontemler | Yil | Indirme |
|---------|--------|-----------|-----|---------|
| **DF40** | Egitim verisi | 40 yontem (FS/FR/EFS/FE) | 2024 | GitHub |
| **Celeb-DF v2** | Resmi ZIP | Autoencoder face-swap | 2020 | Proje sayfasi |
| **FF++ (c23)** | Resmi sunucu | Deepfakes + Face2Face | 2019 | FF++ script |
| **DFDC** | Kaggle | Cesitli challenge yontemleri | 2020 | Kaggle |
| **Deepfake20K** | Kaggle | Cesitli modern yontemler | 2024 | Kaggle |
| **DeepFakeFace** | HuggingFace | Stable Diffusion (t2i+inpaint) | 2024 | HuggingFace |

---

## 5. Temel Bulgular

| Bulgu | Detay |
|-------|-------|
| **In-distribution basari** | DF40 test setinde AUC=0.96, guclu ayirt edicilik |
| **En iyi harici** | Deepfake20K (AUC=0.62) — modern yontemlerle kismi ortusme |
| **En kotu harici** | DeepFakeFace (AUC=0.47) — SD diffusion ciktilari terse siniflandiriliyor |
| **Genel trend** | Tum harici datasetlerde REAL/FAKE olasilik dagilimlari cakismis |
| **Domain gap** | Model sadece egitim sirasinda gordugu artifact turlerini tanimliyor |

---

## 6. Akademik Yorum

Cross-dataset benchmark, deepfake tespit literaturundeki **"cross-method generalization"** problemini dogruluyor:

- Model **DF40 domain'inde guclu** ama farkli yontemlere genelleyemiyor
- **En ilginc bulgu:** DeepFakeFace'te AUC=0.47 — model SD ciktilarini gercek olarak algilıyor (ters korelasyon)
- **Deepfake20K basarisi** (AUC=0.62) kismi yontem ortusmesindan kaynaklaniyor

Bu, **domain adaptation**, **multi-method training** veya **ensemble** yaklasimlarinin gerekliligi icin guclu bir kanit.

---

## 7. Dosya Yapisi

```
evaluation/external/
|-- celeb_df_v2/     (metrics.json, roc_curve.png, confusion_matrix.png, benchmark_report.md)
|-- faceforensics/   (metrics.json, roc_curve.png, confusion_matrix.png, benchmark_report.md)
|-- dfdc/            (metrics.json, roc_curve.png, confusion_matrix.png, benchmark_report.md)
|-- deepfake20k/     (metrics.json, roc_curve.png, confusion_matrix.png, benchmark_report.md)
|-- deepfakeface/    (metrics.json, roc_curve.png, confusion_matrix.png, benchmark_report.md)
```

---

### Referanslar
- Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics", CVPR 2020
- Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019
- Dolhansky et al., "The DeepFake Detection Challenge Dataset", arXiv 2020
- Yan et al., "DF40: Toward Next-Generation Deepfake Detection", NeurIPS 2024
- OpenRL, "DeepFakeFace: Robustness and Generalizability with Diffusion Models", 2024

*Bu rapor DeepfakeULTRA V5 evaluation pipeline tarafindan otomatik olusturulmustur.*
