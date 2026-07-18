# 06 - Veri Analizi

Bu klasör, veri analizi konularını kapsayan 1 uyarlanmış notebook içerir.

## Notebook'lar

| Dosya | Konu | Kaggle Veri Seti | Kaggle İndirme |
|-------|------|-------------------|----------------|
| `CreditScore_DriftTespiti.ipynb` | Veri drift tespiti | Credit Score Classification | `kaggle datasets download -d parisrohan/credit-score-classification` |

## Öğrenme Hedefleri

- **KS Testi (Kolmogorov-Smirnov):** Sayısal özellikler için drift tespiti
- **Chi-Kare Testi:** Kategorik özellikler için drift tespiti
- **PSI (Population Stability Index):** Dağılım değişim ölçümü
- **JS Divergence (Jensen-Shannon):** İstatistiksel benzerlik ölçümü
- **Evidently AI:** Otomatik drift raporu oluşturma

## Colab'da Çalıştırma

1. `!pip install evidently` gereklidir
2. CPU yeterlidir
3. Kaggle API ile veri otomatik indirilir
