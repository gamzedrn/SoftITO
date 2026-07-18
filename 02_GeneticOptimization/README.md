# 02 - Genetik & Optimizasyon Algoritmaları

Bu klasör, evrimsel ve sürü zekası optimizasyon algoritmalarını kapsayan 3 uyarlanmış notebook içerir.

## Notebook'lar

| Dosya | Konu | Kaggle Veri Seti | Kaggle İndirme |
|-------|------|-------------------|----------------|
| `BreastCancer_GeneticAlgorithm.ipynb` | Genetik Algoritma ile özellik seçimi | Breast Cancer Wisconsin | `kaggle datasets download -d uciml/breast-cancer-wisconsin-data` |
| `WineQuality_CMAES.ipynb` | CMA-ES ile SVM hiperparametre optimizasyonu | Red Wine Quality | `kaggle datasets download -d uciml/red-wine-quality-cortez-et-al-2009` |
| `Diabetes_PSO.ipynb` | PSO ile özellik seçimi | Pima Indians Diabetes | `kaggle datasets download -d uciml/pima-indians-diabetes-database` |

## Öğrenme Hedefleri

- **Genetik Algoritma (GA):** Chromosome=özellik alt kümesi, fitness=doğruluk. Turnuva seçimi, tek noktalı çaprazlama, bit-flip mutasyonu, elitizm.
- **CMA-ES:** Kovaryans matrisi adaptasyonu ile türevsiz optimizasyon. Cholesky örnekleme, CSA, rank-1/rank-μ güncelleme.
- **PSO:** Enertia ağırlığı, pbest/gbest, hız sınırlama. Özellik seçimi için eşik tabanlı ikili kodlama.

## Colab'da Çalıştırma

1. `!pip install` hücrelerini çalıştırın
2. Kaggle API otomatik veri indirir
3. CPU yeterlidir, GPU gerekmez
