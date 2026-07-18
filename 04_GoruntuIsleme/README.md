# 04 - Goruntu Isleme

Bu klasor, goruntu isleme ve derin ogrenme konularini 3 notebook uzerinden sirayla ele almaktadir.

## Notebook'lar

| # | Dosya | Konu | Veri Seti |
|---|-------|------|-----------|
| 1 | `01_goruntu_isleme_giris_mnist.ipynb` | Goruntu isleme temelleri (piksel, filtre, kenar tespiti, morfolojik islemler, esikleme, kontur, ozellik cikarimi) | MNIST |
| 2 | `02_goruntu_isleme_ileri_cifar10.ipynb` | Ileri goruntu isleme (veri artirma, PCA, t-SNE, SVM, MLP, CNN) | CIFAR-10 |
| 3 | `03_derin_ogrenme_cifar10.ipynb` | Derin ogrenme mimarileri (CNN siniflandirma, U-Net segmentasyon, Mask R-CNN nesne tespiti) | CIFAR-10 |

## One Cikis Sirasi

1. **Once 01**'i calistirin — piksel duzeyinde goruntu isleme kavramlari
2. **Sonra 02**'yi calistirin — makine ogrenmesi ve temel sinir aglari
3. **En son 03**'u calistirin — ileri derin ogrenme mimarileri

## Gereksinimler

- Python 3.10+
- Kaggle API anahtari (veri setlerini indirmek icin)
- Gerekli kutuphaneler: `numpy`, `pandas`, `matplotlib`, `opencv-python`, `scikit-learn`, `torch`, `torchvision`, `pillow`

Kutuphaneler notebook icinde `!pip install` komutlariyla otomatik yuklenir.

## Kullanim

```bash
# Kaggle API'yi ayarlayin (eger yoksa)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Colab'da dosyalari yukleyin veya ortaklasa calistirin
```
