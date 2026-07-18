# SoftITO - Yapay Zeka Egitimi

Bu depo, **Yapay Zeka** konularini 7 ana kategoride 20+ notebook ile kapsamaktadir.
Tum notebook'lar Colab ve Kaggle uyumludur; Kaggle API calismazsa otomatik sentetik veri uretilir.

---

## Klasor Yapisi

```
SoftITO/
├── 01_NLP/                          # Dogal Dil Isleme (7 notebook)
├── 02_GeneticOptimization/          # Genetik Algoritmalar (3 notebook)
├── 03_LLM/                          # Buyuk Dil Modelleri (5 notebook + 3 .py)
├── 04_GoruntuIsleme/                # Goruntu Isleme (3 notebook)
├── 05_UreticiYapayZeka/             # Uretici Yapay Zeka (1 notebook)
├── 06_VeriAnalizi/                  # Veri Analizi ve Drift Tespiti (1 notebook)
└── 07_Anomally/                     # Anomali Tespiti (3 notebook)
```

---

## 01 - Dogal Dil Isleme (NLP)

| Dosya | Konu | Veri Seti |
|-------|------|-----------|
| `SMSSpamCollection_TFIDF.ipynb` | TF-IDF ile spam tespiti | SMS Spam Collection |
| `IMDBMovieReviews_WordEmbeddings.ipynb` | Word2Vec/FastText ile duygu analizi | IMDB 50K Movie Reviews |
| `TurkishTweets_KelimeVektorleri.ipynb` | Turkce kelime vektorleri | Turkish Tweets with Sentiment |
| `NetflixStock_RNN.ipynb` | RNN ile hisse senedi fiyat tahmini | Yahoo Finance (yfinance) |
| `JenaClimate_LSTM.ipynb` | LSTM ile hava durumu tahmini | Jena Climate |
| `Sentiment140_Transformer.ipynb` | Transformer ile duygu analizi | Sentiment140 Twitter |
| `RoadSignDetection_YOLO.ipynb` | YOLO ile nesne tespiti | Road Sign Detection |

## 02 - Genetik Optimizasyon

| Dosya | Konu | Veri Seti |
|-------|------|-----------|
| `BreastCancer_GeneticAlgorithm.ipynb` | GA ile ozellik secimi | Breast Cancer Wisconsin |
| `WineQuality_CMAES.ipynb` | CMA-ES ile SVM hiperparametre | Wine Quality (UCI) |
| `Diabetes_PSO.ipynb` | PSO ile ozellik secimi | Pima Indians Diabetes |

## 03 - Buyuk Dil Modelleri (LLM)

| Dosya | Konu | Veri Seti |
|-------|------|-----------|
| `TurkishTweets_LLMVeriOnIsleme.ipynb` | BPE, padding, instruction formati | Turkish Tweets with Sentiment |
| `TurkishInstructions_PEFTLoRA.ipynb` | PEFT/LoRA ile ince ayar | merve/turkish_instructions (HuggingFace) |
| `Shakespeare_SicaklikOrnekleme.ipynb` | Sicaklik ve ornekleme stratejileri | Shakespeare Plays |
| `Wikipedia_TurkishSLM.ipynb` | Turkce SLM egitimi | Turkce Vikipedi |
| `Transformer_SifirdanMimari.ipynb` | Transformer mimarisi sifirdan | Sentetik veri |

Ek olarak `.py` dosyalari mevcuttur:
- `TurkishInstructions_PEFTLoRA.py`
- `TurkishTweets_LLMVeriOnIsleme.py`
- `Wikipedia_TurkishSLM.py`

## 04 - Goruntu Isleme

| Dosya | Konu | Veri Seti |
|-------|------|-----------|
| `01_goruntu_isleme_giris_mnist.ipynb` | Temel goruntu isleme (piksel, filtre, kenar tespiti) | MNIST |
| `02_goruntu_isleme_ileri_cifar10.ipynb` | Ileri (PCA, t-SNE, SVM, MLP, CNN) | CIFAR-10 |
| `03_derin_ogrenme_cifar10.ipynb` | CNN, U-Net, Mask R-CNN | CIFAR-10 |

## 05 - Uretici Yapay Zeka

| Dosya | Konu | Veri Seti |
|-------|------|-----------|
| `GenerativeAI_Temeller.ipynb` | GAN, VAE, Diffusion modelleri | MNIST / Sentetik |

## 06 - Veri Analizi

| Dosya | Konu | Veri Seti |
|-------|------|-----------|
| `CreditScore_DriftTespiti.ipynb` | Veri drift tespiti (KS, Chi-kare, PSI, JS, Evidently) | Credit Score Classification |

## 07 - Anomali Tespiti

| Dosya | Konu | Veri Seti |
|-------|------|-----------|
| `one_class_svm_creditcard.ipynb` | One-Class SVM | Credit Card Fraud |
| `isolation_forest_creditcard.ipynb` | Isolation Forest | Credit Card Fraud |
| `copod_creditcard.ipynb` | COPOD (Combination of OD) | Credit Card Fraud |

---

## Nasil Kullanilir?

### Google Colab ile
1. Ilgili `.ipynb` dosyasini Colab'a yukleyin
2. Onceki hucreleri sirayla calistirin
3. Kaggle API anahtari gerekmez; Kaggle calismazsa otomatik sentetik veri uretilir

### Kaggle API Ayarlama (istege bagli)
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Gereksinimler
Tum notebook'larda gerekli kutuphaneler `!pip install` komutlariyla otomatik yuklenir.
Temel kutuphaneler: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `tensorflow`, `transformers`

---

## Lisans

Bu proje ogretim amaclidir.
