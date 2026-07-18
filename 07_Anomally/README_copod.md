# 🚨 Kredi Kartı Sahtekârlık Tespiti - COPOD Analizi

## 📋 Proje Özeti

Bu notebook, **Credit Card Fraud Detection** veri setini kullanarak **COPOD (Copula-Based Outlier Detection)** yöntemiyle kredi kartı sahtekârlık tespiti yapmaktadır.

## 🎯 Amaç

Kredi kartı işlemlerindeki sahtekârlık (anomali) vakalarını tespit etmek için unsupervised anomaly detection yöntemlerinden biri olan COPOD modelini eğitmek ve değerlendirmek.

## 📊 Veri Seti

- **Kaynak:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Boyut:** 284,807 işlem kaydı
- **Özellik Sayısı:** 31 (V1-V28: PCA ile indirgenmiş, Time, Amount, Class)
- **Sınıf Dağılımı:**
  - Normal (0): 284,315 (%99.828)
  - Sahtekârlık (1): 492 (%0.172)

## 🔧 Kullanılan Yöntem: COPOD

**COPOD** (Copula-Based Outlier Detection):
- Parametrik olmayan bir anomaly detection yöntemidir
- Her veri noktası için copula tabanlı bir outlier skoru hesaplar
- Verilerin dağılımını modellemeden çalışır
- Genellikle high-dimensional verilerde başarılıdır

## 📁 Dosya Yapısı

```
Anomally/
├── copod_creditcard.ipynb          # Ana notebook
├── copod.ipynb                     # Orijinal notebook (sentez veri ile)
├── README_copod.md                 # Bu dosya
└── creditcardfraud/                # Veri seti (otomatik indirilir)
    └── creditcard.csv
```

## 🚀 Nasıl Çalıştırılır?

### Google Colab ile:

1. `copod_creditcard.ipynb` dosyasını Google Colab'da açın
2. Hücreleri sırasıyla çalıştırın
3. İlk hücrede gerekli kütüphaneler otomatik olarak yüklenecektir
4. Veri seti otomatik olarak indirilecektir (Kaggle API token gerekebilir)

### Yerel Ortam ile:

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn pyod opendatasets
   ```

2. Kaggle API token'unuzu ayarlayın:
   ```bash
   # ~/.kaggle/kaggle.json dosyasını oluşturun
   ```

3. Notebook'u çalıştırın

## 📈 İçerik

1. **Veri Yükleme ve Keşifçi Analiz (EDA)**
   - Sınıf dağılımı görselleştirmesi
   - Tutar analizi
   - Korelasyon analizi

2. **Veri Ön İşleme**
   - Eksik değer kontrolü
   - Özellik ölçekleme
   - Sınıf ayrımı

3. **COPOD Modeli Eğitimi**
   - Model parametreleri
   - Eğitim süreci

4. **Değerlendirme**
   - Karışıklık matrisi
   - Sınıflandırma raporu
   - ROC-AUC skoru
   - Precision-Recall eğrisi

5. **Görselleştirme**
   - Anomali skoru dağılımı
   - PCA ile 2D görselleştirme
   - Eşik değeri analizi

## 🔍 Sonuçlar

- COPOD modeli, PCA ile indirgenmiş özelliklerde başarılı sonuçlar verir
- Outlier skorları, gerçek sahtekârlık vakalarını iyi ayırt eder
- Ciddi sınıf dengesizliği (%0.172) nedeniyle precision ve recall dengesi önemlidir

## ⚠️ Dikkat Edilmesi Gerekenler

1. **Kaggle API Token:** Veri setini otomatik indirmek için Kaggle API token gereklidir
2. **Bellek:** 284K satırlık veri seti için yeterli bellek gerekir
3. **Çalışma Süresi:** COPOD modeli görece hızlıdır, 1-2 dakikada eğitilir

## 📚 Kaynaklar

- [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [PyOD Documentation](https://pyod.readthedocs.io/)
- [COPOD Paper](https://arxiv.org/abs/2001.00263)

## 👨‍🎓 Eğitim Bilgisi

- **Ders:** Softito Yapay Zeka Eğitimi
- **Konu:** Anomaly Detection - COPOD
- **Veri Seti:** Credit Card Fraud Detection

---

*Bu notebook Google Colab'da çalışacak şekilde tasarlanmıştır.*
