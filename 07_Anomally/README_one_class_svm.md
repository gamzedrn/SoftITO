# 🚨 Kredi Kartı Sahtekârlık Tespiti - One-Class SVM Analizi

## 📋 Proje Özeti

Bu notebook, **Credit Card Fraud Detection** veri setini kullanarak **One-Class SVM (Support Vector Machine)** yöntemiyle kredi kartı sahtekârlık tespiti yapmaktadır.

## 🎯 Amaç

Kredi kartı işlemlerindeki sahtekârlık (anomali) vakalarını tespit etmek için unsupervised anomaly detection yöntemlerinden biri olan One-Class SVM modelini eğitmek ve değerlendirmek.

## 📊 Veri Seti

- **Kaynak:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Boyut:** 284,807 işlem kaydı
- **Özellik Sayısı:** 31 (V1-V28: PCA ile indirgenmiş, Time, Amount, Class)
- **Sınıf Dağılımı:**
  - Normal (0): 284,315 (%99.828)
  - Sahtekârlık (1): 492 (%0.172)

## 🔧 Kullanılan Yöntem: One-Class SVM

**One-Class SVM:**
- Verileri yüksek boyutlu bir uzaya dönüştürür
- Normal verileri çevreleyen bir "sınırlayıcı" (boundary) oluşturur
- Bu sınırlayıcının dışında kalan noktalar anomali olarak kabul edilir
- Çekirdek hilesi (kernel trick) ile doğrusal olmayan sınırlar çizebilir

## ⚠️ Önemli Uyarı

**One-Class SVM, büyük veri setlerinde çok yavaştır!**

- O(n²) karmaşıklığına sahiptir
- 284K veri noktası ile doğrudan çalışmak çok uzun sürer
- Bu notebook'da performans için **alt örneklem (subsampling)** kullanılmaktadır
- Eğitim için ~20,000 normal işlem seçilmektedir

## 📁 Dosya Yapısı

```
Anomally/
├── one_class_svm_creditcard.ipynb  # Ana notebook
├── one_class_svm.ipynb             # Orijinal notebook (sentez veri ile)
├── README_one_class_svm.md         # Bu dosya
└── creditcardfraud/                # Veri seti (otomatik indirilir)
    └── creditcard.csv
```

## 🚀 Nasıl Çalıştırılır?

### Google Colab ile:

1. `one_class_svm_creditcard.ipynb` dosyasını Google Colab'da açın
2. Hücreleri sırasıyla çalıştırın
3. İlk hücrede gerekli kütüphaneler otomatik olarak yüklenecektir
4. Veri seti otomatik olarak indirilecektir (Kaggle API token gerekebilir)

**Not:** One-Class SVM eğitimi diğer yöntemlere göre daha uzun sürebilir (5-15 dakika).

### Yerel Ortam ile:

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn opendatasets
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

2. **Alt Örneklem Seçimi (Subsampling)**
   - Neden alt örneklem gerekli?
   - Eğitim verisi hazırlığı

3. **One-Class SVM Modeli Eğitimi**
   - Model parametreleri (kernel, nu, gamma)
   - Eğitim süreci

4. **Değerlendirme**
   - Karışıklık matrisi
   - Sınıflandırma raporu
   - ROC-AUC skoru
   - Precision-Recall eğrisi

5. **Hiperparametre Analizi**
   - Kernel fonksiyonu etkisi
   - nu değeri etkisi
   - Eğitim veri boyutu etkisi

6. **Görselleştirme**
   - Anomali skoru dağılımı
   - PCA ile 2D görselleştirme

## 🔍 Sonuçlar

- One-Class SVM, küçük veri setlerinde çok başarılıdır
- Alt örneklem ile eğitilmesi gerekmektedir
- RBF kernel genellikle daha iyi sonuç verir
- nu parametresi anomali oranını doğrudan etkiler

## ⚠️ Dikkat Edilmesi Gerekenler

1. **Kaggle API Token:** Veri setini otomatik indirmek için Kaggle API token gereklidir
2. **Bellek:** 284K satırlık veri seti için yeterli bellek gerekir
3. **Çalışma Süresi:** One-Class SVM diğer yöntemlere göre daha yavaştır (5-15 dakika)
4. **Alt Örneklem:** Eğitim için ~20,000 örnek kullanılmaktadır

## 📚 Kaynaklar

- [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation - One-Class SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
- [One-Class SVM Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.5548&rep=rep1&type=pdf)

## 👨‍🎓 Eğitim Bilgisi

- **Ders:** Softito Yapay Zeka Eğitimi
- **Konu:** Anomaly Detection - One-Class SVM
- **Veri Seti:** Credit Card Fraud Detection

---

*Bu notebook Google Colab'da çalışacak şekilde tasarlanmıştır.*
