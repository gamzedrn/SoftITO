# 🚨 Kredi Kartı Sahtekârlık Tespiti - Isolation Forest Analizi

## 📋 Proje Özeti

Bu notebook, **Credit Card Fraud Detection** veri setini kullanarak **Isolation Forest** yöntemiyle kredi kartı sahtekârlık tespiti yapmaktadır.

## 🎯 Amaç

Kredi kartı işlemlerindeki sahtekârlık (anomali) vakalarını tespit etmek için unsupervised anomaly detection yöntemlerinden biri olan Isolation Forest modelini eğitmek ve değerlendirmek.

## 📊 Veri Seti

- **Kaynak:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Boyut:** 284,807 işlem kaydı
- **Özellik Sayısı:** 31 (V1-V28: PCA ile indirgenmiş, Time, Amount, Class)
- **Sınıf Dağılımı:**
  - Normal (0): 284,315 (%99.828)
  - Sahtekârlık (1): 492 (%0.172)

## 🔧 Kullanılan Yöntem: Isolation Forest

**Isolation Forest:**
- Anomalilerin "colay izole edilebilir" olduğunu varsayar
- Rastgele ağaçlar oluşturarak her veri noktasını izole etmeye çalışır
- Anomaliler daha kısa yolda izole edilir (daha düşük path length)
- Yüksek boyutlu verilerde etkilidir ve O(n log n) karmaşıklığı vardır

## 📁 Dosya Yapısı

```
Anomally/
├── isolation_forest_creditcard.ipynb  # Ana notebook
├── isolation_forest.ipynb             # Orijinal notebook (sentez veri ile)
├── README_isolation_forest.md         # Bu dosya
└── creditcardfraud/                   # Veri seti (otomatik indirilir)
    └── creditcard.csv
```

## 🚀 Nasıl Çalıştırılır?

### Google Colab ile:

1. `isolation_forest_creditcard.ipynb` dosyasını Google Colab'da açın
2. Hücreleri sırasıyla çalıştırın
3. İlk hücrede gerekli kütüphaneler otomatik olarak yüklenecektir
4. Veri seti otomatik olarak indirilecektir (Kaggle API token gerekebilir)

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

2. **Veri Ön İşleme**
   - Eksik değer kontrolü
   - Özellik ölçekleme
   - Sınıf ayrımı

3. **Isolation Forest Modeli Eğitimi**
   - Model parametreleri (n_estimators, contamination, max_samples)
   - Eğitim süreci

4. **Değerlendirme**
   - Karışıklık matrisi
   - Sınıflandırma raporu
   - ROC-AUC skoru
   - Precision-Recall eğrisi

5. **Hiperparametre Analizi**
   - Ağaç sayısı etkisi
   - Contamination değeri etkisi

6. **Görselleştirme**
   - Anomali skoru dağılımı
   - PCA ile 2D görselleştirme

## 🔍 Sonuçlar

- Isolation Forest, yüksek boyutlu verilerde etkili bir yöntemdir
- Doğrusal olmayan anomalleri başarılı şekilde tespit eder
- Ağaç sayısı artırıldığında performans stabilize olur
- Contamination parametresi sonuçları doğrudan etkiler

## ⚠️ Dikkat Edilmesi Gerekenler

1. **Kaggle API Token:** Veri setini otomatik indirmek için Kaggle API token gereklidir
2. **Bellek:** 284K satırlık veri seti için yeterli bellek gerekir
3. **Çalışma Süresi:** Isolation Forest görece hızlıdır, 1-2 dakikada eğitilir

## 📚 Kaynaklar

- [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation - Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)

## 👨‍🎓 Eğitim Bilgisi

- **Ders:** Softito Yapay Zeka Eğitimi
- **Konu:** Anomaly Detection - Isolation Forest
- **Veri Seti:** Credit Card Fraud Detection

---

*Bu notebook Google Colab'da çalışacak şekilde tasarlanmıştır.*
