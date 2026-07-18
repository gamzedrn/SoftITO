# 05 - Üretici Yapay Zeka

Bu klasör, üretici yapay zeka temellerini kapsayan 1 uyarlanmış notebook içerir.

## Notebook'lar

| Dosya | Konu | Veri Seti | Kaynak |
|-------|------|-----------|--------|
| `GenerativeAI_Temeller.ipynb` | Üretici yapay zeka temelleri (Markov, GAN, RNN) | Shakespeare + sentetik | Metin dosyası + normal dağılım |

## Öğrenme Hedefleri

- **Markov Zinciri:** Kelime düzeyinde geçiş olasılıkları ile metin üretimi
- **GAN (Generative Adversarial Network):** NumPy ile sıfırdan Generator + Discriminator, manual backpropagation, BCE loss
- **Karakter Düzeyi RNN:** BPTT (Backpropagation Through Time), tanh gizli durum, one-hot encoding, softmax çıktı
- **Karşılaştırma:** Üretici vs Ayırt edici modeller, matematiksel çerçeve

## Colab'da Çalıştırma

1. GPU gerekmez, sadece NumPy ve Matplotlib yeterlidir
2. `!pip install numpy matplotlib` yeterlidir
3. Tüm modeller NumPy ile sıfırdan uygulanmıştır (PyTorch/TensorFlow gerektirmez)
