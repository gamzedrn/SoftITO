# 03 - Büyük Dil Modelleri (LLM)

Bu klasör, LLM konularını kapsayan 5 uyarlanmış notebook içerir.

## Notebook'lar

| Dosya | Konu | Kaggle Veri Seti | Kaggle İndirme |
|-------|------|-------------------|----------------|
| `TurkishTweets_LLMVeriOnIsleme.ipynb` | LLM veri ön işleme (BPE, padding, instruction formatı) | Turkish Tweets with Sentiment | `kaggle datasets download -d ozanerturk/tr-40k-tweets-with-sentiment-labels` |
| `TurkishInstructions_PEFTLoRA.ipynb` | PEFT/LoRA ile ince ayar | Turkish Instructions (HuggingFace) | `datasets` kütüphanesi ile yüklenir |
| `Shakespeare_SicaklikOrnekleme.ipynb` | Sıcaklık ve örnekleme stratejileri (greedy, top-k, top-p) | Shakespeare Plays | `kaggle datasets download -d kinguistics/shakespeare-plays` |
| `Wikipedia_TurkishSLM.ipynb` | Türkçe küçük dil modeli (Bigram + LSTM) | Wikipedia API | Otomatik (wikipedia-api) |
| `Transformer_SifirdanMimari.ipynb` | Transformer mimarisi sıfırdan uygulama | Senteetik (dizi kopyalama) | Gerekmez |

## Öğrenme Hedefleri

- **Veri Ön İşleme:** Metin temizleme, BPE tokenizasyonu, padding/truncation, attention mask, instruction formatları (Alpaca, Chat, Llama)
- **PEFT/LoRA:** Düşük rank adaptasyonu, QLoRA (NF4 kuantizasyonu), parametre verimli ince ayar
- **Sıcaklık & Örnekleme:** Softmax sıcaklık ölçekleme, greedy decoding, top-k, top-p (nucleus) sampling
- **Dil Modelleme:** Karakter düzeyinde bigram ve LSTM dil modelleri, perplexity метриği
- **Transformer:** Sinüsözel konumsal kodlama, multi-head attention, encoder-decoder, teacher forcing

## Colab'da Çalıştırma

1. GPU önerilir (özellikle PEFT/LoRA için): `Runtime > Change runtime type > GPU`
2. Wikipedia tabanlı notebook internet bağlantısı gerektirir
3. `!pip install transformers datasets peft accelerate` gibi komutlar dahildir
