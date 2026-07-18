# 01 - Doğal Dil İşleme (NLP)

Bu klasör, NLP konularını kapsayan 7 uyarlanmış notebook içerir.

## Notebook'lar

| Dosya | Konu | Kaggle Veri Seti | Kaggle İndirme |
|-------|------|-------------------|----------------|
| `SMSSpamCollection_TFIDF.ipynb` | TF-IDF ile spam tespiti | SMS Spam Collection | `kaggle datasets download -d uciml/sms-spam-collection-dataset` |
| `IMDBMovieReviews_WordEmbeddings.ipynb` | Word2Vec/FastText ile duygu analizi | IMDB 50K Movie Reviews | `kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews` |
| `TurkishTweets_KelimeVektorleri.ipynb` | Türkçe kelime vektörleri (Word2Vec, GloVe, FastText) | Turkish Tweets with Sentiment | `kaggle datasets download -d ozanerturk/tr-40k-tweets-with-sentiment-labels` |
| `NetflixStock_RNN.ipynb` | RNN ile hisse senedi fiyat tahmini | Netflix Stock Price | `kaggle datasets download -d sharanbsarker/netflix-stock-price-prediction` |
| `JenaClimate_LSTM.ipynb` | LSTM ile hava durumu tahmini | Jena Climate | `kaggle datasets download -d mnassib/jena-climate` |
| `Sentiment140_Transformer.ipynb` | Attention/Transformer ile duygu analizi | Sentiment140 Twitter | `kaggle datasets download -d kazanova/sentiment140` |
| `RoadSignDetection_YOLO.ipynb` | YOLO ile nesne tespiti | Road Sign Detection | `kaggle datasets download -d andrewmvd/road-sign-detection` |

## Colab'da Çalıştırma

1. Her notebook'un başında `!pip install` komutları bulunmaktadır
2. Kaggle API ile veri otomatik indirilir (fallback olarak sklearn/torchvision dataset kullanılır)
3. GPU kullanımı önerilir: `Runtime > Change runtime type > GPU`
