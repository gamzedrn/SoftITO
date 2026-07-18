# # LLM Veri Ön İşleme (Türkçe Tweet Veri Seti)
# Bu notebook, Large Language Model (LLM) eğitimi için Türkçe metin verilerinin ön işlenmesini kapsamlı bir şekilde ele almaktadır.
# **Veri Seti:** `ozanerturk/tr-40k-tweets-with-sentiment-labels` - 40.000+ Türkçe tweet
# **Kapsanan Konular:**
# 1. Veri setini indirme ve yükleme
# 2. Metin temizleme (HTML, URL, emoji kaldırma)
# 3. Tokenizasyon (BPE从 scratch + HuggingFace)
# 4. Padding ve Truncation
# 5. Instruction formatlama (Alpaca/Chat şablonları)
# 6. HuggingFace Dataset oluşturma
# 7. Görselleştirme

# Gerekli kütüphaneleri kur
# !pip install transformers datasets tokenizers wordcloud matplotlib pandas numpy -q

import re
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
print('Kütüphaneler başarıyla yüklendi!')

# ## 1. Veri Setini İndirme
# Türkçe tweet veri setini Kaggle üzerinden indirip yüklüyoruz. Bu veri seti duygu etiketli tweetler içermektedir.

# Kaggle API ile veri setini indir
# Kaggle API anahtarınızın ~/.kaggle/kaggle.json dosyasında olduğundan emin olun
KAGGLE_AVAILABLE = False

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    KAGGLE_AVAILABLE = True
    print('Kaggle API başarıyla kimlik doğrulandı!')
except Exception as e:
    print(f'Kaggle API mevcut değil: {e}')
    print('Kaggle API anahtarı gerekli. ~/.kaggle/kaggle.json dosyasını oluşturun.')
    print('Alternatif olarak, veri setini manuel olarak indirip yükleyebilirsiniz.')

# Veri setini indir ve yükle
if KAGGLE_AVAILABLE:
    try:
        api.dataset_download_files(
            'ozanerturk/tr-40k-tweets-with-sentiment-labels',
            path='./turkish_tweets_data',
            force=True,
            quiet=False
        )
        print('Veri seti indirildi.')
    except Exception as e:
        print(f'İndirme hatası: {e}')
else:
    print('Kaggle API mevcut değil. Alternatif veri oluşturuluyor...')

# Veri setini yükle
df = None
import zipfile
import glob

try:
    # Zip dosyasını aç
    zip_files = glob.glob('./turkish_tweets_data/*.zip')
    if zip_files:
        for zf in zip_files:
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall('./turkish_tweets_data')
    
    # CSV dosyasını bul ve yükle
    csv_files = glob.glob('./turkish_tweets_data/*.csv')
    if csv_files:
        df = pd.read_csv(csv_files[0])
        print(f'Veri seti yüklendi: {len(df)} satır')
except Exception as e:
    print(f'Yükleme hatası: {e}')

# Eğer veri yüklenemezse, örnek Türkçe tweet verisi oluştur
if df is None or len(df) == 0:
    print('Örnek Türkçe tweet veri seti oluşturuluyor...')
    sample_tweets = [
        ('Bugün hava çok güzel, parkta yürüyüş yapıyorum 🌞', 1),
        ('Bu film gerçekten harika, herkese tavsiye ederim!', 1),
        ('Üzgünüm ama bu restoran çok kötüydü, bir daha gelmem.', 0),
        ('<html><body>Yeni telefonumu çok sevdim!</body></html>', 1),
        ('https://t.co/abc123 harika bir gün geçirdik', 1),
        ('Bu kitap beni çok etkiledi, muhteşem bir eser 👏', 1),
        ('İş toplantısı çok sıkıcı geçti 😒', 0),
        ('@kullanici seni çok seviyorum ❤️', 1),
        ('Yarın sınav var, çok stresliyim 😰', 0),
        ('Annemin yaptığı yemek dünyanın en güzel yemeği! 🍕', 1),
        ('Bu haber beni çok üzdü, insanlar neden kötü?', 0),
        ('Spor salonuna gitmek en iyi karardı 💪', 1),
        ('Trafikte 2 saat kaldım, çıldıracağım!', 0),
        ('Yeni albümü dinledim, şarkılar harika! 🎵', 1),
        ('Bu maçı izlemek inanılmazdı! GOL GOL GOL! ⚽', 1),
        ('İşten kovuldum, hayatım bitti 😭', 0),
        ('Kedim bugün çok tatlı, fotoğraf çekiyorum 🐱', 1),
        ('Bu restoranda yemek yemeyin, midem bozuldu!', 0),
        ('Tatil planları başladı! Heyecanlıyım ✈️', 1),
        ('Sınavım çok kötü geçti, ağlıyorum 😢', 0),
        ('Arkadaşlarımla kahve içtik, çok eğlendik ☕', 1),
        ('Bu siyasetçiyi hiç sevmiyorum, çok yalan söylüyor!', 0),
        ('Yeni year partisi muhteşem olacak! 🎉', 1),
        ('Telefonum kırıldı, çok pahalı tamir olacak 😞', 0),
        ('Kütüphaneye gittim, çok güzel kitaplar buldum 📚', 1),
        ('Bakkalda kuyruk çok uzun, bekliyorum 🕐', 0),
        ('Doğa yürüyüşü yaptım, manzara harikaydı 🏔️', 1),
        ('Bu dersi geçemeyeceğim, çok zor 😰', 0),
        ('Düğün çok güzeldi, gelin harika görünüyordu 💒', 1),
        ('Hastanede 5 saat bekledim, isyan ediyorum!', 0),
        ('Çocuklarla parka gittik, eğlendik 🎡', 1),
        ('Bu restoranın pahalı olduğunu biliyordum ama bu kadar değil!', 0),
        ('Yeni saç modelim çok beğenildi 💇', 1),
        ('Uçak rötar yaptı, 3 saat bekledik ✈️', 0),
        ('Konser muhteşemdi, sahne şovu inanılmazdı 🎸', 1),
        ('Bu haberi duyunca çok sevindim! 🎊', 1),
        ('Arabamın lastiği patladı, yolda kaldım 😩', 0),
        ('Doğum günümü unutmadığınız için teşekkürler 🎂', 1),
        ('Bu projede çok çalıştık, sonuç harika oldu! 🏆', 1),
    ]
    
    # Çeşitlilik için verileri çoğalt
    expanded_tweets = sample_tweets.copy()
    extra_contexts = [
        ('Bu sabah erkenden kalktım, günüm çok verimli geçti', 1),
        ('Akşam yemeği için dışarı çıktık, servis çok yavaştı', 0),
        ('Yeni projemiz hakkında çok heyecanlıyım! 🚀', 1),
        ('Bu haber gerçek mi? İnanamıyorum! 😲', 0),
        ('Sokağa çıkma yasağı geldi, evdeyiz 🏠', 0),
        ('Kargo hala gelmedi, 10 gündür bekliyorum! 📦', 0),
        ('Bu maçı kazandık! Şampiyonuz! 🏅', 1),
        ('Yeni restaurant açıldı, gidip denedik 🍽️', 1),
        ('İnternet kopuyor, online toplantı yapamıyorum!', 0),
        ('Yeni yıl hediyelerini aldım, herkesi mutlu edeceğim 🎁', 1),
        ('Bu konser biletleri çok pahalı 😤', 0),
        ('Bahçede çiçekler açtı, bahar geldi 🌷', 1),
        ('Bu haber beni çok mutlu etti, helal olsun! 👍', 1),
        ('Otobüs 40 dakika gecikmeli geldi, sinirliyim 😠', 0),
        ('Yeni kurs kaydoldum, öğrenmeye başladım 📖', 1),
        ('Bu şehirde yaşamak çok güzel 🌆', 1),
        ('Isınma faturaları çok yüksek geldi, ne yapacağım? 🥶', 0),
        ('Kız kardeşim evlendi, çok mutluyum! 💍', 1),
        ('Bu filmi izlemeyin, zaman kaybı 🎬', 0),
        ('Güneş batarken sahilde yürüyüş, huzur dolu 🌅', 1),
    ]
    expanded_tweets.extend(extra_contexts)
    
    # Veri setini 5 katına çıkar
    final_tweets = []
    for _ in range(5):
        final_tweets.extend(expanded_tweets)
    
    df = pd.DataFrame(final_tweets, columns=['text', 'label'])
    print(f'Örnek veri seti oluşturuldu: {len(df)} tweet')

print(f'\nVeri seti boyutu: {df.shape}')
print(f'Sütunlar: {df.columns.tolist()}')
print(f'\nİlk 5 satır:')
df.head()

# Veri seti hakkında bilgi
print('=== Veri Seti Özeti ===')
print(f'Toplam tweet sayısı: {len(df)}')
print(f'\nEtiket dağılımı:')
print(df['label'].value_counts())
print(f'\nOrtalama tweet uzunluğu (karakter): {df["text"].str.len().mean():.1f}')
print(f'Maks tweet uzunluğu: {df["text"].str.len().max()}')
print(f'Min tweet uzunluğu: {df["text"].str.len().min()}')

# ## 2. Metin Temizleme
# LLM eğitimi için metin verilerinin temizlenmesi kritik bir adımdır. Temizleme işlemleri:
# - HTML etiketlerinin kaldırılması
# - URL'lerin kaldırılması
# - Emoji'lerin temizlenmesi (veya korunması)
# - Fazla boşlukların giderilmesi
# - Kullanıcı adlarının (@) temizlenmesi

# Metin temizleme fonksiyonları

def remove_html(text):
    """HTML etiketlerini kaldırır"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_urls(text):
    """URL'leri kaldırır"""
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_mentions(text):
    """Kullanıcı adlarını (@) kaldırır"""
    return re.sub(r'@\w+', '', text)

def remove_hashtags(text):
    """Hashtag'leri (#) kaldırır"""
    return re.sub(r'#\w+', '', text)

def remove_emojis(text):
    """Emoji'leri kaldırır"""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u200d"
        "\u2640-\u2642"
        "\ufe0f"
        "\u2600-\u2B55"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_extra_whitespace(text):
    """Fazla boşlukları temizler"""
    return re.sub(r'\s+', ' ', text).strip()

def remove_punctuation_keep_turkish(text):
    """Noktalama işaretlerini kaldırır, Türkçe karakterleri korur"""
    return re.sub(r'[^\w\sçğıöşüÇĞİÖŞÜ]', '', text)

def clean_text(text):
    """Tüm temizleme adımlarını uygular"""
    if not isinstance(text, str):
        return ''
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_emojis(text)
    text = remove_extra_whitespace(text)
    return text.strip()

# Temizleme fonksiyonlarını uygula
df['cleaned_text'] = df['text'].apply(clean_text)

# Temizleme sonrası örnekler
print('=== Temizleme Sonrası Örnekler ===\n')
for i in range(min(5, len(df))):
    print(f'Orijinal:    {df.iloc[i]["text"]}')
    print(f'Temizlenmiş: {df.iloc[i]["cleaned_text"]}')
    print()

# ## 3. Tokenizasyon
# ### Byte Pair Encoding (BPE)
# BPE, metin tokenizasyonu için kullanılan popüler bir algoritmadr. Çalışma mantığı:
# 1. Başlangıçta her karakter ayrı bir token
# 2. En sık görülen bitişik çiftleri birleştir
# 3. İstenen boyuta ulaşana kadar tekrarla
# ### HuggingFace Tokenizers
# HuggingFace, önceden eğitilmiş tokenizers sunar (BERT, GPT-2, vb.)

# Sıfırdan basit BPE uygulaması

class SimpleBPE:
    """Basit bir Byte Pair Encoding uygulaması"""
    
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = set()
    
    def get_vocab(self, corpus):
        """Korpusu karakter seviyesinde token'lara böler"""
        vocab = Counter()
        for word in corpus:
            # Her kelimeyi karakterlerine ayır ve kelime sonu ekle
            chars = list(word) + ['</w>']
            vocab[tuple(chars)] += 1
        return vocab
    
    def get_pairs(self, vocab):
        """Tüm bitişik çiftleri sayar"""
        pairs = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs
    
    def merge_pair(self, pair, vocab):
        """En sık görülen çifti birleştirir"""
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in vocab:
            w_out = p.sub(''.join(pair), word)
            new_vocab[w_out] = vocab[word]
        return new_vocab
    
    def train(self, corpus, verbose=True):
        """BPE eğitimini çalıştırır"""
        vocab = self.get_vocab(corpus)
        
        num_merges = self.vocab_size - len(set(c for word in vocab for c in word))
        
        for i in range(min(num_merges, 50)):  # Maksimum 50 merge
            pairs = self.get_pairs(vocab)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_pair(best_pair, vocab)
            self.merges.append(best_pair)
            
            if verbose:
                print(f'Merge {i+1}: {best_pair} -> {"".join(best_pair)} (frekans: {pairs[best_pair]})')
        
        self.vocab = set(c for word in vocab for c in word)
        return vocab
    
    def tokenize(self, text):
        """Metni tokenize eder"""
        tokens = list(text) + ['</w>']
        
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    tokens = tokens[:i] + [''.join(pair)] + tokens[i+2:]
                else:
                    i += 1
        
        return tokens

# BPE eğitimi
sample_texts = df['cleaned_text'].str.lower().tolist()[:100]  # İlk 100 tweet

bpe = SimpleBPE(vocab_size=200)
bpe_vocab = bpe.train(sample_texts, verbose=True)

print(f'\n=== BPE Sonuçları ===')
print(f'Öğrenilen merge sayısı: {len(bpe.merges)}')
print(f'Kelime dağarcığı boyutu: {len(bpe.vocab)}')

# Örnek tokenizasyon
test_text = "bugün hava çok güzel"
tokens = bpe.tokenize(test_text)
print(f'\nÖrnek: "{test_text}" -> {tokens}')

# HuggingFace Tokenizer karşılaştırması

# Türkçe için uygun tokenizer'lar: dbmdz/bert-base-turkish-cased, dbmdz/bert-base-turkish-uncased
model_names = ['bert-base-uncased']  # Demo için basit model

results = {}
for model_name in model_names:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenizasyon
        encoded = tokenizer(test_text, return_tensors='pt')
        decoded = tokenizer.decode(encoded['input_ids'][0])
        
        results[model_name] = {
            'tokens': tokenizer.convert_ids_to_tokens(encoded['input_ids'][0]),
            'token_ids': encoded['input_ids'][0].tolist(),
            'decoded': decoded
        }
        
        print(f'\n=== {model_name} ===')
        print(f'Tokenlar: {results[model_name]["tokens"]}')
        print(f'Token IDleri: {results[model_name]["token_ids"]}')
        print(f'Çözümlenmiş: {decoded}')
        print(f'Kelime dağarcığı boyutu: {tokenizer.vocab_size}')
    except Exception as e:
        print(f'{model_name} yüklenemedi: {e}')

# ## 4. Padding ve Truncation
# LLM'ler sabit uzunlukta girdi bekler. Bu nedenle:
# - **Padding:** Kısa dizileri belirli bir uzunluğa tamamlama
# - **Truncation:** Uzun dizileri belirli bir uzunluğa kısaltma
# Önemli kavramlar:
# - `max_length`: Maksimum dizi uzunluğu
# - `padding`: 'max_length', 'longest', 'do_not_pad'
# - `truncation`: True/False

# Padding ve Truncation uygulaması

MAX_LENGTH = 32

def pad_sequences_manual(sequences, max_len, pad_value=0):
    """Manuel padding uygular"""
    padded = []
    attention_masks = []
    
    for seq in sequences:
        if len(seq) >= max_len:
            # Truncation
            padded_seq = seq[:max_len]
            mask = [1] * max_len
        else:
            # Padding
            padded_seq = seq + [pad_value] * (max_len - len(seq))
            mask = [1] * len(seq) + [0] * (max_len - len(seq))
        
        padded.append(padded_seq)
        attention_masks.append(mask)
    
    return padded, attention_masks

# Örnek tokenizasyon
sample_texts = df['cleaned_text'].head(5).tolist()

# Basit karakter tabanlı tokenizasyon
char_to_id = {}
id_to_char = {}
current_id = 1  # 0 padding için ayrıldı

def simple_tokenize(text):
    tokens = []
    for char in text:
        if char not in char_to_id:
            char_to_id[char] = current_id
            id_to_char[current_id] = char
            current_id += 1
        tokens.append(char_to_id[char])
    return tokens

# Metinleri tokenize et
tokenized = [simple_tokenize(text) for text in sample_texts]

# Padding ve truncation uygula
padded, attention_masks = pad_sequences_manual(tokenized, MAX_LENGTH)

# Sonuçları göster
print(f'=== Padding ve Truncation Sonuçları (max_length={MAX_LENGTH}) ===\n')
for i, (text, tokens, padded_seq, mask) in enumerate(zip(sample_texts, tokenized, padded, attention_masks)):
    print(f'Örnek {i+1}:')
    print(f'  Orijinal: "{text[:50]}..."')
    print(f'  Token sayısı: {len(tokens)}')
    print(f'  Padding sonrası: {padded_seq[:10]}... (toplam: {len(padded_seq)})')
    print(f'  Attention mask: {mask[:10]}... (toplam: {len(mask)})')
    print()

# ## 5. Instruction Formatlama
# LLM fine-tuning için verilerin belirli formatlarda sunulması gerekir.
# ### Yaygın Formatlar:
# 1. **Alpaca Formatı:** `<instruction, input, output>` üçlüsü
# 2. **Chat Formatı:** `<system, user, assistant>` mesajları
# 3. **ShareGPT Formatı:** `<conversations>` listesi

# Instruction formatları oluştur

def create_alpaca_format(instruction, input_text, output):
    """Alpaca formatında instruction oluşturur"""
    if input_text:
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"

def create_chat_format(messages):
    """Chat formatında instruction oluşturur"""
    formatted = "<s>[INST] <<SYS>>\nSen yardımcı bir asistansın.\n<</SYS>>\n\n"
    
    for i, msg in enumerate(messages):
        if msg['role'] == 'user':
            formatted += f"{msg['content']} [/INST] "
        elif msg['role'] == 'assistant':
            formatted += f"{msg['content']}</s>\n<s>[INST] "
    
    return formatted.strip()

# Tweetleri instruction formatına dönüştür
instruction_samples = []

for idx, row in df.head(10).iterrows():
    tweet = row['cleaned_text']
    label = row['label']
    
    # Duygu analizi instruction'ı
    instruction = "Bu tweetin duygu durumunu analiz et."
    input_text = f"Tweet: {tweet}"
    output = "Pozitif" if label == 1 else "Negatif"
    
    alpaca = create_alpaca_format(instruction, input_text, output)
    
    # Chat formatı
    chat_messages = [
        {'role': 'user', 'content': f'Bu tweetin duygu durumunu analiz et: {tweet}'},
        {'role': 'assistant', 'content': f'Bu tweetin duygu durumu: {output}'}
    ]
    chat = create_chat_format(chat_messages)
    
    instruction_samples.append({
        'instruction': instruction,
        'input': input_text,
        'output': output,
        'alpaca_format': alpaca,
        'chat_format': chat
    })

# Örnekleri göster
print('=== Alpaca Formatı Örneği ===\n')
print(instruction_samples[0]['alpaca_format'])

print('\n=== Chat Formatı Örneği ===\n')
print(instruction_samples[0]['chat_format'])

# ## 6. HuggingFace Dataset Oluşturma
# HuggingFace `datasets` kütüphanesi, veri setlerini yönetmek için güçlü araçlar sunar.
# Özellikler:
# - Bellek dostu lazy loading
# - Otomatik batch işleme `.map()` ile
# - Train/validation/test split

# HuggingFace Dataset oluştur

# Instruction verilerini hazırla
dataset_dict = {
    'text': df['cleaned_text'].tolist(),
    'label': df['label'].tolist(),
    'instruction': [s['instruction'] for s in instruction_samples] * (len(df) // len(instruction_samples) + 1),
    'input': [s['input'] for s in instruction_samples] * (len(df) // len(instruction_samples) + 1),
    'output': [s['output'] for s in instruction_samples] * (len(df) // len(instruction_samples) + 1)
}

# Uzunlukları eşitle
min_len = min(len(v) for v in dataset_dict.values())
for key in dataset_dict:
    dataset_dict[key] = dataset_dict[key][:min_len]

# Dataset oluştur
full_dataset = Dataset.from_dict(dataset_dict)

# Train/test split
split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
print(f'Eğitim seti boyutu: {len(split_dataset["train"])}')
print(f'Test seti boyutu: {len(split_dataset["test"])}')

# .map() ile tokenizasyon
def tokenize_function(examples):
    """Her örneği tokenizar"""
    tokenized = []
    for text in examples['text']:
        tokens = simple_tokenize(str(text))[:MAX_LENGTH]
        tokenized.append(tokens)
    
    # Padding
    padded, masks = pad_sequences_manual(tokenized, MAX_LENGTH)
    
    return {
        'input_ids': padded,
        'attention_mask': masks
    }

# Tokenizasyon uygula
tokenized_dataset = split_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32,
    remove_columns=['text', 'instruction', 'input', 'output']
)

print(f'\nTokenize edilmiş eğitim seti: {tokenized_dataset["train"]}')
print(f'Örnek: {tokenized_dataset["train"][0]}')

# ## 7. Görselleştirme
# Veri setinin özelliklerini görsel olarak analiz ediyoruz:
# - Token uzunluğu dağılımı
# - Kelime bulutu
# - Etiket dağılımı

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Token uzunluğu dağılımı
token_lengths = [len(text.split()) for text in df['cleaned_text']]
axes[0, 0].hist(token_lengths, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Kelime Sayısı Dağılımı', fontsize=14)
axes[0, 0].set_xlabel('Kelime Sayısı')
axes[0, 0].set_ylabel('Frekans')
axes[0, 0].axvline(np.mean(token_lengths), color='red', linestyle='--', label=f'Ortalama: {np.mean(token_lengths):.1f}')
axes[0, 0].legend()

# 2. Karakter uzunluğu dağılımı
char_lengths = [len(text) for text in df['cleaned_text']]
axes[0, 1].hist(char_lengths, bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Karakter Sayısı Dağılımı', fontsize=14)
axes[0, 1].set_xlabel('Karakter Sayısı')
axes[0, 1].set_ylabel('Frekans')

# 3. Etiket dağılımı
label_counts = df['label'].value_counts()
axes[1, 0].bar(['Negatif (0)', 'Pozitif (1)'], label_counts.values, color=['indianred', 'seagreen'])
axes[1, 0].set_title('Etiket Dağılımı', fontsize=14)
axes[1, 0].set_ylabel('Sayı')
for i, v in enumerate(label_counts.values):
    axes[1, 0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# 4. Kelime bulutu
try:
    from wordcloud import WordCloud
    all_text = ' '.join(df['cleaned_text'].tolist())
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(all_text)
    axes[1, 1].imshow(wordcloud, interpolation='bilinear')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Kelime Bulutu', fontsize=14)
except ImportError:
    # wordcloud yoksa en sık kelimeleri göster
    word_freq = Counter(all_text.split()).most_common(20)
    words, freqs = zip(*word_freq)
    axes[1, 1].barh(words, freqs, color='purple', alpha=0.7)
    axes[1, 1].set_title('En Sık 20 Kelime', fontsize=14)
    axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()

# İstatistikler
print('\n=== Veri Seti İstatistikleri ===')
print(f'Ortalama kelime sayısı: {np.mean(token_lengths):.1f}')
print(f'Ortalama karakter sayısı: {np.mean(char_lengths):.1f}')
print(f'Maks kelime sayısı: {max(token_lengths)}')
print(f'Toplam benzersiz kelime: {len(set(all_text.split()))}')

# ## 8. Sonuç
# Bu notebook'ta öğrendiklerimiz:
# 1. **Veri İndirme:** Kaggle API ile Türkçe tweet veri setini indirme ve yükleme
# 2. **Metin Temizleme:** HTML, URL, emoji kaldırma gibi temizleme teknikleri
# 3. **Tokenizasyon:** BPE algoritması ve HuggingFace tokenizer karşılaştırması
# 4. **Padding/Truncation:** Sabit uzunluklu diziler oluşturma ve attention mask
# 5. **Instruction Formatlama:** Alpaca ve Chat formatlarında veri hazırlama
# 6. **HuggingFace Dataset:** Veri seti oluşturma ve `.map()` ile batch işleme
# 7. **Görselleştirme:** Veri analizi ve kelime bulutu
# **Sonraki Adımlar:**
# - Veri setini bir LLM fine-tuning pipeline'ına entegre etme
# - Daha gelişmiş tokenizasyon yöntemleri deneme (SentencePiece, WordPiece)
# - Veri kalitesini artırma (duplike temizleme, normalization)
