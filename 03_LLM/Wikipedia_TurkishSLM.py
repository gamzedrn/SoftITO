# # Türkçe Küçük Dil Modeli (Wikipedia)
# Bu notebook'da Türkçe Wikipedia verileri üzerinde bigram ve LSTM dil modelleri eğitilecektir.
# **Veri Kaynağı:** Wikipedia API (Türkçe)
# **Modeller:** Bigram (Markov Zinciri), LSTM

# Gerekli paketlerin kurulumu
# !pip install -q wikipedia-api numpy matplotlib torch tqdm

# İçe aktarmalar
import re
import math
import random
import warnings
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import wikipediaapi
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Rastgelelik tohumu
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print('Tüm paketler başarıyla yüklendi!')
print(f'PyTorch versiyonu: {torch.__version__}')
print(f'Cihaz: {"cuda" if torch.cuda.is_available() else "cpu"}')

# ## 1. Wikipedia Veri Çekme
# Wikipedia API kullanarak Türkçe makaleler çekeceğiz.

# Wikipedia API ile Türkçe makaleler çekme
wiki_tr = wikipediaapi.Wikipedia(
    language='tr',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='TurkishSLMBot/1.0 (Educational Project)'
)

# Çekilecek makale başlıkları
article_titles = [
    'Yapay_zeka', 'Machine_learning', 'Python_(programlama_dili)',
    'Internet', 'Bilgisayar_bilimi', 'Veri_madenciliği',
    'Doğal_dil_işleme', 'Sinir_ağı', 'Derin_öğrenme',
    'Türkiye', 'İstanbul', 'Ankara',
    'Matematik', 'İstatistik', 'Fizik',
    'Kimya', 'Biyoloji', 'Tıp',
    'Ekonomi', 'Eğitim'
]

articles = {}
for title in article_titles:
    page = wiki_tr.page(title)
    if page.exists():
        text = page.text
        if len(text) > 100:  # Çok kısa makaleleri atla
            articles[title] = text
            print(f'  ✓ {title}: {len(text)} karakter')
        else:
            print(f'  ✗ {title}: Çok kısa ({len(text)} karakter)')
    else:
        print(f'  ✗ {title}: Bulunamadı')

print(f'\nToplam çekilen makale sayısı: {len(articles)}')
total_chars = sum(len(text) for text in articles.values())
print(f'Toplam karakter sayısı: {total_chars:,}')

# Tüm metinleri birleştir
full_text = '\n'.join(articles.values())

# Metin istatistikleri
print('=== Metin İstatistikleri ===')
print(f'Toplam karakter: {len(full_text):,}')
print(f'Toplam kelime: {len(full_text.split()):,}')
print(f'Toplam satır: {full_text.count(chr(10)):,}')

# Benzersiz karakterler
unique_chars = sorted(set(full_text))
print(f'\nBenzersiz karakter sayısı: {len(unique_chars)}')
print(f'Karakter seti: {chr(34).join(unique_chars[:50])}...')

# ## 2. Metin Ön İşleme
# Karakter düzeyinde tokenizasyon ve sözlük oluşturma.

# Metin temizleme
def clean_text(text):
    """Metni temizle"""
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    # Parantez içindeki referansları temizle
    text = re.sub(r'\[.*?\]', '', text)
    # Parantez içi notları temizle
    text = re.sub(r'\(.*?\)', '', text)
    return text.strip()

cleaned_text = clean_text(full_text)
print(f'Temizlenmiş metin: {len(cleaned_text):,} karakter')

# Karakter sözlüğü oluştur
chars = sorted(list(set(cleaned_text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f'\nSözlük boyutu: {vocab_size}')
print(f'İlk 20 karakter: {chars[:20]}')

# Metni sayılara dönüştür
encoded_text = [char_to_idx[ch] for ch in cleaned_text]
print(f'\nKodlanmış metin uzunluğu: {len(encoded_text):,}')
print(f'İlk 50 kod: {encoded_text[:50]}')
print(f'Çözülmüş: {[idx_to_char[i] for i in encoded_text[:50]]}')

# Eğitim ve test seti ayırma
train_ratio = 0.9
split_idx = int(len(encoded_text) * train_ratio)

train_data = encoded_text[:split_idx]
test_data = encoded_text[split_idx:]

print(f'Eğitim seti: {len(train_data):,} karakter')
print(f'Test seti: {len(test_data):,} karakter')

# Karakter frekansları
char_freq = Counter(cleaned_text)
print(f'\nEn sık 10 karakter:')
for ch, freq in char_freq.most_common(10):
    print(f'  {repr(ch)}: {freq:,} ({freq/len(cleaned_text)*100:.2f}%)')

# ## 3. Bigram Modeli
# İki karakter arasındaki olasılıkları öğrenen basit bir Markov zinciri.

# Bigram modeli oluştur
class BigramModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.bigram_counts = np.zeros((vocab_size, vocab_size))
        self.unigram_counts = np.zeros(vocab_size)

    def train(self, data):
        """Bigram sayılarını hesapla"""
        for i in range(len(data) - 1):
            self.bigram_counts[data[i], data[i+1]] += 1
            self.unigram_counts[data[i]] += 1
        self.unigram_counts[data[-1]] += 1

        # Laplace düzeltmesi ile olasılık matrisi
        self.prob_matrix = (self.bigram_counts + 1) / \
            (self.bigram_counts.sum(axis=1, keepdims=True) + vocab_size)

    def generate(self, seed, length=100, temperature=1.0):
        """Bigram modeli ile metin üret"""
        generated = [seed]
        current = seed

        for _ in range(length):
            # Olasılık dağılımından örnekleme
            probs = self.prob_matrix[current]
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / probs.sum()

            next_char = np.random.choice(self.vocab_size, p=probs)
            generated.append(next_char)
            current = next_char

        return generated

    def perplexity(self, data):
        """Perplexity hesapla"""
        log_prob = 0
        for i in range(len(data) - 1):
            log_prob += np.log(self.prob_matrix[data[i], data[i+1]] + 1e-10)
        return np.exp(-log_prob / (len(data) - 1))


# Bigram modelini eğit
bigram_model = BigramModel(vocab_size)
bigram_model.train(train_data)

print('Bigram modeli eğitildi!')
print(f'Eğitim perplexity: {bigram_model.perplexity(train_data):.2f}')
print(f'Test perplexity: {bigram_model.perplexity(test_data):.2f}')

# Rastgele bir başlangıç karakteri ile metin üret
seed_idx = char_to_idx.get('T', 0)  # 'T' karakteri ile başla
generated = bigram_model.generate(seed_idx, length=200, temperature=0.8)
generated_text = ''.join([idx_to_char[i] for i in generated])
print(f'\n--- Üretilen Metin (Bigram, T={0.8}) ---')
print(generated_text)

# Farklı sıcaklık değerleri ile üretim
temperatures = [0.5, 0.8, 1.0, 1.5]

print('=== Farklı Sıcaklık Değerleri ile Bigram Üretimi ===')
for temp in temperatures:
    generated = bigram_model.generate(seed_idx, length=100, temperature=temp)
    text = ''.join([idx_to_char[i] for i in generated])
    print(f'\n--- Sıcaklık = {temp} ---')
    print(text[:200])

# ## 4. LSTM Dil Modeli
# LSTM (Long Short-Term Memory) ağı ile karakter düzeyinde dil modeli eğiteceğiz.

# LSTM Modeli
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.2):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.dropout(self.embedding(x))

        if hidden is None:
            lstm_out, hidden = self.lstm(embed)
        else:
            lstm_out, hidden = self.lstm(embed, hidden)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


# Veri seti sınıfı
class CharDataset(Dataset):
    def __init__(self, data, seq_length=100):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# Model parametreleri
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Cihaz: {device}')

# Hyperparameters
SEQ_LENGTH = 100
BATCH_SIZE = 64
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# Veri setini oluştur
train_dataset = CharDataset(train_data, SEQ_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f'\nEğitim örnek sayısı: {len(train_dataset)}')
print(f'Batch sayısı: {len(train_loader)}')

# Modeli oluştur
model = CharLSTM(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nModel mimarisi:')
print(model)
print(f'\nToplam parametre: {total_params:,}')
print(f'Eğitilebilir parametre: {trainable_params:,}')

# ## 5. Eğitim Takibi
# Loss ve perplexity grafikleri, gradient clipping demonstration.

# Eğitim döngüsü
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# Kayıt defteri
train_losses = []
train_perplexities = []
gradient_norms = []

print('=== Eğitim Başlıyor ===')
print(f'Epoch sayısı: {NUM_EPOCHS}')
print(f'Batch boyutu: {BATCH_SIZE}')
print(f'Öğrenme hızı: {LEARNING_RATE}')
print()

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    num_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        hidden = model.init_hidden(batch_x.size(0), device)
        output, _ = model(batch_x, hidden)

        # Loss hesapla
        loss = criterion(output.reshape(-1, vocab_size), batch_y.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Gradient norm'u kaydet
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        gradient_norms.append(grad_norm.item())

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    # Epoch ortalaması
    avg_loss = epoch_loss / num_batches
    perplexity = np.exp(avg_loss)
    train_losses.append(avg_loss)
    train_perplexities.append(perplexity)

    # Öğrenme hızını güncelle
    scheduler.step(avg_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch {epoch+1:3d}/{NUM_EPOCHS} | '
              f'Loss: {avg_loss:.4f} | '
              f'Perplexity: {perplexity:.2f} | '
              f'Grad Norm: {gradient_norms[-1]:.4f}')

print('\nEğitim tamamlandı!')

# Eğitim grafikleri
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss grafiği
axes[0].plot(train_losses, 'b-', linewidth=2)
axes[0].set_title('Eğitim Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].grid(True, alpha=0.3)

# Perplexity grafiği
axes[1].plot(train_perplexities, 'r-', linewidth=2)
axes[1].set_title('Eğitim Perplexity')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Perplexity')
axes[1].grid(True, alpha=0.3)

# Gradient norm grafiği
axes[2].plot(gradient_norms, 'g-', linewidth=1, alpha=0.7)
axes[2].axhline(y=5.0, color='r', linestyle='--', label='Clip Eşiği (5.0)')
axes[2].set_title('Gradient Norm (Clipping ile)')
axes[2].set_xlabel('Step')
axes[2].set_ylabel('Gradient L2 Norm')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test perplexity hesapla
test_dataset = CharDataset(test_data, SEQ_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
test_loss = 0
test_batches = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        hidden = model.init_hidden(batch_x.size(0), device)
        output, _ = model(batch_x, hidden)

        loss = criterion(output.reshape(-1, vocab_size), batch_y.reshape(-1))
        test_loss += loss.item()
        test_batches += 1

test_perplexity = np.exp(test_loss / test_batches)
print(f'Test Perplexity: {test_perplexity:.2f}')

# ## 6. Metin Üretimi
# Eğitilen LSTM modeli ile farklı sıcaklık değerlerinde metin üreteceğiz.

def generate_text_lstm(model, start_text, char_to_idx, idx_to_char,
                       length=200, temperature=1.0, device='cpu'):
    """LSTM modeli ile metin üret"""
    model.eval()

    # Başlangıç metnini kodla
    input_seq = [char_to_idx.get(ch, 0) for ch in start_text]
    input_seq = torch.tensor([input_seq], dtype=torch.long).to(device)

    generated = list(start_text)
    hidden = model.init_hidden(1, device)

    # Başlangıç bağlamını oluştur
    with torch.no_grad():
        _, hidden = model(input_seq, hidden)

    # Karakter karakter üret
    current_char = torch.tensor([[input_seq[0, -1]]], dtype=torch.long).to(device)

    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(current_char, hidden)

        # Sıcaklık uygula
        logits = output[0, 0] / temperature
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        # Örnekleme
        next_idx = np.random.choice(len(probs), p=probs)
        next_char = idx_to_char[next_idx]

        generated.append(next_char)
        current_char = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return ''.join(generated)


# Farklı sıcaklık değerleri ile üretim
start_texts = ['Yapay', 'Bilgi', 'İstanbul']
temperatures = [0.5, 0.8, 1.0]

print('=== LSTM ile Metin Üretimi ===')
for start in start_texts:
    print(f'\nBaşlangıç: "{start}"')
    print('-' * 60)
    for temp in temperatures:
        generated = generate_text_lstm(model, start, char_to_idx, idx_to_char,
                                       length=150, temperature=temp, device=device)
        print(f'\nT={temp}: {generated[:200]}...')

# ## 7. Bigram vs LSTM Karşılaştırması
# İki modeli aynı başlangıç cümleleriyle karşılaştıracağız.

# Yan yana karşılaştırma
start_text = 'Yapay zeka'
seed_idx = char_to_idx.get(start_text[0], 0)

print('=== Bigram vs LSTM Karşılaştırması ===')
print(f'Başlangıç: "{start_text}"\n')

# Bigram üretimi
bigram_gen = bigram_model.generate(seed_idx, length=150, temperature=0.8)
bigram_text = ''.join([idx_to_char[i] for i in bigram_gen])

# LSTM üretimi
lstm_text = generate_text_lstm(model, start_text, char_to_idx, idx_to_char,
                               length=150, temperature=0.8, device=device)

print('--- Bigram Modeli ---')
print(bigram_text[:300])
print(f'\nPerplexity: {bigram_model.perplexity(test_data):.2f}')

print('\n--- LSTM Modeli ---')
print(lstm_text[:300])
print(f'\nTest Perplexity: {test_perplexity:.2f}')

# Karşılaştırma tablosu
print('=== Model Karşılaştırma Tablosu ===')
print(f'{"Özellik":<25} {"Bigram":<20} {"LSTM":<20}')
print('-' * 65)
print(f'{"Parametre Sayısı":<25} {vocab_size**2:<20,} {trainable_params:<20,}')
print(f'{"Eğitim Süresi":<25} {"Hızlı":<20} {"Yavaş":<20}')
print(f'{"Test Perplexity":<25} {bigram_model.perplexity(test_data):<20.2f} {test_perplexity:<20.2f}')
print(f'{"Metin Kalitesi":<25} {"Düşük":<20} {"Yüksek":<20}')
print(f'{"Bağlam Uzunluğu":<25} {"1 karakter":<20} {"100 karakter":<20}')
print(f'{"Gereken Veri":<25} {"Az":<20} {"Çok":<20}')

# ## 8. Sonuç
# Bu notebook'da:
# 1. Wikipedia API ile 20 Türkçe makale çektik
# 2. Karakter düzeyinde tokenizasyon yaptık
# 3. Bigram (Markov Zinciri) modeli eğittik
# 4. LSTM dil modeli eğittik
# 5. Farklı sıcaklık değerlerinde metin üretimi yaptık
# 6. İki modeli karşılaştırdık
# **Sonuçlar:**
# - Bigram modeli hızlıdır ancak kısa vadeli bağımlılıkları yakalar
# - LSTM daha karmaşık yapıları öğrenir ancak daha fazla veri ve eğitim süresi gerektirir
# - Sıcaklık değeri üretimin yaratıcılığını ve tutarlılığını dengeler
# **İyileştirme Önerileri:**
# - Daha büyük veri seti (Wikipedia'nın tamamı)
# - Transformer tabanlı model (GPT benzeri)
# - Subword tokenizasyonu (BPE)
