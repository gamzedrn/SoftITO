# # PEFT/LoRA ile Türkçe Dil Modeli İnce Ayarı
# Bu notdefteri, **PEFT (Parameter-Efficient Fine-Tuning)** ve **LoRA (Low-Rank Adaptation)** yöntemlerini Türkçe instruction verisiyle uygulamalı olarak gösterir.
# **Kullanılan Veri Seti:** `merve/turkish_instructions` (HuggingFace Datasets)
# **Model:** `gpt2` (124M parametre)
# **Konular:**
# 1. Türkçe instruction verisi yükleme ve hazırlama
# 2. LoRA'nın matematiksel temeli ve parametre tasarrufu
# 3. LoRA ile GPT-2 fine-tuning
# 4. QLoRA karşılaştırması ve bellek analizi
# 5. Tahmin ve değerlendirme

# ## 0. Gereksinimlerin Kurulması

# !pip install -q transformers datasets peft accelerate bitsandbytes torch matplotlib numpy

# ## 1. Veri Setini Yükleme
# HuggingFace `datasets` kütüphanesi ile Türkçe instruction veri setini yüklüyoruz. `merve/turkish_instructions` veri seti, çeşitli Türkçe görevler için hazırlanmış instruction-response çiftleri içerir.

from datasets import load_dataset

dataset = load_dataset("merve/turkish_instructions", split="train[:5000]")
print(f"Örnek sayısı: {len(dataset)}")
print(f"Sütunlar: {dataset.column_names}")
print(f"\n--- İlk Örnek ---")
print(dataset[0])

for i in range(3):
    print(f"\n=== Örnek {i+1} ===")
    for col in dataset.column_names:
        val = dataset[i][col]
        print(f"{col}: {val[:200] if isinstance(val, str) else val}")

# ## 2. LoRA'nın Matematiksel Temeli
# LoRA, büyük ağırlık matrislerini düşük ranklı (low-rank) çarpanlar olarak ayrıştırarak modeldeki trainable parametre sayısını büyük ölçüde azaltır.
# **Temel formül:**
# $$W' = W + \Delta W = W + BA$$
# Burada:
# - $W \in \mathbb{R}^{d \times d}$: Orijinal donmuş ağırlık matrisi
# - $B \in \mathbb{R}^{d \times r}$: Düşük ranklı ağırlık matrisi
# - $A \in \mathbb{R}^{r \times d}$: Düşük ranklı ağırlık matrisi
# - $r \ll d$: Lo rankı (örn. r=8)
# **Parametre tasarrufu:**
# - Orijinal: $d^2$ parametre
# - LoRA: $2 \cdot d \cdot r$ parametre
# - Tasarruf: $1 - \frac{2r}{d}$

import numpy as np
import matplotlib.pyplot as plt

d = 768  # GPT-2 gizli boyutu
r_values = [1, 2, 4, 8, 16, 32, 64]

params_full = d * d
params_lora = [d * r + r * d for r in r_values]
savings = [(1 - p / params_full) * 100 for p in params_lora]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Parametre tasarrufu grafiği
axes[0].bar(range(len(r_values)), savings, color='steelblue', edgecolor='navy')
axes[0].set_xticks(range(len(r_values)))
axes[0].set_xticklabels([f'r={r}' for r in r_values])
axes[0].set_ylabel('Parametre Tasarrufu (%)')
axes[0].set_title('LoRA Rank (r) vs Parametre Tasarrufu')
for i, (s, p) in enumerate(zip(savings, params_lora)):
    axes[0].text(i, s + 0.5, f'{s:.1f}%\n({p:,})', ha='center', fontsize=8)
axes[0].grid(axis='y', alpha=0.3)

# Parametre sayısı karşılaştırması
axes[1].plot(r_values, [p / 1e6 for p in params_lora], 'o-', color='crimson', linewidth=2, markersize=8, label='LoRA')
axes[1].axhline(y=params_full / 1e6, color='gray', linestyle='--', linewidth=2, label=f'Orijinal ({params_full/1e6:.2f}M)')
axes[1].set_xlabel('LoRA Rank (r)')
axes[1].set_ylabel('Parametre Sayısı (Milyon)')
axes[1].set_title('LoRA vs Orijinal Parametre Sayısı')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Orijinal parametre (d×d): {params_full:,}")
print(f"\nRank'a göre LoRA parametreleri:")
for r, p, s in zip(r_values, params_lora, savings):
    print(f"  r={r:2d}: {p:>8,} parametre | {s:.1f}% tasarruf")

# ## 3. Veri Hazırlığı - Instruction Formatı
# Veri setindeki instruction-response çiftlerini modelin anlayabileceği formata dönüştürüyoruz. **Alpaca formatı** kullanacağız:
# ```
# ### Talimat:
# {instruction}
# ### Yanıt:
# {response}
# ```

def format_instruction(sample):
    instruction = sample.get("instruction", "")
    response = sample.get("response", "")
    input_text = sample.get("input", "")

    if input_text:
        prompt = f"### Talimat:\n{instruction}\n\n### Girdi:\n{input_text}\n\n### Yanıt:\n{response}"
    else:
        prompt = f"### Talimat:\n{instruction}\n\n### Yanıt:\n{response}"
    return prompt

# Veri setindeki sütun adlarını kontrol edelim
print("Mevcut sütunlar:", dataset.column_names)
print(f"\nÖrnek formatlanmış veri:")
print(format_instruction(dataset[0]))

from transformers import AutoTokenizer

# Tokenizer'ı geçici olarak yükleyip token uzunluğunu analiz edelim
temp_tokenizer = AutoTokenizer.from_pretrained("gpt2")

formatted_texts = [format_instruction(dataset[i]) for i in range(min(100, len(dataset)))]
token_lengths = [len(temp_tokenizer.encode(text)) for text in formatted_texts]

plt.figure(figsize=(10, 4))
plt.hist(token_lengths, bins=30, color='teal', edgecolor='black', alpha=0.7)
plt.axvline(x=np.median(token_lengths), color='red', linestyle='--', label=f'Medyan: {np.median(token_lengths):.0f} token')
plt.axvline(x=np.percentile(token_lengths, 90), color='orange', linestyle='--',
            label=f'90. persentil: {np.percentile(token_lengths, 90):.0f} token')
plt.xlabel('Token Uzunluğu')
plt.ylabel('Örnek Sayısı')
plt.title('Formatlanmış Talimatların Token Uzunluk Dağılımı')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"Ortalama token uzunluğu: {np.mean(token_lengths):.1f}")
print(f"Medyan token uzunluğu: {np.median(token_lengths):.0f}")
print(f"Maks token uzunluğu: {max(token_lengths)}")

MAX_LENGTH = 256

def preprocess_function(examples):
    cols = list(examples.keys())
    instr_col = next((c for c in cols if 'instruction' in c.lower() or 'prompt' in c.lower()), cols[0])
    resp_col = next((c for c in cols if 'response' in c.lower() or 'output' in c.lower()), cols[-1])
    input_col = next((c for c in cols if c not in [instr_col, resp_col] and 'input' in c.lower()), None)
    all_texts = []
    n = len(examples[instr_col])
    for i in range(n):
        sample = {
            'instruction': examples[instr_col][i],
            'response': examples[resp_col][i],
            'input': examples[input_col][i] if input_col else ''
        }
        all_texts.append(format_instruction(sample))

    tokenized = temp_tokenizer(
        all_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Veri tokenize ediliyor"
)

print(f"İşlenmiş veri boyutu: {len(tokenized_dataset)}")
print(f"Örnek input_ids uzunluğu: {len(tokenized_dataset[0]['input_ids'])}")
print(f"Örnek labels[0:10]: {tokenized_dataset[0]['labels'][:10]}")

# ## 4. Model ve Tokenizer Yükleme
# HuggingFace Transformers ile GPT-2 modelini yüklüyoruz. GPT-2, İngilizce odaklı bir modeldir; Türkçe instruction verisiyle ince ayar yaparak Türkçe yeteneklerini artıracağız.

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Model: {model_name}")
print(f"Toplam parametre: {model.num_parameters():,}")
print(f"Model mimarisi: {model.config}")
print(f"Tokenizer boyutu: {tokenizer.vocab_size}")
print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

from torch import no_grad

# Fine-tuning öncesi modelin Türkçe tahminini test edelim
test_prompt = "### Talimat:\nTürkçe yazarak selamlaşma kurallarını açıkla.\n\n### Yanıt:\n"
inputs = tokenizer(test_prompt, return_tensors="pt")

with no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2
    )

print("=== Fine-tuning ÖNCESİ (Orijinal GPT-2) ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ## 5. LoRA Yapılandırması
# LoRA (Low-Rank Adaptation) parametrelerini yapılandırıyoruz.
# ### Önemli Parametreler:
# | Parametre | Açıklama | Seçim | 
# |-----------|----------|-------|
# | `r` | LoRA rankı (düşük rank boyutu) | r=8 (dengeli) |
# | `lora_alpha` | Ölçekleme faktörü (genellikle r'nin 2-4 katı) | alpha=32 |
# | `lora_dropout` | Aşırı öğrenmeyi önleme | 0.1 |
# | `target_modules` | LoRA uygulanacak katmanlar | c_attn, c_proj |
# ### Hedef Modüller Neden `c_attn` ve `c_proj`?
# - `c_attn`: Dikkat (attention) katmanındaki Q, K, V ağırlıkları
# - `c_proj`: Dikkat çıktısının projeksiyon katmanı
# - Bu katmanlar modelin "anlama" kapasitesinin çoğunu taşır

from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],
    bias="none"
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

# LoRA uygulanan katmanları görelim
print("=== LoRA Uygulanan Katmanlar ===")
for name, module in peft_model.named_modules():
    if "lora" in name:
        print(f"  {name}: {module}")

print(f"\n=== Model Parametre Dağılımı ===")
total = sum(p.numel() for p in peft_model.parameters())
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
frozen = total - trainable
print(f"Toplam parametre:      {total:>12,}")
print(f"Eğitilebilir (LoRA):   {trainable:>12,} ({100*trainable/total:.2f}%)")
print(f"Donmuş (orijinal):     {frozen:>12,} ({100*frozen/total:.2f}%)")

# ## 6. Eğitim
# LoRA parametrelerini Türkçe instruction verisi üzerinde eğitiyoruz. Eğitim sırasında:
# - Orijinal GPT-2 ağırlıkları **donmuş** (frozen) kalır
# - Yalnızca LoRA ağırlıkları ($A$ ve $B$ matrisleri) güncellenir
# - Bu sayede çok daha az bellek ve hesaplama gücü kullanılır

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Eğitim ve doğrulama bölmesi
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Eğitim verisi: {len(train_dataset)} örnek")
print(f"Doğrulama verisi: {len(eval_dataset)} örnek")

import os

output_dir = "./lora-turkish-gpt2-results"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    remove_unused_columns=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("Eğitim yapılandırması hazır.")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"FP16: {training_args.fp16}")

# Eğitimi başlat
train_result = trainer.train()

# Eğitim istatistiklerini göster
print(f"\n=== Eğitim Tamamlandı ===")
print(f"Toplam eğitim adımı: {train_result.global_step}")
print(f"Son eğitim kaybı: {train_result.training_loss:.4f}")
print(f"Eğitim süresi: {train_result.metrics['train_runtime']:.1f} saniye")

# Eğitim loglarını al ve kayıp grafiğini çiz
log_history = trainer.state.log_history

train_losses = [(log["step"], log["loss"]) for log in log_history if "loss" in log]
eval_losses = [(log["step"], log["eval_loss"]) for log in log_history if "eval_loss" in log]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Kayıp (Loss) grafiği
if train_losses:
    steps, losses = zip(*train_losses)
    axes[0].plot(steps, losses, 'b-', alpha=0.7, label='Eğitim Kaybı')
if eval_losses:
    steps, losses = zip(*eval_losses)
    axes[0].plot(steps, losses, 'r-o', linewidth=2, label='Doğrulama Kaybı')
axes[0].set_xlabel('Adım')
axes[0].set_ylabel('Kayıp (Loss)')
axes[0].set_title('Eğitim ve Doğrulama Kaybı')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Öğrenme oranı grafiği
lr_entries = [(log["step"], log["learning_rate"]) for log in log_history if "learning_rate" in log]
if lr_entries:
    steps, lrs = zip(*lr_entries)
    axes[1].plot(steps, lrs, 'g-', linewidth=2)
    axes[1].set_xlabel('Adım')
    axes[1].set_ylabel('Öğrenme Oranı')
    axes[1].set_title('Öğrenme Oranı Zamanlaması (Scheduler)')
    axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Modeli kaydet
peft_model.save_pretrained(os.path.join(output_dir, "lora-adapter"))
tokenizer.save_pretrained(os.path.join(output_dir, "lora-adapter"))
print(f"Model kaydedildi: {output_dir}/lora-adapter")

# ## 7. Tahmin ve Değerlendirme
# Fine-tuning sonrası modelin Türkçe instruction karşılama performansını test ediyoruz.

def generate_response(model, tokenizer, prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Yanıt kısmını ayıkla
    if "### Yanıt:" in response:
        response = response.split("### Yanıt:")[-1].strip()
    return response

print("Tahmin fonksiyonu hazır.")

# Test talimatları
test_instructions = [
    "Türkçe yazarak selamlaşma kurallarını açıkla.",
    "İstanbul'un tarihi ve turistik mekanlarını listele.",
    "Python programlama dilinin temel özelliklerini açıkla.",
    "İklim değişikliğinin Türkiye'ye etkilerini tartış.",
    "Bir günlük beslenme programı oluştur."
]

print("=" * 70)
print("FINE-TUNING SONRASI TAHMİNLER")
print("=" * 70)

for i, instruction in enumerate(test_instructions, 1):
    prompt = f"### Talimat:\n{instruction}\n\n### Yanıt:\n"
    response = generate_response(peft_model, tokenizer, prompt)
    print(f"\n--- Test {i} ---")
    print(f"Talimat: {instruction}")
    print(f"Yanıt: {response[:300]}")

# Aynı talimatları orijinal GPT-2 ile karşılaştıralım
print("=" * 70)
print("ORİJİNAL GPT-2 İLE KARŞILAŞTIRMA")
print("=" * 70)

for i, instruction in enumerate(test_instructions[:3], 1):
    prompt = f"### Talimat:\n{instruction}\n\n### Yanıt:\n"

    # Orijinal model
    orig_response = generate_response(model, tokenizer, prompt)

    # Fine-tuned model
    ft_response = generate_response(peft_model, tokenizer, prompt)

    print(f"\n--- Test {i} ---")
    print(f"Talimat: {instruction}")
    print(f"\n[Orijinal GPT-2]\n{orig_response[:200]}")
    print(f"\n[LoRA Fine-tuned]\n{ft_response[:200]}")
    print("-" * 70)

# ## 8. QLoRA Karşılaştırması
# Farklı fine-tuning yöntemlerinin bellek kullanımı ve parametre verimliliği karşılaştırması.
# | Yöntem | Açıklama | Bellek | Parametre | Hız |
# |--------|----------|--------|-----------|-----|
# | **Full Fine-Tuning** | Tüm parametreler eğitilir | Yüksek | 124M | Yavaş |
# | **LoRA** | Düşük ranklı adaptasyon | Orta | ~0.8M | Orta |
# | **QLoRA** | 4-bit quantize + LoRA | Düşük | ~0.8M | Orta |
# | **Adapter** | Ek katmanlar eklenir | Orta | ~3M | Orta |

import pandas as pd

# Bellek ve performans karşılaştırma tablosu
comparison_data = {
    "Yöntem": ["Full Fine-Tuning", "LoRA (r=8)", "LoRA (r=16)", "QLoRA (4-bit)", "Adapter"],
    "Eğitilebilir Parametre": [
        "124M (%100)",
        f"{2 * 768 * 8:,} (%{(2*768*8)/124000000*100:.2f})",
        f"{2 * 768 * 16:,} (%{(2*768*16)/124000000*100:.2f})",
        f"{2 * 768 * 8:,} (%{(2*768*8)/124000000*100:.2f})",
        "~3M (%2.4)"
    ],
    "Tahmini Bellek (GPU)": [
        ">10 GB",
        "~3-4 GB",
        "~3-5 GB",
        "~2-3 GB",
        "~4-5 GB"
    ],
    "Eğitim Hızı": [
        "Yavaş",
        "Orta",
        "Orta",
        "Orta",
        "Orta"
    ],
    "Kalite": [
        "En Yüksek",
        "Yüksek",
        "Çok Yüksek",
        "Yüksek",
        "Orta"
    ],
    "Avantaj": [
        "Tam esneklik",
        "Hafif, esnek",
        "Daha güçlü adaptasyon",
        "En az bellek",
        "Modüler"
    ],
    "Dezavantaj": [
        "Çok pahalı",
        "Rank seçimi kritik",
        "Biraz daha fazla bellek",
        "Quantization hatası",
        "Ek karmaşıklık"
    ]
}

df = pd.DataFrame(comparison_data)
print("=== Fine-tuning Yöntem Karşılaştırması (GPT-2, 124M) ===")
print(df.to_string(index=False))

# Rank (r) değerine göre LoRA parametre analizi
d = 768
r_values_analysis = [1, 2, 4, 8, 16, 32, 64, 128, 256]

full_params = d * d
lora_params = [2 * d * r for r in r_values_analysis]
ratios = [p / full_params * 100 for p in lora_params]

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2ecc71' if r <= 16 else '#f39c12' if r <= 64 else '#e74c3c' for r in r_values_analysis]

bars = ax.bar(range(len(r_values_analysis)), ratios, color=colors, edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(r_values_analysis)))
ax.set_xticklabels([f'r={r}' for r in r_values_analysis])
ax.set_ylabel('Orijinal Parametreye Oran (%)')
ax.set_title('LoRA Rank Seçimi vs Eğitim Oranı (GPT-2, d=768)')

# Referans çizgileri
ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='%5 eşik (iyi tercih)')
ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='%10 eşik')
ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='%20 eşik (fazla)')

for bar, ratio, p in zip(bars, ratios, lora_params):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{ratio:.1f}%\n({p:,})', ha='center', va='bottom', fontsize=8)

ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("\nÖnerilen rank aralıkları:")
print("  r=4-8  : Hafif görevler, hızlı eğitim")
print("  r=8-16 : Dengeli (en çok tercih edilen)")
print("  r=16-64: Karmaşık görevler, daha güçlü adaptasyon")
print("  r>64   : Nadiren gerekli, overfitting riski")

# ## 9. Sonuç
# ### Bu notdefterde neler yaptık:
# 1. **Veri Hazırlığı:** `merve/turkish_instructions` veri setini Alpaca formatına dönüştürdük ve tokenleştirdik.
# 2. **LoRA Matematiği:** $W' = W + BA$ formülünü ve parametre tasarrufunu grafiklerle gösterdik.
# 3. **LoRA Fine-Tuning:** GPT-2 modeline LoRA uygulayarak ~%0.66 parametre ile Türkçe instruction verisi üzerinde eğittik.
# 4. **Karşılaştırma:** Full fine-tuning, LoRA ve QLoRA yöntemlerini bellek ve performans açısından karşılaştırdık.
# 5. **Değerlendirme:** Fine-tuning öncesi ve sonrası Türkçe tahmin kalitesini karşılaştırdık.
# ### Önemli Bulgular:
# - LoRA, **%0.66** parametre ile orijinal modelin büyük bir kısmını koruyarak Türkçe yetenekler kazandırır
# - **r=8** değeri genellikle iyi bir başlangıç noktasıdır
# - **QLoRA** ile bellek kullanımı daha da azaltılabilir (4-bit quantization)
# - Türkçe instruction fine-tuning, modelin Türkçe anlama ve üretme kapasitesini artırır
# ### İleri Düzey Konular:
# - Daha büyük modeller (LLaMA, Mistral) ile deneme
# - Farklı rank değerleri ile hyperparameter araması
# - RLHF (Reinforcement Learning from Human Feedback) entegrasyonu
# - Türkçe evaluation benchmark'ları ile sistematik değerlendirme
