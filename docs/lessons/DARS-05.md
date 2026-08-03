# 5-dars: Trek A — wav2vec2 fine-tuning (Colab GPU'da)

## Maqsad

Bu darsdan keyin siz **transfer learning**ni tushunasiz: minglab soat nutqda
oldindan o'qitilgan katta modelni olib, o'z vazifangizga moslashtirishni.
Va 6-darsda taqqoslash uchun ikkinchi model tayyor bo'ladi.

Kutilgan natija: val aniqligi **98–99.5%**, trening ~30–45 daqiqa (T4 GPU).

---

## 1-qism. Nazariya

### 5.1 Transfer learning — asosiy g'oya

4-darsda model **noldan** boshladi: birinchi batchda u nutq nima ekanini ham
bilmasdi, hamma narsani 24 000 klipdan o'rganishi kerak edi.

wav2vec2 esa **53 000 soat** nutqni allaqachon "eshitgan". U buni yorliqsiz
o'rgangan — bunga **self-supervised learning** deyiladi: modeldan audio
bo'lagining berkitilgan qismini bashorat qilish talab qilinadi. Yorliq shart emas,
signalning o'zi o'qituvchi bo'ladi. Natijada model fonemalar, ovoz tembri,
artikulyatsiya kabi tushunchalarni o'rganib oladi.

Bizga faqat shu bilimni **ozgina burish** qoladi: "shu tushunchalardan foydalanib,
endi jinsni ayt". Shuning uchun 3 epoch yetadi, 15 emas.

### 5.2 Model tuzilishi va nimani muzlatamiz

`Wav2Vec2ForSequenceClassification` uch qismdan iborat:

| Qism | Nima qiladi | Biz nima qilamiz |
|---|---|---|
| Feature encoder (CNN) | xom to'lqinni 20 ms lik vektorlarga aylantiradi | **muzlatamiz** (freeze) |
| Transformer (12 qatlam) | kontekstni hisobga oladi | fine-tune qilamiz |
| Klassifikatsiya boshi | 2 ta logit chiqaradi | **noldan** o'qitamiz |

Nega feature encoder muzlatiladi? U eng past darajadagi akustikani o'rgangan —
bu barcha vazifalar uchun bir xil, qayta o'rgatish shart emas. Muzlatish
tezlikni ~30% oshiradi va kam datada barqarorlikni saqlaydi. Bu HuggingFace
hujjatlarida ham tavsiya etilgan standart amaliyot.

### 5.3 Nega LR 3e-5, 4-darsdagidek 3e-3 emas?

Bu fine-tuning'ning eng muhim qoidasi. Pretrained og'irliklar **allaqachon
yaxshi joyda** turibdi. Katta qadam tashlasangiz, model o'rgangan bilimini
buzib yuboradi — bunga **catastrophic forgetting** deyiladi.

Noldan o'qitishda LR katta (3e-3), fine-tuning'da 100 barobar kichik (3e-5).

Bundan ham yaxshirog'i — **ikki xil LR** (discriminative learning rates):

- backbone (transformer): `3e-5` — ehtiyotkorlik bilan
- klassifikatsiya boshi: `1e-3` — u tasodifiy og'irliklar bilan boshlaydi,
  unga tez o'rganish kerak

Buni PyTorch'da parametr guruhlari orqali beramiz (TODO-4).

### 5.4 Warmup

Trening boshida LR ni noldan asta ko'taramiz (birinchi ~10% qadamlarda).
Sababi: birinchi batchlarda klassifikatsiya boshi tasodifiy, gradientlar katta
va tartibsiz — ular to'g'ridan-to'g'ri backbone'ga urilsa, pretrained bilim
shikastlanadi. Warmup shu zarbani yumshatadi.

### 5.5 Mixed precision (AMP) — GPU'da 2 barobar tezlik

Odatda hisob float32 da ketadi. GPU esa float16 da ~2 barobar tez ishlaydi va
xotira ham 2 barobar kam ketadi. Lekin float16 diapazoni tor — kichik gradientlar
nolga aylanib qolishi mumkin (underflow).

Yechim — **AMP**: og'irliklar float32 da saqlanadi, hisob float16 da ketadi, va
`GradScaler` loss'ni katta songa ko'paytirib, gradientlarni float16 diapazonida
ushlab turadi, keyin qaytarib bo'ladi.

```python
with torch.autocast("cuda", dtype=torch.float16):
    out = model(x); loss = criterion(out, y)
scaler.scale(loss).backward()   # loss'ni kattalashtirib backward
scaler.step(optimizer)          # gradientni qaytarib kichraytirib qadam
scaler.update()                 # ko'paytiruvchini moslash
```

### 5.6 Kirish: mel emas, XOM to'lqin

Diqqat: bu yerda 3-darsdagi mel kesh **ishlatilmaydi**. wav2vec2 o'zining
feature encoder'iga ega — unga to'g'ridan-to'g'ri waveform beriladi.
Shuning uchun Colab'ga WAV fayllar kerak bo'ladi.

### 5.7 Halol taqqoslash sharti

6-darsdagi taqqoslash ma'noli bo'lishi uchun ikkala model **bir xil** train /
val / test bo'linishida ishlashi shart. Shuning uchun Colab'da datani qaytadan
tanlamaymiz — lokal `metadata.csv` ni yuklaymiz va aynan o'sha kliplarni
tortib olamiz.

---

## 2-qism. TOPSHIRIQ

### A. Colab tayyorgarligi

1. [colab.research.google.com](https://colab.research.google.com) → yangi notebook
2. **Runtime → Change runtime type → T4 GPU** (bu qadamni unutmang!)
3. Chap paneldagi papka belgisi orqali yuklang:
   - `scripts/05a_fetch_audio.py`
   - `scripts/05_finetune_w2v2.py`
   - `data/processed/metadata.csv`

Keyin katakchalarda:

```python
!nvidia-smi                       # GPU bormi tekshiring
!pip install -q transformers      # torch Colab'da allaqachon bor
!python 05a_fetch_audio.py        # ~8-15 daqiqa, /content/audio ga 30K WAV
!python 05_finetune_w2v2.py       # ~30-45 daqiqa
```

Natijani Drive'ga saqlashni unutmang (sessiya o'lsa yo'qoladi):

```python
from google.colab import drive; drive.mount('/content/drive')
!cp -r /content/models/w2v2_best /content/drive/MyDrive/
!cp /content/models/w2v2_history.json /content/drive/MyDrive/
```

### B. Kod TODO'lari

`scripts/05_finetune_w2v2.py` da **4 ta TODO**:

| TODO | Nima | Nimani o'rgatadi |
|---|---|---|
| 1 | `AudioDataset.__getitem__` | xom to'lqin + crop (mel emas!) |
| 2 | `build_model` | pretrained yuklash + feature encoder'ni muzlatish |
| 3 | `train_one_epoch` | AMP bilan training loop |
| 4 | `build_optimizer` | backbone va head uchun har xil LR |

`05a_fetch_audio.py` da TODO yo'q — tayyor.

**Muvaffaqiyat mezonlari**: val aniqligi ≥ 98%, muzlatilgan parametrlar soni
~4.2M, o'qitiladiganlar ~90M.

## 3-qism. Savollar

1. Nega fine-tuning'da LR 100 barobar kichik olinadi?
2. Feature encoder'ni muzlatish nima beradi va nima yo'qotadi?
3. AMP'da `GradScaler` aynan qanday muammoni hal qiladi?
4. Nega bu darsda 3-darsdagi mel kesh ishlatilmaydi?
