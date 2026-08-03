# 3-dars: Log-mel spektrogramma — audioni "rasm"ga aylantirish

## Maqsad

Bu darsdan keyin siz audioni CNN tushunadigan 2D tasvirga aylantirishni bilasiz va
30 000 klipning barchasi tayyor feature sifatida diskda keshlangan bo'ladi.
Va o'z ko'zingiz bilan ko'rasiz: erkak va ayol spektrogrammasi qanday farq qiladi.

---

## 1-qism. Nazariya

### 3.1 Nega waveform'ni to'g'ridan-to'g'ri CNN'ga bermaymiz?

2 sekund audio = 32 000 ta son. Muammolar:

- **Juda uzun**: CNN uchun 32 000 uzunlikdagi ketma-ketlik og'ir.
- **Ma'lumot yashirin**: bizga kerak narsa — F0 (pitch) va formantlar, ya'ni
  **chastota** haqidagi ma'lumot. Waveform'da u bevosita ko'rinmaydi.
- **Faza ahamiyatsiz**: waveform'ni bir necha millisekundga surib qo'ysangiz,
  sonlar butunlay o'zgaradi, lekin quloq farqni sezmaydi.

Yechim: signalni **vaqt × chastota** tekisligiga yoyamiz. Natija — matritsa,
ya'ni kulrang rasm. CNN esa aynan rasmlar uchun yaratilgan.

### 3.2 STFT — qisqa vaqtli Fure almashtirishi

Fure almashtirishi (FFT) signalni chastotalarga yoyadi, lekin "qachon" degan
ma'lumotni yo'qotadi. Nutq esa doim o'zgaradi. Yechim oddiy: signalni qisqa
bo'laklarga bo'lib, **har bo'lakda alohida** FFT qilamiz.

Bizning parametrlar:
- `n_fft = 512` → oyna uzunligi 512 namuna = **32 ms**
- `hop_length = 160` → oynalar orasidagi qadam 160 namuna = **10 ms**

Oynalar ustma-ust tushadi (32 ms oyna, 10 ms qadam). Natija: sekundiga 100 ta
"vaqt kadri", har kadrda 257 ta chastota qiymati → `(257, T)` matritsa.

**Nega aynan 32 ms?** Bu vaqt-chastota kompromissi:
- Oyna qisqa bo'lsa → vaqtni aniq ko'ramiz, chastotani xira
- Oyna uzun bo'lsa → chastotani aniq, lekin nutq o'zgarishini o'tkazib yuboramiz

32 ms — nutq uchun standart: shu vaqt ichida nutq taxminan o'zgarmaydi, lekin
100 Hz atrofidagi pitch'ni ajratishga yetarli (32 ms ichida 100 Hz to'lqin 3 marta tebranadi).

### 3.3 Mel shkalasi — 257 chastotani 64 taga siqish

257 chiziqli chastota kanali ortiqcha va noqulay. Inson qulog'i chastotani
**chiziqli emas, logarifmik** eshitadi: 100 va 200 Hz orasidagi farqni yaqqol
sezasiz, 5000 va 5100 Hz orasidagini deyarli yo'q.

Mel shkalasi shuni taqlid qiladi. 64 ta mel filtri: past chastotalarda **tor va
zich**, yuqorida **keng va siyrak**. `(257, T)` → `(64, T)`.

Bizning vazifa uchun bu ayni muddao: F0 (85–255 Hz) va formantlar past
chastotalarda joylashgan — aynan mel shkalasi eng aniq bo'lgan zonada.

### 3.4 dB — logarifmik amplituda

Energiya qiymatlari juda keng diapazonda (million marta farq). Neyron tarmoq
bunday sonlarni yomon hazm qiladi. `power_to_db` logarifm oladi va diapazonni
**-80 … 0 dB** ga siqadi.

`ref=np.max` degani: har klipning eng baland nuqtasi 0 dB deb olinadi. Bu
yozuv balandligini (mikrofon kuchaytirishi) tenglashtiradi — biz uchun foydali,
chunki jins ovoz balandligidan emas, spektr **shaklidan** bilinadi.

### 3.5 Kesh va memmap

Har epochda 30 000 klipni qayta hisoblash — 3 daqiqa isrof, ×15 epoch = 45 daqiqa.
Bir marta hisoblab, diskka yozamiz.

Hajmi: `30000 × 64 × 400 × 2 bayt (float16)` ≈ **1.5 GB**. RAM'ga to'liq
yuklash shart emas — `np.memmap` diskdagi massivni xuddi oddiy massivdek
indekslashga imkon beradi, faqat kerakli qismi o'qiladi.

**Nega 400 kadr (4 s)?** Kliplarning yarmi undan qisqa, uzunlarining oxiri
kesiladi. Trening paytida baribir 2 sekundlik tasodifiy bo'lak olamiz —
4 sekundlik zaxira yetarli xilma-xillik beradi, lekin hajm ikki barobar oshmaydi.
Bu — muhandislik kelishuvi.

**Padding qiymati muhim**: qisqa kliplarni to'ldirishda 0 ishlatmang! Bizning
shkalada 0 dB = *eng baland tovush*. Sukunat = **-80 dB**. Noto'g'ri padding
modelga "bu yerda qattiq shovqin bor" deb yolg'on aytadi.

### 3.6 Normalizatsiya — va yana leakage

Neyron tarmoq kirish qiymatlari ~0 atrofida, dispersiyasi ~1 bo'lsa yaxshi
o'rganadi. Shuning uchun har mel bandidan o'rtachani ayirib, standart og'ishga
bo'lamiz.

**Muhim**: o'rtacha va std **faqat train** splitda hisoblanadi. Agar butun
datasetda hisoblasangiz, test'dagi ma'lumot statistika orqali modelga sizib
o'tadi. Bu ham leakage — 2-darsdagi speaker leakage bilan bir oilada.

---

## 2-qism. TOPSHIRIQ

`scripts/03_features.py` da **3 ta TODO**:

| TODO | Funksiya | Nimani o'rgatadi |
|---|---|---|
| 1 | `wav_to_logmel(y, sr)` | STFT → mel → dB zanjiri |
| 2 | `pad_or_trim(mel, max_frames)` | qat'iy shakl + to'g'ri padding qiymati |
| 3 | `compute_train_stats(...)` | oqimli statistika, train-only qoidasi |

Ishga tushirish (~4–6 daqiqa):

```powershell
python scripts\03_features.py
```

Natija:
- `data/processed/mels.npy` (~1.5 GB, memmap)
- `data/processed/mel_lengths.npy`
- `data/processed/norm_stats.json`
- `data/processed/mel_examples.png` ← **buni albatta oching va ko'ring!**

**Muvaffaqiyat mezonlari**: shakl `(30000, 64, 400)`, normalizatsiyadan keyin
train o'rtachasi ≈ 0 va std ≈ 1, rasmda erkak/ayol farqi ko'rinadi.

## 3-qism. Savollar

1. `hop_length = 160` bo'lsa, 1 sekund audiodan nechta vaqt kadri chiqadi?
2. Nega padding uchun 0 emas, -80 ishlatamiz?
3. Nega normalizatsiya statistikasi faqat train'da hisoblanadi?
4. Rasmga qarab ayting: erkak va ayol spektrogrammasida qaysi qism farq qilyapti?
