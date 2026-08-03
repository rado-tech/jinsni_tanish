# 2-dars: Data tayyorlash — filtr, balans, speaker-disjoint split

## Maqsad

Bu darsdan keyin sizda diskda **tayyor, toza, ishonchli training dataset** bo'ladi: ~30 000 ta 16 kHz WAV + metadata.csv, train/val/test ga spiker bo'yicha bo'lingan. Va siz data engineering'ning 4 ta tamoyilini kod bilan qo'llagan bo'lasiz.

---

## 1-qism. Nazariya

### 2.1 Garbage in, garbage out

Model sifati hech qachon data sifatidan oshmaydi. Crowdsourced datasetda har xil chiqindi bo'ladi: mikrofoni buzuq yozuvlar, bo'sh kliplar, noto'g'ri bosilgan yorliqlar. Yaxshiyamki datasetda jamoa bahosi bor: `reported_count` (shikoyatlar), `upvotes_count`/`downvotes_count`. Bundan filtr qilamiz. Filtr qoidasi **oddiy va tushuntiriladigan** bo'lishi kerak — "nega bu klip tashlandi?" degan savolga bir gapda javob bera olishingiz kerak.

Muhim tezlik hiylasi: 1-darsda metadata'da tayyor `duration` ustuni borligini topdik. Filtrni **audio dekodlashdan OLDIN** metadata bo'yicha qilsak, yomon kliplarning og'ir audio qismini umuman ochmaymiz.

### 2.2 Per-speaker cap (har spikerdan limit)

1-darsda o'lchadingiz: top-5 spiker klipларning 23%ini yozgan. Cheklovsiz olsak, model bir hovuch "super-spiker"ning ovoz xususiyatlariga moslashib qoladi — diversity past, generalization yomon. `MAX_PER_SPEAKER = 40` qo'ysak, 30 000 klip uchun kamida 750 turli spiker kerak bo'ladi. **Diversity = generalization.**

### 2.3 Train / Validation / Test — har birining roli

| Split | Ulush | Kim ishlatadi | Qachon |
|---|---|---|---|
| train | 80% | model (og'irliklar o'rganadi) | har epoch |
| val | 10% | **biz** (learning rate, epoch soni, model tanlash) | trening davomida |
| test | 10% | hech kim! | faqat 5-darsda, final taqqoslashda 1 marta |

Val'ni model to'g'ridan-to'g'ri ko'rmaydi, lekin biz unga qarab qaror qilamiz — shuning uchun u ham asta "ifloslanadi". Test — muhrlangan sandiq: unga qarab birorta qaror qilinmaydi, aks holda final baho yolg'on bo'ladi.

Va asosiysi: bo'lish birligi klip emas, **SPIKER** — bitta odamning barcha kliplari faqat bitta splitda yotadi.

### 2.4 Yorliqqa ishonma — tekshir (data validation)

Agar bitta `client_id` bazi kliplarda "Ayol", bazilarida "Erkak" bo'lsa — kimdir formani noto'g'ri to'ldirgan. Bunday spikerning QAYSI yorlig'i to'g'riligini bilmaymiz → hammasini chiqarib tashlaymiz. Bir necha yuz klip yo'qotamiz, lekin yorliqlarga ishonch ortadi. Arzon sug'urta.

### 2.5 Reproducibility (qayta tiklanuvchanlik)

Split tasodifiy — lekin `random.Random(SEED)` bilan: skriptni qayta ishlatsangiz AYNAN o'sha split chiqadi. Busiz "model yaxshilandi"mi yoki "split omadli tushdi"mi — hech qachon bilolmaysiz. Metadata.csv esa har klipning pasporti: qayerdan kelgan, qaysi splitda.

---

## 2-qism. TOPSHIRIQ

`scripts/02_prepare.py` da **4 ta TODO** (endi funksiya darajasida — 1-darsdagidan qiyinroq):

| TODO | Funksiya | Nimani o'rgatadi |
|---|---|---|
| 1 | `is_good_clip(row)` — sifat filtri | metadata bo'yicha arzon filtrlash |
| 2 | `should_take(...)` — kvota va cap | balans + diversity nazorati |
| 3 | `find_conflicting_speakers(...)` — yorliq ziddiyatlari | data validation |
| 4 | `assign_splits(...)` — speaker-disjoint 80/10/10 | leakage'siz split algoritmi |

Har funksiya ustida aniq spetsifikatsiya va hint yozilgan. Skriptning qolgan qismi (yuklab olish sikli, WAV saqlash, CSV, tekshiruv assert'lari) tayyor.

Ishga tushirishdan oldin (tavsiya): HF token o'rnating — anonim rate-limit bilan katta yuklashda to'xtab qolishi mumkin:

1. huggingface.co da akkaunt oching → Settings → Access Tokens → New token (Read turi)
2. Terminalda: `huggingface-cli login` → tokenni qo'ying

Keyin:

```powershell
venv\Scripts\Activate.ps1
python scripts\02_prepare.py
```

Kutilyapti: ~10–25 daqiqa (internetga bog'liq), ~4 GB disk. Natija: `data/processed/audio/` da ~30K WAV, `metadata.csv`, `prepare_report.json`.

**Muvaffaqiyat mezonlari** (report'da ko'rinadi, men shuni tekshiraman):
- Umumiy klip ≈ 30 000, har splitda gender ~50/50
- Split ulushlari ~80/10/10 (±1%)
- Spiker overlap = 0 (assert'lar yiqilmagan)
- Ziddiyatli spikerlar topilgan va chiqarilgan (soni report'da)

## 3-qism. O'z-o'zini tekshirish savollari

1. Nega filtr avval metadata bo'yicha, audio dekodlashdan oldin qilinadi?
2. Val va test'ning farqi nima — nega ikkalasi ham kerak?
3. Agar SEED'ni o'zgartirsangiz nima o'zgaradi va nima o'zgarmasligi kerak?
