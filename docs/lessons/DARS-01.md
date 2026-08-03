# 1-dars: Raqamli audio asoslari va dataset bilan tanishish

## Maqsad

Bu darsdan keyin siz: (1) audio kompyuterda qanday saqlanishini, (2) nega jinsni ovozdan aniqlash mumkinligini, (3) 20 GB dataset'ni to'liq yuklamasdan qanday o'rganishni bilasiz — va buni kod bilan isbotlaysiz.

---

## 1-qism. Nazariya

### 1.1 Ovoz qanday raqamga aylanadi

Mikrofon havo tebranishini elektr signalga aylantiradi. Kompyuter bu uzluksiz signalni sekundiga N marta "rasmga oladi" (o'lchaydi). Har bir o'lchov — bitta son (odatda -1.0 … +1.0 oralig'ida float).

- **Sampling rate (diskretlash chastotasi)** — sekundiga necha o'lchov. 16 000 Hz (16 kHz) degani: 1 sekund audio = 16 000 ta son (bu massiv **waveform** deyiladi).
- **Nyquist qoidasi**: sampling rate F bo'lsa, signalda maksimum F/2 chastotani saqlay olamiz. Inson nutqining ma'noli qismi ~8 kHz gacha — shuning uchun nutq uchun 16 kHz standart (8000 × 2). Musiqa uchun 44.1 kHz ishlatiladi.
- **Mono vs stereo**: nutq tahlilida bitta kanal (mono) yetadi.
- Bizning dataset klipilari katta ehtimol 32–48 kHz'da saqlangan — biz ularni keyingi darsda 16 kHz'ga **resample** qilamiz (ortiqcha chastotalarni tashlab, hajmni 3 barobar kichraytiramiz, ma'lumot deyarli yo'qolmaydi).

### 1.2 Nega jinsni ovozdan aniqlash mumkin?

Fiziologiya: erkaklarda ovoz paychalari uzunroq va qalinroq → sekinroq tebranadi.

- **F0 (fundamental chastota, "pitch")**: erkaklarda odatda ~85–155 Hz, ayollarda ~165–255 Hz. Bu eng kuchli signal, lekin oraliqlar ustma-ust tushadi — faqat F0 bilan ~90-95% dan nariga o'tib bo'lmaydi.
- **Formantlar** — og'iz-tomoq bo'shlig'ining rezonans chastotalari (traktning uzunligiga bog'liq, erkaklarda past). 
- ML model bu belgilarni qo'lda bermaymiz — model spektrogrammadan (3-dars) o'zi topadi. Lekin *nimani* topayotganini bilish — debugging paytida juda muhim.

### 1.3 Dataset qanday saqlangan va streaming nima

`uzbekvoice-filtered` HuggingFace'da **28 ta parquet fayl**da yotadi (~500K qator, taxminan 15–20 GB). Parquet — ustunli format: har qatorda audio baytlari + metadata (gender, client_id, ...).

To'liq yuklab olish shart emas: `load_dataset(..., streaming=True)` fayllarni **oqim sifatida** o'qiydi — qatorlar ketma-ket internetdan keladi, diskka to'liq tushmaydi. Kamchiligi: tasodifiy indeksga sakrab bo'lmaydi, faqat boshidan ketma-ket o'qiladi. Bugungi tanishuv uchun birinchi ~2000 qator yetadi.

### 1.4 Muhim tushuncha: speaker leakage (oldindan bilib qo'ying)

Bitta odam datasetda yuzlab klip yozgan. Agar train va test'ga bir odamning kliplari tushsa — model "jins belgilari"ni emas, o'sha odamning ovozini eslab qoladi, test aniqligi yolg'on yuqori chiqadi. Bunga **speaker leakage** deyiladi. Yechim (2-darsda): split'ni klip bo'yicha emas, **client_id (spiker) bo'yicha** qilamiz. Bugungi topshiriqda unique spikerlarni sanashimiz bejiz emas — split o'shanga quriladi.

---

## 2-qism. TOPSHIRIQ

`scripts/01_explore.py` faylini oching. Skelet tayyor, ichida **4 ta TODO** bor — faqat o'sha joylarni yozasiz:

| TODO | Vazifa | Nimani o'rgatadi |
|---|---|---|
| 1 | Har bir gender qiymatini sanash (`Counter`) | dataset balansini tekshirish |
| 2 | Unique `client_id` soni + eng ko'p klip yozgan 5 spiker | speaker leakage xavfini his qilish |
| 3 | Har klip davomiyligini hisoblash: `len(waveform) / sampling_rate` | waveform ↔ sekund bog'liqligi |
| 4 | 1 ta erkak va 1 ta ayol klipni `data/raw/` ga WAV qilib saqlash | audio massiv bilan amaliy ishlash |

Ishga tushirish:

```powershell
.venv\Scripts\Activate.ps1
python scripts\01_explore.py
```

Skript oxirida `data/explore_report.json`, `data/duration_hist.png` va 2 ta WAV chiqishi kerak. WAV'larni o'zingiz eshitib ko'ring!

**Qoida**: tayyor yechimni internetdan ko'chirmang. Hint kerak bo'lsa — so'rang, men yo'naltiraman (lekin yozib bermayman). Tugagach "tayyor" deng — men natijalarni tekshirib, feedback beraman.

## 3-qism. O'z-o'zini tekshirish savollari (javobni tayyorlang, og'zaki so'rayman)

1. 16 kHz'da 3.5 sekundlik klip necha ta sondan iborat?
2. Nega test aniqligi uchun train va test'da bir xil spiker bo'lishi xavfli?
3. 48 kHz → 16 kHz resample qilganda qaysi chastotalar yo'qoladi? Nutq uchun bu muammomi?
