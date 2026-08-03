# 8-dars: Deploy — HuggingFace Space va Telegram bot

## Maqsad

Modelni o'z kompyuteringizdan chiqarib, doimiy ishlaydigan ikki xizmatga
aylantirish. Va deploy'ning uch asosiy tamoyilini o'rganish: **paritetni
tekshirish**, **bog'liqliklarni kesish**, **muhitni ajratish**.

---

## 1-qism. Nazariya

### 8.1 Training kodi deploy kodi emas

7-darsdagi ilova ishladi, lekin uni shundayligicha serverga qo'yib bo'lmaydi:

| Muammo | Nega yomon | Yechim |
|---|---|---|
| `importlib` bilan `04_train_cnn.py` dan model tortish | trening skripti serverga chiqmasligi kerak | `model.py` — alohida, mustaqil |
| `librosa` (numba + scipy, ~200 MB) | image kattalashadi, build sekinlashadi | `torch.stft` + oldindan hisoblangan filtrbank |
| Gradio'ning audio yuklashi | Telegram OGG/Opus yuboradi, gradio buni bilmaydi | `ffmpeg` bilan dekodlash |
| Yo'llar qat'iy (`data/processed/...`) | serverda bunday papka yo'q | faqat `cnn_best.pt` va `mel_fb.npy` |

Deploy — kodni ko'chirish emas, **qayta yig'ish**.

### 8.2 Paritet tekshiruvi — eng muhim qadam

Biz `librosa` ni `torch.stft` ga almashtirdik. Agar ular biroz boshqacha
natija bersa, model **jimgina** yomonlashadi: xato chiqmaydi, faqat aniqlik
tushadi va siz buni bilmaysiz.

Shuning uchun `08_export.py` ikki tekshiruvni **majburiy** bajaradi:

1. **Mel pariteti**: 60 ta test faylida `librosa` va `torch` natijalari
   solishtiriladi. Farq 0.05 dB dan katta bo'lsa — skript to'xtaydi.
2. **Uchdan-uchgacha paritet**: deploy modeli va 4-darsdagi model bir xil
   ehtimollik beradimi (farq < 0.001).

Bu — production ML'ning oltin qoidasi: **optimizatsiyadan keyin har doim
eski va yangi yo'lni solishtiring.** Tezlik uchun aniqlikni jimgina
qurbon qilish — eng ko'p uchraydigan deploy xatosi.

### 8.3 Ikki muhit, ikki tanlov

| | HuggingFace Space | Railway |
|---|---|---|
| Nima | Gradio veb-ilova | Telegram bot |
| Narx | bepul | ~$5/oy kredit |
| Uxlaydimi | ha, 48 soatdan keyin | yo'q |
| HTTPS | avtomatik (mikrofon ishlaydi) | kerak emas (polling) |
| ffmpeg | shart emas | **shart** (OGG/Opus) |

**Nega faqat CNN, wav2vec2 emas?** 6-darsdagi o'lchov: wav2vec2 +0.65 foiz punkt
aniqlik beradi, lekin 360 MB model + `transformers` kutubxonasi + har so'rovda
~500 MB RAM va 504 ms. Railway sarflangan resursga qarab hisoblaydi. Bu
almashuv foydali emas.

### 8.4 Webhook emas, polling

Telegram bot ikki xil ishlashi mumkin:

- **Webhook**: Telegram sizning serveringizga so'rov yuboradi. Tez, lekin
  public HTTPS manzil va sertifikat kerak.
- **Polling**: bot o'zi Telegram'dan "yangilik bormi?" deb so'rab turadi.
  Sekinroq (~1 s kechikish), lekin **hech qanday tashqi manzil kerak emas**.

Biz polling tanladik: Railway'da port ochish, domen sozlash shart emas.
Bitta muhim cheklov: **bir token bilan bir vaqtda faqat bitta polling**.
Lokalda ham, Railway'da ham ishga tushirsangiz — `409 Conflict`.

### 8.5 Maxfiy ma'lumot kodda turmaydi

Bot tokeni — parol. U hech qachon kodga yozilmaydi va git'ga tushmaydi.
Yo'l: **environment variable**. Railway'da Variables bo'limiga qo'yasiz,
kod esa `os.environ["TELEGRAM_BOT_TOKEN"]` orqali o'qiydi.

Token oshkor bo'lsa — darhol BotFather'da `/revoke` qiling.

### 8.6 Foydalanuvchiga halol javob

Bot ishlatadigan odam ML bilmaydi. Shuning uchun:

- Ishonchni **foizda** ko'rsating, `p(Erkak) = 0.87` deb emas
- "Aniq emas" holatini yashirmang va **sababini** ayting
- Xato bo'lsa stack trace emas, **nima qilish kerakligini** ayting
- Modelning cheklovini oshkor qiling (o'zbek nutqi, taxminiy natija)

---

## 2-qism. TOPSHIRIQ

`deploy/bot.py` da **2 ta TODO**:

| TODO | Nima | Nimani o'rgatadi |
|---|---|---|
| 1 | `format_result` | natijani odamga tushunarli qilib berish (8.6) |
| 2 | `handle_audio` | async handler: yuklash → tahlil → javob, xatolar bilan |

Qolgani tayyor: `model.py`, `app.py`, `Dockerfile`, `08_export.py`.

### Qadamlar

**1. Artefaktlarni yig'ing va tekshiring:**

```powershell
python scripts\08_export.py
```

Ikkala paritet tekshiruvi o'tishi shart. O'tmasa — deploy qilmang.

**2. Lokal sinov** (ixtiyoriy, `ffmpeg` bo'lsa):

```powershell
$env:TELEGRAM_BOT_TOKEN="BotFather bergan token"
python deploy\build\railway\bot.py
```

**3. Deploy:** `deploy/README.md` dagi A va B bo'limlari.

### Sinov rejasi (ikkala xizmatda ham)

1. Ovozli xabar / audio fayl — to'g'ri javob beradimi?
2. **Jim yozib yuboring** — "nutq topilmadi" chiqadimi?
3. 1 sekundlik juda qisqa yozuv — tushunarli xabar beradimi?
4. Rasm yoki matn yuboring — bot qulaydimi yoki xushmuomala javob beradimi?
5. Uzun (2 daqiqadan ortiq) fayl — kesiladimi yoki osilib qoladimi?

## 3-qism. Yakuniy savollar

1. Nega `librosa` ni almashtirgandan keyin paritet tekshiruvi majburiy?
2. Polling va webhook farqi — bizning holatda qaysi biri va nega?
3. Nega bot tokeni kodga yozilmaydi?
4. Bir token bilan ikkita polling ishga tushsa nima bo'ladi?
