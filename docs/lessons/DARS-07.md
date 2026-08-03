# 7-dars: Gradio ilova — fayl yuklash va realtime mikrofon

## Maqsad

Loyihaning yakuni: ishlaydigan ilova. Ikki rejim — audio fayl yuklash va
mikrofondan jonli oqim. Va yo'l-yo'lakay production'ga xos uch masalani hal
qilamiz: VAD, sliding window agregatsiyasi, va ishonch chegarasi.

---

## 1-qism. Nazariya

### 7.1 Qaysi model qayerda — 6-dars javob berdi

| | CNN | wav2vec2 |
|---|---|---|
| Aniqlik (test) | 98.10% | 98.75% |
| Kechikish | 4.02 ms | 504 ms |
| Hajm | 0.2 MB | 360.8 MB |

Realtime uchun hisob-kitob oddiy. 2 sekundlik oyna, har 0.5 sekundda yangi
bashorat kerak — ya'ni sekundiga 2 ta chaqiruv:

- CNN: 2 × ~7 ms (mel + inference) = **14 ms/sek** → protsessorning ~1.4% i
- wav2vec2: 2 × 504 ms = **1008 ms/sek** → 100% dan ko'p, imkonsiz

Shuning uchun **gibrid**: realtime → CNN, fayl yuklash → wav2vec2 (u yerda
yarim sekund kechikish sezilmaydi, aniqlik esa muhim).

Bu — muhandislikning tipik javobi: "qaysi model yaxshi?" degan savolning
javobi kontekstga bog'liq.

### 7.2 VAD — nega majburiy

3-darsdagi spektrogramma rasmini eslang: klip boshida qora zona — sukunat.
Agar oynaga faqat sukunat yoki shovqin tushsa, model baribir "Ayol" yoki
"Erkak" deydi — u "bilmayman" deyishni o'rganmagan. Natijada realtime
chiqish siz jim turganingizda ham sakrab turadi.

**VAD (Voice Activity Detection)** — oynada nutq bormi yoki yo'qmi degan
savolga javob beradi. Eng sodda usul — energiya:

```
rms    = sqrt(mean(y²))            # o'rtacha kvadratik amplituda
rms_db = 20 * log10(rms + 1e-10)   # dB shkalasiga
nutq bor  <=>  rms_db > chegara     (masalan -45 dB)
```

Bu usul tinch xonada yaxshi ishlaydi, shovqinli joyda esa aldaydi (shovqin
ham energiyali). Production uchun **Silero VAD** kabi neyron VAD ishlatiladi —
u nutqni shovqindan farqlaydi. Biz energiya bilan boshlaymiz: g'oyani
tushunish uchun yetarli, va bog'liqlik qo'shmaydi.

### 7.3 Sliding window va agregatsiya

Uzun audio (masalan 10 sekundlik fayl) modelga bir bo'lak sifatida berilmaydi —
model 2 sekundga o'qitilgan. Uni **ustma-ust tushuvchi oynalarga** bo'lamiz:

```
|--- 2s ---|
     |--- 2s ---|
          |--- 2s ---|
     ^ hop = 1s
```

Har oyna alohida bashorat beradi, keyin ularni **birlashtiramiz**. Eng oddiy
va samarali usul — ehtimolliklarning o'rtachasi:

```
p_erkak = mean([p1, p2, ..., pn])     # faqat nutqli oynalar bo'yicha
```

Nega o'rtacha, ovoz berish (majority vote) emas? Chunki o'rtacha
**ishonchni** hisobga oladi: 0.51 bergan oyna 0.99 bergan oynadan kamroq
"ovoz" berishi to'g'ri. Bu — kichik ansambl, va u bitta oynaga qaraganda
aniqroq ishlaydi (xatolar bir-birini yuvadi).

### 7.4 Ishonch chegarasi va "aniq emas" javobi

Model har doim javob beradi, hatto 50.4% ishonch bilan ham. Foydalanuvchiga
bunday javobni "Erkak" deb ko'rsatish — yolg'on aniqlik.

Yechim: chegara qo'yish. `0.35 < p < 0.65` bo'lsa — "aniq emas" deb ayting.
6-darsda ko'rdik: `3d3fca02` spikerida modellar aynan shu zonada ikkilangan
edi va haq edi — u yerda rostdan ham ikki xil ovoz bor edi.

**Ishonchni tan olish — sifat belgisi, kamchilik emas.**

### 7.5 Realtime oqim: aylanma bufer

Mikrofon audioni kichik bo'laklar bilan beradi (masalan 100 ms). Bizga esa
2 sekundlik oyna kerak. Yechim — **aylanma bufer**: yangi bo'lak kelganda
uni oxiriga qo'shamiz, boshidan ortiqchasini kesamiz:

```python
bufer = np.concatenate([bufer, yangi_bolak])[-WINDOW_LEN:]
```

Har yangi bo'lakda bashorat qilish shart emas — bu isrof. Har ~0.5 sekundda
bir marta yetarli.

### 7.6 Gradio

Gradio — bir necha qator kod bilan veb-interfeys yasaydigan kutubxona.
Bizga ikki komponent kerak:

- `gr.Audio(sources=["upload", "microphone"])` — fayl yuklash yoki yozib olish
- `gr.Audio(sources=["microphone"], streaming=True)` — jonli oqim; handler
  har bo'lakda chaqiriladi va `gr.State` orqali bufer saqlanadi

---

## 2-qism. TOPSHIRIQ

```powershell
pip install gradio
```

`scripts/07_app.py` da **4 ta TODO**:

| TODO | Funksiya | Nimani o'rgatadi |
|---|---|---|
| 1 | `is_speech` | energiya asosidagi VAD |
| 2 | `wav_to_windows` | sliding window + mel + normalizatsiya |
| 3 | `predict_cnn_windows` | ko'p oynali bashorat |
| 4 | `aggregate` | o'rtachalash + ishonch chegarasi |

Ishga tushirish:

```powershell
python scripts\07_app.py
```

Brauzerda `http://127.0.0.1:7860` ochiladi.

**Sinash rejasi** (shunchaki ishlashini emas, TO'G'RI ishlashini tekshiring):

1. `data/raw/sample_Ayol.wav` va `sample_Erkak.wav` ni yuklang — 1-darsdan
   qolgan fayllar. Ikkalasi ham to'g'ri topilishi kerak
2. Mikrofonga gapiring — o'z jinsingizni to'g'ri aniqlaydimi?
3. **Jim turing** — "nutq topilmadi" chiqishi kerak, tasodifiy javob emas
4. Pichirlab gapiring, keyin baland gapiring — natija o'zgaradimi?
5. Musiqa qo'ying — energiya VAD'i uni "nutq" deb qabul qiladimi? (ha, qiladi —
   7.2-bo'limdagi cheklov shu)

## 3-qism. Yakuniy savollar

1. Nega realtime'da wav2vec2 ishlatib bo'lmaydi — raqam bilan asoslang.
2. VAD bo'lmasa foydalanuvchi nimani ko'radi?
3. Nega oynalarni majority vote emas, ehtimollik o'rtachasi bilan birlashtiramiz?
4. Loyihaning boshidan oxirigacha eng katta xato qaysi edi va uni nima ochib berdi?
