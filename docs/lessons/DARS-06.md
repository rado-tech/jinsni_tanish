# 6-dars: Halol taqqoslash — test to'plami, metrikalar, kesimlar, tezlik

## Maqsad

Ikki modelni **muhrlangan test to'plamida** birinchi va oxirgi marta yuzlashtirish.
Va eng muhimi: "qaysi model yaxshi?" degan savolga aniqlik raqamidan kengroq
javob berish — tezlik, hajm, kesimlar va statistik ishonch bilan.

---

## 1-qism. Nazariya

### 6.1 Test — bir marta ochiladigan sandiq

2-darsdan beri test splitiga tegmadik. Sabab: unga qarab **birorta qaror**
qilinmagan bo'lishi kerak. Val'ni ko'rib LR tanladik, epoch sonini tanladik,
checkpoint tanladik — shuning uchun val allaqachon "ifloslangan", u optimistik
baho beradi.

Test esa toza. Bugungi raqam — modelning yangi foydalanuvchi oldidagi haqiqiy
ko'rsatkichi. **Test natijasini ko'rgandan keyin modelni sozlamaslik kerak** —
sozlasangiz, test ham val'ga aylanadi.

### 6.2 Accuracy yolg'on gapirishi mumkin

Aniqlik — bitta son, u ko'p narsani yashiradi. **Confusion matrix** to'liq
manzarani beradi:

|  | bashorat: Ayol | bashorat: Erkak |
|---|---|---|
| **haqiqat: Ayol** | TN | FP |
| **haqiqat: Erkak** | FN | TP |

("Erkak"ni musbat sinf deb olsak.) Bundan uch metrika chiqadi:

- **Precision** = TP/(TP+FP) — "Erkak" degan bashoratlarning nechtasi to'g'ri?
- **Recall** = TP/(TP+FN) — haqiqiy erkaklarning nechtasini topdik?
- **F1** = 2·P·R/(P+R) — ikkalasining garmonik o'rtachasi

Bizning data 50/50 balanslangan, shuning uchun accuracy ham ishonchli. Lekin
90/10 balansda accuracy aldaydi: har doim "ko'pchilik sinf" desangiz 90%
bo'ladi, model esa hech narsa o'rganmagan.

### 6.3 Xato darajasi bilan o'ylang

98.50% va 99.67% — farqi atigi 1.17 foiz punkti, arzimasdek. Lekin **xato
darajasi**ga qarang: 1.50% va 0.33%. Bu **4.5 barobar** farq.

Yuqori aniqliklar zonasida har doim shunday hisoblang. 99% dan 99.9% ga
o'tish "0.9% yaxshilanish" emas — bu xatolarning **10 barobar** kamayishi.

### 6.4 Kesimlar (slices) — o'rtacha yashiradi

Umumiy 99% aniqlik ichida "12–17 yosh guruhida 85%" yashiringan bo'lishi mumkin.
Foydalanuvchi esa o'rtachada yashamaydi — u aniq bir guruhda.

Biz uch kesimni tekshiramiz:
- **Klip uzunligi** — qisqa kliplarda model qanchalik ishonchli? (realtime uchun kritik)
- **Yosh guruhi** — 4-darsdagi gipoteza: 12–17 yoshli o'g'il bolalar
- **Viloyat** — biror aksent guruhida tizimli xato bormi?

Kichik guruhda past aniqlik ko'rsangiz, avval **guruh hajmini** qarang: 20 ta
namunada 85% — bu 3 ta xato, tasodif bo'lishi mumkin.

### 6.5 Farq haqiqiymi? McNemar testi

Ikki model **bir xil** test to'plamida ishladi — demak namunalar juftlashgan.
Bunday holda oddiy taqqoslash emas, **McNemar testi** ishlatiladi. U faqat
**kelishmovchiliklarni** sanaydi:

- `b` = CNN to'g'ri, w2v2 xato
- `c` = CNN xato, w2v2 to'g'ri

Ikkalasi ham to'g'ri yoki ikkalasi ham xato bo'lgan holatlar hech narsa
demaydi — ular tashlanadi. Agar `b` va `c` taxminan teng bo'lsa, modellar
statistik jihatdan farq qilmaydi. `c >> b` bo'lsa — w2v2 haqiqatan yaxshiroq.

Bu muhim: 3009 namunada tasodifiy tebranish ~±0.2% bor. Ikki modelning
0.1% farqi hech narsani anglatmaydi, 1.2% farqi esa — anglatishi mumkin.
McNemar shu farqni aniqlaydi.

### 6.6 Production tanlovi: uchburchak

Model tanlashda uch o'lcham bor va ular bir-biriga qarshi:

```
        aniqlik
          /\
         /  \
   tezlik --- hajm
```

Realtime ilova uchun **latency** (kechikish) aniqlikdan muhimroq bo'lishi
mumkin: 99.7% aniq, lekin 800 ms kechikadigan model foydalanuvchiga
98.5% aniq, 5 ms kechikadigan modeldan yomonroq tuyuladi.

Shuning uchun bugun **CPU'da o'lchaymiz** — aynan foydalanuvchining
kompyuterida qanday ishlashini.

### 6.7 To'g'ri o'lchash: warmup va median

Tezlikni o'lchaganda ikki qoida:

- **Warmup**: birinchi chaqiruv sekin (xotira ajratish, kesh isishi, lazy init).
  Uni hisobga olmang — 1-darsda mel benchmark'da aynan shu 107 ms vs 6 ms
  farqni bergan edi.
- **Median, o'rtacha emas**: OS boshqa ishlar bilan band bo'lganda bitta
  o'lchov 10 barobar sekin chiqishi mumkin. Median bunga chidamli.

---

## 2-qism. TOPSHIRIQ

### A. Colab qismi (wav2vec2 test bashoratlari)

Colab'da (audio hali `/content/audio` da bo'lsa; bo'lmasa avval `05a_fetch_audio.py`):

```bash
!python 06a_predict_w2v2.py
```

Bu `w2v2_test_preds.csv` yaratadi (~200 KB). Uni yuklab olib, loyihangizdagi
`models/` papkasiga qo'ying.

Va tezlik o'lchovi uchun modelning o'zini ham oling: Drive'dagi `w2v2_best`
papkasini `models/w2v2_best/` ga yuklab oling (~380 MB). Ixtiyoriy —
bo'lmasa skript tezlik qismini o'tkazib yuboradi.

### B. Lokal qism

`scripts/06_compare.py` da **4 ta TODO**:

| TODO | Funksiya | Nimani o'rgatadi |
|---|---|---|
| 1 | `predict_cnn` | saqlangan checkpoint'ni yuklab, test'da bashorat |
| 2 | `compute_metrics` | confusion matrix, precision/recall/F1 — qo'lda |
| 3 | `slice_report` | guruhlar kesimida aniqlik |
| 4 | `benchmark_latency` | warmup + median bilan tezlik o'lchash |

```powershell
python scripts\06_compare.py
```

Natija: `models/comparison.json`, `models/comparison.png`, va konsolda to'liq hisobot.

## 3-qism. Savollar

1. Nega test'ni ko'rgandan keyin modelni sozlash mumkin emas?
2. 99.0% va 99.5% — xato darajasida bu qanchalik farq?
3. McNemar testi nima uchun oddiy taqqoslashdan afzal?
4. Kichik guruhda past aniqlik ko'rsangiz, birinchi navbatda nimani tekshirasiz?
5. Natijalarga qarab: 7-darsdagi realtime ilovada qaysi modelni ishlatamiz va nega?
