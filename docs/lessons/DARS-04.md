# 4-dars: Trek B — CNN'ni noldan qurish va o'qitish

## Maqsad

Bu darsdan keyin sizda o'zingiz qurgan, o'zingiz o'qitgan model bo'ladi va siz
chuqur o'rganishning yuragi — **training loop**ni tushunasiz. Bu bilim keyingi
har qanday model uchun ishlaydi.

Kutilgan natija: val aniqligi **95–97%**, trening ~45–75 daqiqa (CPU'da).

---

## 1-qism. Nazariya

### 4.1 Nega aynan CNN?

Spektrogramma — rasm, lekin oddiy rasm emas: o'qlari **mel bandi × vaqt**.
CNN uch sababga ko'ra mos:

- **Lokal naqshlar**: garmonik chiziqlar, formant dog'lari — bularning
  hammasi kichik hududda joylashgan. 3×3 filtr aynan shuni ko'radi.
- **Joy o'zgarishiga chidamlilik**: "ovoz balandligi" naqshi klipning
  boshida ham, oxirida ham bir xil ma'noni bildiradi. CNN filtri butun rasm
  bo'ylab **bir xil og'irliklar** bilan yuradi.
- **Kam parametr**: 3×3 filtrda 9 ta og'irlik, u butun rasmga qo'llanadi.
  To'liq bog'langan qatlam bo'lsa 64×200 = 12 800 ta og'irlik kerak bo'lardi.

### 4.2 Bitta blok nimadan iborat

Bizning har blok: `Conv2d → BatchNorm2d → ReLU → MaxPool2d`.

- **Conv2d(in, out, 3, padding=1)** — `out` ta filtr, har biri 3×3. Har filtr
  o'z "naqsh detektorini" o'rganadi. `padding=1` shaklni saqlaydi.
- **BatchNorm2d** — har kanal chiqishini batch bo'yicha normallashtiradi.
  3-darsda kirishni normallashtirdik; BatchNorm buni **ichkarida, har qatlamda**
  qiladi. Natijada trening ancha barqaror va tez.
- **ReLU** — `max(0, x)`. Chiziqli bo'lmagan element. Busiz butun tarmoq bitta
  matritsaga teng bo'lib qolardi va murakkab naqshlarni o'rgana olmasdi.
- **MaxPool2d(2)** — o'lchamni 2 barobar kichraytiradi, eng kuchli signalni
  saqlaydi. Har blokdan keyin "ko'rish maydoni" kengayadi: chuqur qatlamlar
  kattaroq naqshlarni ko'radi.

Bloklar zanjiri: `(1,64,200) → (16,32,100) → (32,16,50) → (64,8,25) → (64,4,12)`.
Fazoviy o'lcham kichrayadi, kanallar soni ortadi — bu klassik CNN naqshi.

### 4.3 Global Average Pooling — nega Flatten emas?

Oxirgi blokdan keyin `(64, 4, 12)` tensor qoladi. Ikki yo'l bor:

- `Flatten` → 3072 ta son → `Linear(3072, 2)` = **6144 parametr**, va kirish
  o'lchami qat'iy bo'lishi shart.
- `AdaptiveAvgPool2d(1)` → har kanaldan o'rtacha → 64 ta son →
  `Linear(64, 2)` = **130 parametr**, va **istalgan uzunlikdagi** kirish ishlaydi.

Ikkinchisini tanlaymiz. Bu realtime uchun ham muhim: modelga 2 s ham, 5 s ham
berish mumkin bo'ladi.

### 4.4 Training loop — 5 ta qadam

Bu chuqur o'rganishning butun mohiyati. Har batch uchun:

```
1. optimizer.zero_grad()   -> eski gradientlarni tozalash
2. out = model(x)          -> forward: bashorat
3. loss = criterion(out,y) -> xato qanchalik katta?
4. loss.backward()         -> backward: har og'irlik xatoga qanchalik "aybdor"?
5. optimizer.step()        -> og'irliklarni gradient yo'nalishi teskarisiga surish
```

**Nega `zero_grad()` kerak?** PyTorch gradientlarni **to'plab boradi** (qo'shadi),
avtomatik tozalamaydi. Unutsangiz — oldingi batch gradientlari qo'shilib ketadi
va model o'rganmaydi. Bu eng ko'p uchraydigan xato.

### 4.5 Logits va CrossEntropyLoss

Model 2 ta son chiqaradi — **logit**lar (normallashtirilmagan ballar). Ularni
o'zingiz softmax qilmang: `nn.CrossEntropyLoss` softmax'ni **ichida** bajaradi
(shunday qilish sonli jihatdan barqarorroq). Ikki marta softmax qilsangiz
model sekin o'rganadi.

Bashorat olish uchun: `pred = out.argmax(dim=1)`. Ehtimollik kerak bo'lsa
(7-darsdagi ilova uchun): `torch.softmax(out, dim=1)`.

### 4.6 Random crop — bepul augmentatsiya

Keshda har klip 4 sekund. Treningda undan **tasodifiy 2 sekundlik** bo'lak
olamiz. Natijada model har epochda o'sha klipni biroz boshqacha ko'radi —
bu overfitting'ga qarshi kurashadi va bepul.

Baholashda (val/test) esa **markazdan** olamiz — natija takrorlanuvchi bo'lishi
uchun. Baholash hech qachon tasodifiy bo'lmasligi kerak.

### 4.7 `model.train()` va `model.eval()`

Bu ikki rejim BatchNorm va Dropout xatti-harakatini o'zgartiradi:

| | train() | eval() |
|---|---|---|
| BatchNorm | joriy batch statistikasi | trening davomida to'plangan statistika |
| Dropout | neyronlarni tasodifiy o'chiradi | hech narsa o'chirilmaydi |

`eval()` ni unutish — klassik xato: val aniqligi tushib ketadi va sababi
tushunarsiz bo'ladi. Baholashda `torch.no_grad()` ham qo'shiladi — gradient
kerak emas, xotira va vaqt tejaladi.

### 4.8 Overfitting va eng yaxshi checkpoint

Trening davomida train loss doim tushadi. Val aniqligi esa bir joyda to'xtaydi
va keyin **tusha boshlashi** mumkin — model datani yodlashga o'tdi.

Shuning uchun har epochdan keyin val'da baholaymiz va **eng yaxshi val
natijasini bergan** modelni saqlaymiz, oxirgisini emas.

Test'ga hali ham tegmaymiz — u 6-darsgacha muhrlangan.

---

## 2-qism. TOPSHIRIQ

Avval PyTorch (CPU versiyasi, ~200 MB):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`scripts/04_train_cnn.py` da **4 ta TODO**:

| TODO | Nima | Nimani o'rgatadi |
|---|---|---|
| 1 | `MelDataset.__getitem__` | crop (train=tasodifiy, eval=markaz) + normalizatsiya |
| 2 | `GenderCNN` — `__init__` va `forward` | CNN arxitekturasi |
| 3 | `train_one_epoch` | 5 qadamli training loop |
| 4 | `evaluate` | eval rejimi, no_grad, aniqlik |

**Avval tez sinov** (2 daqiqa, kod ishlayaptimi tekshiradi):

```powershell
python scripts\04_train_cnn.py --quick
```

`--quick` rejimda 1500 klip va 2 epoch ishlatiladi. Aniqlik past bo'ladi —
muhimi xato chiqmasligi. Shundan keyin to'liq trening:

```powershell
python scripts\04_train_cnn.py
```

~45–75 daqiqa. Natija: `models/cnn_best.pt`, `models/cnn_history.json`,
`models/cnn_curves.png`.

**Muvaffaqiyat mezonlari**: val aniqligi ≥ 95%, train va val loss egri
chiziqlari bir-biridan juda uzoqlashib ketmagan.

## 3-qism. Savollar

1. `optimizer.zero_grad()` ni unutsangiz nima bo'ladi?
2. Nega baholashda markazdan, treningda tasodifiy crop olamiz?
3. Global Average Pooling Flatten'dan nimasi bilan afzal?
4. Val aniqligi o'smay qolgan, lekin train loss tushishda davom etayotgan
   bo'lsa — bu nima degani?
