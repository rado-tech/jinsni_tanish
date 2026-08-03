# Ovozdan jins aniqlash (o'zbek nutqi)

Spiker jinsini ovozidan aniqlaydi — audio fayldan yoki jonli mikrofon oqimidan.

60 706 parametrli CNN speaker-disjoint test to'plamida **98.1%** aniqlik beradi
va noutbuk protsessorida **4 ms** da ishlaydi. Taqqoslash uchun fine-tune
qilingan wav2vec2 (94.6M parametr) 98.8% beradi, lekin bitta bashorat uchun
504 ms sarflaydi.

*English: [README.md](README.md)*

---

## Natijalar

Test to'plami — 3 049 klip, 364 spiker. Bu spikerlar treningda ham,
validatsiyada ham qatnashmagan.

| | CNN (noldan) | wav2vec2-base (fine-tune) |
|---|---|---|
| Aniqlik | 98.10% | **98.75%** |
| Xato | 58 | 38 |
| CPU kechikish (2 s audio) | **4.0 ms** | 504 ms |
| Model hajmi | **0.2 MB** | 361 MB |
| Parametrlar | **60 706** | 94 569 090 |
| Trening | 55 daq, 4 yadro CPU | 15 daq, T4 GPU |

McNemar testi: χ² = 16.41 (chegara 3.84) — farq statistik jihatdan haqiqiy.
Lekin CNN 126 barobar tez va 1480 barobar kichik, shuning uchun realtime uchun
aynan u ishlatiladi.

To'liq tahlil: [docs/RESULTS.md](docs/RESULTS.md).

## Qanday ishlaydi

```
audio ──> 16 kHz mono ──> 2 s oynalar (0.5 s qadam) ──> log-mel (64 × 200)
                                    │
                                    ├─ energiya VAD sukunatli oynalarni chiqaradi
                                    ▼
                          CNN ──> har oyna uchun p(erkak)
                                    │
                                    ▼
                    o'rtacha ──> Ayol / Erkak / Aniq emas
```

To'rtta konvolyutsion blok (16→32→64→64 kanal), har biri `Conv2d → BatchNorm →
ReLU → MaxPool`, keyin global average pooling va chiziqli qatlam.

O'rtacha ehtimollik 0.35–0.65 oralig'iga tushsa, model **"aniq emas"** deydi.
Sinovda bu chegara datasetdagi bir akkauntni to'g'ri belgiladi — o'sha akkauntda
haqiqatan ikki xil odamning ovozi bor edi.

## Tez boshlash

```bash
git clone <repository-url>
cd rado-gender_classification
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

```python
from genderid import GenderClassifier

clf = GenderClassifier("models/cnn_best.pt")
natija = clf.predict_file("sample.wav")
print(natija.label, natija.confidence)
```

Veb-interfeys:

```bash
python -m app.gradio_app
```

## Modelni qayta o'qitish

Skriptlar ishga tushirish tartibida raqamlangan. 5–7 qadamlar GPU talab qiladi
va faqat transformer taqqoslamasi uchun kerak.

| Qadam | Skript | Vazifasi | Vaqt |
|---|---|---|---|
| 1 | `01_explore.py` | Dataset sxemasi va balansini tekshirish | 1 daq |
| 2 | `02_prepare.py` | Filtr, spiker cheklovi, speaker-disjoint split | 17 daq |
| 3 | `03_features.py` | Log-mel keshini yasash | 6 daq |
| 4 | `04_train_cnn.py` | CNN treningi | 55 daq (CPU) |
| 5 | `05_fetch_audio.py` | GPU mashinasiga o'sha kliplarni tortish | 8 daq |
| 6 | `06_finetune_w2v2.py` | wav2vec2 fine-tuning | 15 daq (T4) |
| 7 | `07_predict_w2v2.py` | Test bashoratlari | 2 daq |
| 8 | `08_compare.py` | Metrikalar, kesimlar, tezlik | 2 daq |
| 9 | `09_export.py` | Paritet tekshiruvi va deploy paketlari | 1 daq |

Har bir skriptda `--help` bor. Trening skriptlarida `--quick` rejimi bor —
to'liq ishga tushirishdan oldin ikki daqiqada kodni sinab ko'rish uchun.

## Dataset

[DavronSherbaev/uzbekvoice-filtered](https://huggingface.co/datasets/DavronSherbaev/uzbekvoice-filtered)
— crowdsourced o'zbek nutqi, Apache 2.0, ~500K klip; spiker, jins, yosh va
viloyat metadatasi bilan.

Bu loyihada 30 000 klip (34.8 soat), 3 505 spiker ishlatilgan. Uchta qoida hajmdan
muhimroq:

- **Metadata bo'yicha sifat filtri** — audio ochilmasdan oldin qo'llanadi.
- **Har spikerdan ko'pi bilan 40 klip.** Xom datada top-5 spiker namunaning
  23% ini egallagan edi; cheklovdan keyin 0.7%.
- **Speaker-disjoint split.** Bir odamning barcha kliplari bitta splitda
  qoladi — aks holda model ovozni yodlaydi va test natijasi yolg'on chiqadi.

Batafsil: [docs/DATASET.md](docs/DATASET.md).

## Deploy

`09_export.py` ikkita mustaqil paket yasaydi va paritet tekshiruvlari
o'tmasa ishni to'xtatadi.

- **Hugging Face Space** — Gradio interfeysi, bepul CPU, HTTPS (mikrofon ishlaydi)
- **Railway / Docker** — Telegram bot va veb-interfeys bitta konteynerda

Qadamlar: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Loyiha tuzilishi

```
genderid/          Umumiy paket: config, model, features, inference
scripts/           Raqamlangan pipeline: xom datasetdan deploy paketigacha
app/               Gradio interfeysi, Telegram bot, Dockerfile
docs/              Dataset, natijalar, deploy, model kartasi
docs/lessons/      Loyiha qanday qurilgani — bosqichma-bosqich darslar
```

## Cheklovlar

- Model o'zbek nutqida o'qitilgan; boshqa tillarda sifat tekshirilmagan.
- Jins ikkilik (Ayol/Erkak) sifatida modellashtirilgan, chunki dataset shunday
  yorliqlangan. Bu — datasetning tuzilishi, inson identifikatsiyasining to'liq
  tasnifi emas.
- Energiya asosidagi VAD baland shovqinni nutq deb qabul qiladi.
- Trening datasi yosh tomonga og'gan: spikerlarning ~75% i 18–24 yoshda.
- Natijalar taxminiy. Shaxsga oid muhim qarorlar uchun ishlatilmasin.

## Litsenziya

Apache 2.0 — [LICENSE](LICENSE).
