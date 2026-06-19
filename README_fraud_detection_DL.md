# 🔐 Finalterm — Deep Learning: Fraud Detection (Klasifikasi)

Repositori ini berisi pengerjaan tugas individu **"Hands-On End-to-End Models Machine Learning and Deep Learning"** untuk mata kuliah Machine Learning & Deep Learning, dengan studi kasus **deteksi fraud pada transaksi online (IEEE-CIS Fraud Detection dataset)**.

## 👤 Identitas

| | |
|---|---|
| **Nama** | Rangga Timotius |
| **NIM** | 101032300137 |
| **Kelas** | TK46GAB |

---

## 📌 Tujuan Repositori

Repositori ini mendemokan pipeline **deep learning klasifikasi biner end-to-end** yang mencakup: loading data memory-efficient, EDA, preprocessing tabular, penanganan class imbalance (SMOTE), perbandingan tiga arsitektur neural network, hyperparameter tuning otomatis (Optuna), experiment tracking (MLflow), evaluasi metrik klasifikasi imbalanced, dan generate file submission.

---

## 🧾 Ringkasan Proyek

**Dataset:** IEEE-CIS Fraud Detection — `train_transaction.csv` (590.540 transaksi, 394 kolom) dan `test_transaction.csv` (506.691 transaksi).

**Target:** `isFraud` — 0 = transaksi normal, 1 = transaksi fraud.

**Karakteristik dataset:**
- **Sangat imbalanced** — hanya **3.50% fraud** (20.663 dari 590.540 transaksi)
- Banyak kolom `V*` (hasil PCA, V1–V339) dan `D*` dengan missing value tinggi (hingga 93.63%)
- Fitur kategorikal (`ProductCD`, `card4`, `card6`, email domain, `M1`–`M9`) perlu di-encode
- Ukuran file besar (train: 756.9 MB raw → 256.2 MB setelah downcasting)

**Pipeline yang dibangun:**
1. Mount Google Drive & setup library
2. Load data memory-efficient — dtype downcasting (float64→float32) + seleksi kolom (168 dari 394)
3. EDA — distribusi target, missing values, distribusi `TransactionAmt`
4. Preprocessing — drop kolom missing >70%, Label Encoding, median imputation, feature engineering (`log_TransactionAmt`)
5. Split train/val (80/20, stratified) → SMOTE untuk menyeimbangkan kelas training
6. StandardScaler — wajib untuk neural network
7. Tiga arsitektur deep learning (baseline)
8. Hyperparameter tuning dengan **Optuna** (10 trial per arsitektur) + tracking **MLflow**
9. Training final dengan parameter terbaik
10. Evaluasi (ROC curve, confusion matrix, classification report)
11. Generate `submission_fraud_dl.csv`

---

## 🧠 Deskripsi Model & Hasil Evaluasi

### Arsitektur yang Dibandingkan

| Arsitektur | Deskripsi |
|---|---|
| **Shallow MLP** | 1 hidden layer + Dropout — baseline paling sederhana |
| **Deep MLP** | 4 hidden layer dengan BatchNormalization + Dropout, ukuran layer mengecil progresif |
| **Deep Residual MLP** | Beberapa residual block (Dense → BN → ReLU → Dropout → Dense + skip connection) untuk stabilitas training jaringan lebih dalam |

Semua arsitektur menggunakan output `sigmoid` (probabilitas fraud) dengan loss `binary_crossentropy`. Class imbalance ditangani dengan **SMOTE** pada training set (hasil: 455.902 vs 455.902 sampel setelah resampling).

---

### Hasil Baseline (sebelum tuning Optuna)

| Model | AUC-ROC | Avg Precision (PR-AUC) | F1-Fraud | Time |
|---|---|---|---|---|
| **Deep Residual MLP** | **0.9180** | **0.5981** | **0.5446** | 58.7s |
| Deep MLP | 0.9075 | 0.5520 | 0.4307 | 87.4s |
| Shallow MLP | 0.9050 | 0.5474 | 0.4324 | 61.1s |

---

### Hyperparameter Terbaik (Optuna, 10 trial per arsitektur)

| Arsitektur | Best val AUC (Optuna) | units | dropout | lr | parameter lain |
|---|---|---|---|---|---|
| **Shallow MLP** | **0.9253** | **256** | **0.3514** | **0.003047** | — |
| Deep MLP | 0.9125 | 128 | 0.1155 | 0.000949 | n_layers=2 |
| Deep Residual MLP | 0.9266 | 256 | 0.3372 | 0.000315 | n_blocks=3 |

---

### Hasil Final (setelah training ulang dengan hyperparameter terbaik)

| Model | AUC-ROC | PR-AUC |
|---|---|---|
| **Shallow MLP (Tuned) ✅ Best** | **0.9317** | **0.6475** |
| Deep Residual MLP (Tuned) | 0.9259 | 0.6500 |
| Deep MLP (Tuned) | 0.9208 | 0.6025 |

**Model terbaik berdasarkan AUC-ROC: Shallow MLP (Tuned)** — AUC **0.9317**, PR-AUC **0.6475**

> *Catatan: Deep Residual MLP (Tuned) unggul tipis di PR-AUC (0.6500 vs 0.6475), metrik yang lebih representatif untuk kelas minoritas.*

---

### Classification Report — Model Terbaik (Shallow MLP Tuned, threshold 0.5)

```
              precision    recall  f1-score   support

      Normal       0.99      0.98      0.98    113,975
       Fraud       0.55      0.63      0.59      4,133

    accuracy                           0.97    118,108
   macro avg       0.77      0.81      0.79    118,108
weighted avg       0.97      0.97      0.97    118,108
```

**Interpretasi:** model berhasil mendeteksi **63% transaksi fraud** (recall 0.63) pada validation set dengan precision 0.55 — artinya dari setiap 100 transaksi yang ditandai fraud, 55 di antaranya benar-benar fraud. Untuk kasus fraud detection, recall yang tinggi biasanya lebih diprioritaskan (fraud yang lolos lebih merugikan daripada false alarm).

---

## 🗂️ How to Navigate

```
finalterm-machine-learning/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── Finalterm_fraud_detection_DL_FIX.ipynb   ← notebook utama dengan output lengkap
```

Seluruh alur (EDA → preprocessing → 3 arsitektur DL → Optuna → MLflow → evaluasi → submission) ada dalam **satu notebook** dengan penjelasan markdown di setiap tahap.

### Cara Menjalankan

1. Clone repo: `git clone <url-repo-ini>`
2. Install dependencies: `pip install -r requirements.txt`
3. Pastikan `train_transaction.csv` dan `test_transaction.csv` tersedia di Google Drive (sesuaikan `DRIVE_PATH` di notebook)
4. Jalankan notebook dari atas ke bawah di **Google Colab** (disarankan GPU runtime)
5. Untuk melihat dashboard MLflow: `mlflow ui` → buka `http://localhost:5000`

---

## 📊 Metrik Evaluasi yang Digunakan

| Metrik | Alasan digunakan |
|---|---|
| **AUC-ROC** | Mengukur kemampuan model membedakan fraud vs normal secara keseluruhan, threshold-agnostic |
| **PR-AUC (Average Precision)** | Lebih representatif untuk kelas sangat minoritas (3.5% fraud) — tidak terpengaruh dominasi kelas normal |
| **F1-Fraud** | Harmonic mean precision & recall untuk kelas fraud |
| **Precision & Recall** | Trade-off penting: recall rendah = banyak fraud lolos; precision rendah = banyak normal salah ditandai |

> Akurasi **tidak digunakan** sebagai metrik utama karena model yang selalu memprediksi "normal" pun akan mencapai akurasi 96.5% — angka yang menyesatkan untuk dataset seimbalanced ini.

---

## 💡 Catatan Teknis

| Teknik | Dampak |
|---|---|
| `dtype downcasting` float64→float32 | Train: 756.9 MB → 256.2 MB (hemat 66.1%); Test: 645.6 MB → 221.3 MB (hemat 65.7%) |
| `usecols` — load 168 kolom dari 394 | Hemat RAM signifikan saat loading |
| Drop kolom missing >70% | 168 → 106 kolom fitur final |
| `del train` + `gc.collect()` | Bebaskan RAM segera setelah tidak dipakai |
| SMOTE (hanya pada training set) | Menyeimbangkan 20.663 fraud vs 455.902 normal → 455.902 vs 455.902 |
| `StandardScaler` | Wajib untuk konvergensi neural network yang stabil |
| Optuna (10 trial per arsitektur) | Tuning otomatis units, dropout, learning rate (+ n_layers / n_blocks) |
| MLflow | Logging seluruh parameter & metrik tiap trial + model final |

---

## 🔍 Insight Utama

- **Shallow MLP setelah tuning (AUC 0.9317) justru mengungguli Deep MLP maupun Deep Residual MLP** — menunjukkan bahwa untuk data tabular imbalanced seperti ini, arsitektur yang lebih sederhana dengan hyperparameter yang tepat bisa mengalahkan jaringan yang lebih dalam.
- **Peningkatan AUC sangat signifikan setelah tuning Optuna** pada Shallow MLP: 0.9050 (baseline) → 0.9317 (tuned), kenaikan +0.0267.
- **PR-AUC Deep Residual MLP (Tuned) sedikit lebih tinggi** (0.6500 vs 0.6475) — jika prioritas utama adalah mendeteksi fraud dengan presisi tinggi pada kelas minoritas, Deep Residual MLP bisa jadi pilihan alternatif.
- **Recall fraud 63%** pada threshold 0.5 berarti ~37% fraud di validation set masih lolos — threshold dapat diturunkan (mis. 0.3) untuk meningkatkan recall dengan konsekuensi precision yang turun.
- Model memprediksi hanya **1.719 transaksi (0.34%)** sebagai fraud pada test set — lebih rendah dari fraud rate training (3.50%), kemungkinan karena karakteristik test set dari periode waktu yang berbeda.
