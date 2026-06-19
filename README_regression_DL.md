# 🎵 Finalterm — Deep Learning: Prediksi Tahun Rilis Lagu (Regresi)

Repositori ini berisi pengerjaan tugas individu **"Hands-On End-to-End Models Machine Learning and Deep Learning"** untuk mata kuliah Deep Learning, dengan studi kasus **regresi: memprediksi tahun rilis lagu dari fitur audio**.

## 👤 Identitas

| | |
|---|---|
| **Nama** | Rangga Timotius |
| **NIM** | 101032300137 |
| **Kelas** | TK46GAB |

---

## 📌 Tujuan Repositori

Repositori ini mendemokan pipeline **deep learning regresi end-to-end** yang mencakup:
preprocessing tabular, perbandingan beberapa arsitektur neural network, hyperparameter tuning otomatis, evaluasi metrik regresi multi-dimensi, interpretabilitas model, dan experiment tracking.

---

## 🧾 Ringkasan Proyek

**Dataset:** `midterm-regresi-dataset.csv` — berisi fitur audio numerik dari ratusan ribu lagu, tanpa header. Kolom pertama adalah **target** (tahun rilis, rentang 1922–2010); 90 kolom berikutnya adalah fitur audio (`feature_1`–`feature_90`) hasil komputasi dari sinyal timbre, tanpa nama yang mudah diinterpretasikan.

**Ukuran data yang digunakan:** 30.000 sampel (stratified random sample dari 95.069 baris total) — untuk menjaga keterjangkauan memori dan waktu training.

**Karakteristik dataset:**
- Tidak ada header pada file CSV asli
- Terdapat 29 missing value yang ditangani otomatis
- Distribusi target condong ke tahun-tahun yang lebih baru (mayoritas > 1990)
- Beberapa fitur memiliki nilai ekstrem / outlier

**Pipeline yang dibangun:**
1. Load & Eksplorasi Data (EDA) — distribusi target, missing value, korelasi
2. Preprocessing — IQR clipping (multiplier=3.0) + StandardScaler + penanganan NaN dari fitur bervarians nol
3. Split Train/Test (80/20, random split)
4. Standardisasi target `year` → training lebih stabil + inverse transform untuk evaluasi
5. Tiga arsitektur deep learning dibandingkan (baseline)
6. Hyperparameter tuning dengan **Optuna** (10 trial per arsitektur)
7. Training final dengan parameter terbaik + logging ke **MLflow**
8. Evaluasi multi-metrik (RMSE, MAE, R²) + plot Actual vs Predicted
9. **Permutation Importance** — interpretasi fitur terpenting untuk model terbaik

---

## 🧠 Deskripsi Model & Hasil Evaluasi

### Arsitektur yang Dibandingkan

| Arsitektur | Deskripsi |
|---|---|
| **Shallow MLP** | 1 hidden layer + Dropout — baseline paling sederhana |
| **Deep MLP** | 4–5 hidden layer dengan BatchNormalization + Dropout, ukuran layer mengecil progresif |
| **Deep Residual MLP** | Beberapa residual block (Dense → BN → ReLU → Dropout → Dense + skip connection) untuk stabilitas training jaringan lebih dalam |

Semua arsitektur menggunakan **output layer linear** (tanpa aktivasi) dan loss **MSE**, sesuai task regresi.

---

### Hasil Baseline (sebelum tuning Optuna)

| Model | RMSE | MAE | R² | Time |
|---|---|---|---|---|
| **Deep MLP** | **8.9277** | **6.2324** | **0.3078** | 19.9s |
| Shallow MLP | 9.0784 | 6.4349 | 0.2843 | 16.4s |
| Deep Residual MLP | 9.1020 | 6.4337 | 0.2805 | 16.7s |

---

### Hyperparameter Terbaik (Optuna, 10 trial per arsitektur)

| Arsitektur | Best RMSE (Optuna) | units | dropout | lr | parameter lain |
|---|---|---|---|---|---|
| Shallow MLP | 8.9761 | 64 | 0.2151 | 0.000963 | — |
| **Deep MLP** | **8.8271** | **256** | **0.2202** | **0.000735** | n_layers=5 |
| Deep Residual MLP | 9.0118 | 64 | 0.3203 | 0.001927 | n_blocks=2 |

---

### Hasil Final (setelah training ulang dengan hyperparameter terbaik)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Deep MLP (Tuned)** ✅ | **8.8591** | **6.2148** | **0.3184** |
| Shallow MLP (Tuned) | 9.0018 | 6.4124 | 0.2963 |
| Deep Residual MLP (Tuned) | 9.0399 | 6.2563 | 0.2903 |

**Model terbaik: Deep MLP (Tuned)** — RMSE **8.86 tahun**, MAE **6.21 tahun**, R² **0.318**

---

### Top 10 Fitur Terpenting — Permutation Importance (Deep MLP)

Permutation Importance mengukur kenaikan RMSE ketika nilai suatu fitur diacak secara acak (semakin besar kenaikan, semakin penting fitur tersebut):

| Rank | Fitur | Kenaikan RMSE |
|---|---|---|
| 1 | `feature_1` | +2.208 tahun |
| 2 | `feature_2` | +0.703 tahun |
| 3 | `feature_3` | +0.531 tahun |
| 4 | `feature_23` | +0.437 tahun |
| 5 | `feature_6` | +0.409 tahun |
| 6 | `feature_13` | +0.230 tahun |
| 7 | `feature_58` | +0.165 tahun |
| 8 | `feature_57` | +0.152 tahun |
| 9 | `feature_63` | +0.131 tahun |
| 10 | `feature_16` | +0.123 tahun |

`feature_1` mendominasi jauh di atas fitur lainnya (+2.208 tahun), disusul `feature_2` dan `feature_3` — konsisten dengan literatur dataset ini yang menyatakan bahwa **rata-rata timbre pertama** adalah fitur paling informatif untuk memprediksi era rilis lagu.

---

## 🗂️ How to Navigate

```
finalterm-deep-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Finalterm_regression_DL.ipynb    ← notebook utama dengan output lengkap
└── data/                                  ← letakkan midterm-regresi-dataset.csv di sini
```

Seluruh alur (EDA → preprocessing → 3 arsitektur → Optuna → MLflow → evaluasi → permutation importance) ada dalam **satu notebook** dengan penjelasan markdown di setiap tahap.

### Cara Menjalankan

1. Clone repo: `git clone <url-repo-ini>`
2. Install dependencies: `pip install -r requirements.txt`
3. Letakkan `midterm-regresi-dataset.csv` di folder `data/`
4. Buka dan jalankan `notebooks/Finalterm_regression_DL.ipynb` dari atas ke bawah
5. Untuk melihat dashboard MLflow: `mlflow ui` → buka `http://localhost:5000`

> Disarankan dijalankan di **Google Colab** (GPU runtime) untuk training yang lebih cepat.

---

## 📊 Metrik Evaluasi yang Digunakan

| Metrik | Keterangan |
|---|---|
| **RMSE** | Root Mean Squared Error — error dalam satuan **tahun**, paling mudah diinterpretasikan |
| **MAE** | Mean Absolute Error — rata-rata error absolut dalam satuan tahun |
| **R²** | Proporsi variansi target yang berhasil dijelaskan model (0–1, makin tinggi makin baik) |

---

## 💡 Catatan Teknis

- **NaN handling setelah scaling:** beberapa fitur memiliki IQR ≈ 0 sehingga setelah clipping semua nilainya identik → variance = 0 → StandardScaler menghasilkan NaN. Ditangani dengan `np.nan_to_num(..., nan=0.0)` sebelum masuk ke model.
- **Target standardization:** `year` distandarisasi sebelum training (`y_train_s = (y - y_mean) / y_std`) agar loss MSE dalam skala yang stabil, lalu di-*inverse transform* untuk evaluasi metrik dalam satuan tahun asli.
- **Permutation Importance** diimplementasikan secara manual (bukan `sklearn.inspection.permutation_importance`) agar kompatibel dengan Keras model tanpa perlu inheritance dari `BaseEstimator`.
- **R² ~0.32** adalah hasil yang wajar untuk dataset ini — tahun rilis lagu tidak semata-mata ditentukan oleh fitur timbre audio saja, melainkan juga dipengaruhi tren industri, label rekaman, dan faktor non-audio yang tidak tersedia di dataset.
