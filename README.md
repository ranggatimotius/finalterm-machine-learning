# 📝 Kesimpulan Akhir: Evaluasi Model Deep Learning untuk Fraud Detection

## Rangkuman Hasil Semua Model

| Model | AUC-ROC | Avg Precision (PR-AUC) | F1-Fraud | Time |
|---|---|---|---|---|
| Shallow MLP (baseline) | 0.9047 | 0.5484 | 0.4576 | 62.9s |
| Deep MLP (baseline) | 0.9073 | 0.5591 | 0.4942 | 78.5s |
| Deep Residual MLP (baseline) | 0.9152 | 0.6178 | 0.5568 | 66.9s |
| Shallow MLP (Tuned) | 0.9186 | 0.5948 | — | — |
| **Deep MLP (Tuned) — Best AUC-ROC** | **0.9214** | 0.6092 | — | — |
| Deep Residual MLP (Tuned) — Best PR-AUC | 0.9202 | **0.6342** | — | — |

*\*Dataset: 590.540 transaksi training, fraud rate asli 3.50% / 20.663 transaksi; setelah preprocessing tersisa 106 fitur final; SMOTE menyeimbangkan training set menjadi 455.902 vs 455.902*

---

## Interpretasi & Analisis Mendalam

### 1. Kedalaman Arsitektur vs Hyperparameter Tuning
Sebelum tuning, **Deep Residual MLP** paling unggul (AUC 0.9152) dibanding Deep MLP (0.9073) dan Shallow MLP (0.9047). Hal ini membuktikan bahwa *residual connection* sangat membantu model dalam menangkap hubungan non-linear antar fitur ter-enkripsi (`V*`) dengan lebih baik. Namun setelah di-tuning menggunakan **Optuna**, gap performa ketiganya menjadi sangat tipis (0.9186–0.9214). Ini menunjukkan bahwa *hyperparameter* yang optimal dapat mendongkrak arsitektur sederhana (Shallow MLP) hingga mendekati performa model yang lebih kompleks.

### 2. Trade-off Metrik: AUC-ROC vs PR-AUC
Berdasarkan **AUC-ROC**, **Deep MLP (Tuned)** memenangkan pengujian dengan skor **0.9214**. Namun, jika melihat **Average Precision/PR-AUC**—metrik yang jauh lebih representatif untuk kasus kelas minoritas ekstrem seperti fraud (3.5%)—**Deep Residual MLP (Tuned)** justru menjadi yang tertinggi (**0.6342** vs 0.6092). 
* **Rekomendasi Strategis:** Jika prioritas utama sistem adalah mendeteksi fraud dengan presisi tinggi pada kelas minoritas, Deep Residual MLP merupakan pilihan yang lebih tepat dibanding hanya mengandalkan AUC-ROC.

### 3. Evaluasi Ketajaman Deteksi (Recall Fraud)
Berdasarkan *classification report* model terbaik (Deep MLP Tuned, threshold 0.5) pada validation set:
* **Fraud:** Precision 0.55, **Recall 0.59**, F1 0.57 *(Support: 4.133 transaksi fraud)*
* **Normal:** Precision 0.99, Recall 0.98, F1 0.98

*Recall* sebesar 59% mengindikasikan bahwa **41% transaksi fraud pada validation set masih lolos tidak terdeteksi** (False Negative) pada penggunaan default threshold 0.5. Untuk studi kasus *fraud detection*, di mana kerugian akibat fraud yang lolos jauh lebih besar daripada transaksi normal yang salah blokir (False Positive), threshold probabilitas sebaiknya diturunkan (misal ke 0.3) untuk meningkatkan *recall*.

### 4. Temuan Penting: Perbedaan Rate Prediksi pada Test Set (Data Drift)
Model memprediksi **17.19% transaksi pada test set sebagai fraud** (87.094 dari 506.691). Angka ini melonjak tajam dibanding fraud rate asli di training (3.50%) maupun validation set (≈3.7%). Faktor penyebab utama:
* **Data Drift / Konsep Pergeseran Waktu:** Pada dataset kompetisi ini, test set diambil dari periode waktu yang berbeda, sehingga karakteristik transaksi atau pola fraud telah bergeser.
* **Efek SMOTE terhadap Kalibrasi:** Model dilatih pada data yang di-resample seimbang (50:50). Hal ini menyebabkan bias probabilitas output, sehingga ambang batas (threshold) 0.5 menjadi tidak optimal saat dihadapkan kembali pada distribusi data asli yang sangat imbalanced.
* **Solusi:** Sebelum model di-deploy ke lingkungan produksi, wajib dilakukan kalibrasi ulang threshold menggunakan metode seperti *Platt scaling* atau *Isotonic Regression*.

### 5. Komparasi dengan Model Klasik (Tree-Based)
Pada eksperimen pendahulu, **XGBoost yang di-tuning berhasil mencapai AUC-ROC 0.9549**—berada di atas ketiga arsitektur deep learning pada eksperimen ini (kisaran 0.92). Temuan ini sejalan dengan literatur ilmiah bahwa **Gradient Boosting Tree-based models sering kali lebih unggul untuk data tabular terstruktur** dibandingkan dengan neural network standar, kecuali menggunakan arsitektur DL khusus tabular (seperti TabNet atau FT-Transformer) dengan tuning yang sangat ekstensif.

---

## Implementasi Optimasi RAM

| Teknik Optimasi | Dampak & Efisiensi |
|---|---|
| **Dtype Downcasting** (`float64` → `float32`) | **Train RAM:** 756.9 MB → 256.2 MB (Hemat **66.1%**)<br>**Test RAM:** 645.6 MB → 221.3 MB (Hemat **65.7%**) |
| **Selective Loading** (`usecols`) | Mereduksi load kolom dari 394 kolom menjadi hanya **168 kolom** terpilih (113 dari 339 kolom `V*` + 55 kolom utama) |
| **Missing Value Filtering** | Mengeliminasi kolom dengan data hilang > 70% (Menyisakan **106 kolom final**) |
| **Garbage Collection** | Penggunaan `del objek` disertai `gc.collect()` secara berkala untuk membebaskan RAM sesegera mungkin |
| **Feature Scaling** | Penerapan `StandardScaler` sebelum data masuk ke Neural Network untuk menjaga stabilitas konvergensi |

---

## Saran Pengembangan Selanjutnya

1. **Kalibrasi Threshold:** Lakukan penyesuaian threshold berdasarkan kurva target recall/precision pada validation set sebelum melakukan final submission.
2. **Interpretabilitas DL:** Integrasikan pustaka **SHAP** atau **Permutation Importance** untuk memahami kontribusi fitur ter-enkripsi terhadap hasil prediksi Neural Network.
3. **Advanced Tabular DL:** Uji coba arsitektur khusus data tabular seperti *TabNet* atau *FT-Transformer* untuk mengejar ketertinggalan performa dari XGBoost.
4. **Feature Engineering Agregatif:** Buat fitur turunan berbasis agregasi perilaku (misal: jumlah transaksi per kartu atau domain email dalam kurun waktu tertentu).
5. **Data Enrichment:** Gabungkan data transaksi dengan tabel identitas (*identity table*) untuk memanfaatkan fitur perangkat (`device`), browser, dan sistem operasi (`OS`).
6. **Eksplorasi Hyperparameter:** Tingkatkan parameter `N_TRIALS` pada Optuna (saat ini dibatasi 5 trial per arsitektur) guna menemukan kombinasi hyperparameter yang jauh lebih optimal.
