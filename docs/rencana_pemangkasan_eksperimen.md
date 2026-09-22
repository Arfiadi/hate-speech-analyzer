# Rencana Pemangkasan Eksperimen: Detail & Justifikasi

## Asumsi Dasar Perhitungan

| Parameter | Nilai | Sumber |
|:---|:---|:---|
| Waktu 1× training run (full dataset, 3 epoch) | ~1–1.5 jam | Estimasi XLM-RoBERTa Base pada Kaggle T4 |
| Waktu 1× training run (subset 5.200 baris, 3 epoch) | ~15 menit | Proporsional terhadap jumlah data |
| Kuota GPU Kaggle per akun per minggu | ~30 jam | Kebijakan Kaggle 2026 |
| Jumlah anggota tim dengan akun Kaggle | 5 orang | Kelompok 6 |
| Total kuota GPU tersedia per minggu | ~150 jam | 5 × 30 jam |

---

## TAHAP 1: Eksperimen Data-Centric

### 1.1 Preprocessing Teks (P-A / P-B / P-C)

|  | Rencana Saat Ini | Rencana Pemangkasan |
|:---|:---|:---|
| **Skenario** | 3 skenario (P-A, P-B, P-C) | **3 skenario** (P-A, P-B, P-C) — tidak dipangkas |
| **Data** | Subset unanimous ~5.200 baris | Subset unanimous ~5.200 baris |
| **Repetisi** | 3 run per skenario (rata-rata ± std) | **1 run** per skenario |
| **Total run** | 9 run | **3 run** |
| **Estimasi GPU** | 9 × 15 menit = ~2.25 jam | **3 × 15 menit = ~0.75 jam** |

**Apa yang dipangkas:** Repetisi 3× dikurangi menjadi 1×.

**Justifikasi:**
- Repetisi 3× dengan random seed berbeda memang ideal untuk melaporkan stabilitas (± std). Namun pada fine-tuning Transformer, varian antar-seed **sangat kecil** (umumnya ±0.5–1% F1) karena model sudah *pre-trained* dengan miliaran data.
- Perbedaan antar-skenario preprocessing (P-B vs P-C) diprediksi **jauh lebih besar** (5–10% F1) daripada varian antar-seed. Artinya, 1 run sudah cukup untuk menentukan pemenang.
- Jika dosen meminta angka ± std untuk jurnal, kita bisa menambahkan 2 run tambahan **hanya untuk skenario pemenang** di akhir proyek.

**Apakah tetap optimal?** ✅ Ya. Kita tidak kehilangan informasi apa pun tentang skenario mana yang terbaik. Kita hanya menunda pelaporan stabilitas.

---

### 1.2.1 Resolusi Duplikat (D-A / D-B / D-C)

|  | Rencana Saat Ini | Rencana Pemangkasan |
|:---|:---|:---|
| **Skenario** | 3 skenario (D-A, D-B, D-C) | **2 skenario** (D-A, D-C) |
| **Total run** | 3 run | **2 run** |
| **Estimasi GPU** | 3 × 15 menit = ~0.75 jam | **2 × 15 menit = ~0.5 jam** |

**Apa yang dipangkas:** Skenario D-B (*Drop Duplicates* sederhana) dihilangkan.

**Justifikasi:**
- D-B (`drop_duplicates(keep='first')`) adalah solusi naif yang **secara pasti lebih buruk** dari D-C (*Pool & Re-vote*). D-B membuang informasi anotator tambahan, sedangkan D-C justru menggabungkannya.
- D-B juga **secara pasti lebih baik** dari D-A (karena minimal menghilangkan data leakage). Posisinya di antara D-A dan D-C sudah bisa diprediksi tanpa perlu dijalankan.
- Yang benar-benar perlu dibuktikan adalah: **"Apakah Pool & Re-vote (D-C) signifikan lebih baik daripada tidak melakukan apa-apa (D-A)?"**

**Apakah tetap optimal?** ✅ Ya. D-C hampir pasti menang. Membuktikan D-A vs D-C sudah cukup untuk jurnal.

---

### 1.2.2 Resolusi Tie / Disagreement (T-A / T-B / T-C)

|  | Rencana Saat Ini | Rencana Pemangkasan |
|:---|:---|:---|
| **Skenario** | 3 skenario (T-A, T-B, T-C) | **3 skenario** (T-A, T-B, T-C) — tidak dipangkas |
| **Total run** | 3 run | 3 run |
| **Estimasi GPU** | 3 × 15 menit = ~0.75 jam | **0.75 jam** |

**Apa yang dipangkas:** Tidak ada.

**Justifikasi:**
- Ketiga skenario ini benar-benar **tidak bisa diprediksi hasilnya** tanpa eksperimen. Apakah menghapus data *tie* (T-A) lebih baik daripada memperlakukannya sebagai sinyal ambiguitas (T-B *soft label*)? Ini adalah pertanyaan ilmiah terbuka.
- Biayanya sangat murah (0.75 jam GPU). Tidak ada alasan untuk memangkas.
- Hasilnya akan menjadi kontribusi menarik di jurnal.

**Apakah tetap optimal?** ✅ Ya. Eksperimen ini utuh.

---

### 1.2.3 Penanganan Single-Annotator (A-A / A-B / A-C / A-D)

|  | Rencana Saat Ini | Rencana Pemangkasan |
|:---|:---|:---|
| **Skenario** | 4 skenario (A-A, A-B, A-C, A-D) | **3 skenario** (A-A, A-B, A-C) |
| **Total run** | 4 run | **3 run** |
| **Estimasi GPU** | 4 × ~1 jam = ~4 jam ⚠️ | **3 × ~1 jam = ~3 jam** |

> [!WARNING]
> Mulai dari eksperimen ini, kita menggunakan **seluruh dataset** (~24.000–28.000 baris), bukan subset 5.200. Waktu per run melonjak menjadi 1–1.5 jam.

**Apa yang dipangkas:** Skenario A-D (*Consensus Only*) dihilangkan.

**Justifikasi:**
- A-D membuang 55% dataset (15.748 baris single-annotator). Dari perspektif deep learning, **lebih banyak data hampir selalu lebih baik**, terutama untuk model Transformer yang lapar data.
- Paper IndoToxic2024 sendiri sudah membuktikan bahwa single-annotator data mengandung informasi linguistik yang berharga meskipun label-nya noisier. Membuang separuh dataset ini hampir pasti kontraproduktif.
- Yang benar-benar menarik secara ilmiah adalah perbandingan **A-A vs A-B vs A-C**: apakah memberi bobot lebih rendah pada single-annotator (A-B) atau menggunakan curriculum learning (A-C) bisa mengungguli perlakuan setara (A-A)?

**Apakah tetap optimal?** ✅ Ya. A-D adalah skenario dengan probabilitas menang paling rendah. Menghapusnya tidak mengurangi peluang menemukan Golden Dataset.

---

### Ringkasan Tahap 1

| Metrik | Rencana Saat Ini | Rencana Pemangkasan | Penghematan |
|:---|:---:|:---:|:---:|
| **Total run** | 19 run | **11 run** | -8 run |
| **Total GPU** | ~8 jam | **~5 jam** | ~3 jam |
| **Skenario ilmiah yang hilang** | — | D-B dan A-D | Keduanya bisa diprediksi hasilnya tanpa eksperimen |

---

## TAHAP 2: Eksperimen Model-Centric

### 2.1 Loss Function Biner (L-A / L-B / L-C)

|  | Rencana Saat Ini | Rencana Pemangkasan |
|:---|:---|:---|
| **Skenario** | 3 skenario (L-A, L-B, L-C) | **3 skenario** (L-A, L-B, L-C) — tidak dipangkas |
| **Total run** | 3 run | 3 run |
| **Estimasi GPU** | 3 × ~1.5 jam = ~4.5 jam | **4.5 jam** |

**Apa yang dipangkas:** Tidak ada.

**Justifikasi:**
- Ini adalah inti dari Tahap 2. Perbedaan antara BCE biasa, Weighted BCE, dan Focal Loss bisa sangat signifikan (selisih 10–20% Recall) pada dataset dengan imbalance 1:11.
- Hanya 3 run, biaya rendah, dampak tinggi. Wajib dijalankan semua.

**Apakah tetap optimal?** ✅ Ya. Eksperimen ini utuh.

---

### 2.2 Hyperparameter Tuning (Grid Search)

|  | Rencana Saat Ini | Rencana Pemangkasan |
|:---|:---|:---|
| **Parameter yang di-tune** | LR (4) × Batch (3) × Max Len (3) × Epochs (3) | **LR (3) × Threshold (8)** |
| **Total run** | **108 run** ⚠️⚠️⚠️ | **3 run** + threshold gratis |
| **Estimasi GPU** | ~162 jam (TIDAK MUNGKIN!) | **~4.5 jam** |

**Apa yang dipangkas:** Batch Size, Max Length, dan Epochs **dikunci** (tidak dieksplorasi).

**Parameter yang dikunci dan alasannya:**

| Parameter | Dikunci di | Justifikasi |
|:---|:---:|:---|
| **Max Length** | 256 | Data EDA menunjukkan P95 panjang teks = 174 kata. Max Length 256 token sudah mencakup >96% teks secara utuh. Menaikkan ke 384 hanya menambah VRAM tanpa manfaat. Menurunkan ke 128 berisiko memotong ~15% teks yang panjang |
| **Batch Size** | 16 | Standar paper IndoToxic2024. Batch 8 terlalu lambat (training 2× lebih lama). Batch 32 seringkali menyebabkan Out of Memory (OOM) di Kaggle T4 (15GB VRAM) untuk model XLM-RoBERTa |
| **Epochs** | Diganti *Early Stopping* | Alih-alih menguji 3 vs 4 vs 5 epoch secara manual, kita set maximum 5 epoch dengan `early_stopping_patience=2`. Model akan otomatis berhenti di epoch optimal. Ini adalah pendekatan yang **lebih superior** daripada grid search manual pada jumlah epoch |

**Parameter yang TETAP di-tune:**

| Parameter | Range | Jumlah Run | Justifikasi |
|:---|:---|:---:|:---|
| **Learning Rate** | {1e-5, 2e-5, 3e-5} | 3 | LR adalah hyperparameter **paling berpengaruh** untuk fine-tuning Transformer. Perbedaan antara 1e-5 dan 5e-5 bisa berarti konvergensi vs divergensi total. Tiga titik (konservatif, standar, agresif) sudah cukup untuk menemukan sweet spot |
| **Threshold Biner** | [0.15, 0.50] step 0.05 | **0** (gratis) | Threshold tuning **tidak memerlukan re-training!** Cukup simpan probabilitas prediksi dari 1 run, lalu iterasi threshold secara offline di laptop lokal menggunakan Scikit-learn. Biaya GPU = 0 |

**Apakah tetap optimal?** ✅ Ya, bahkan **lebih baik.** Alasannya:
1. Mengunci Max Length dan Batch Size berdasarkan fakta EDA dan constraint hardware adalah keputusan *engineering* yang bijak, bukan kompromi.
2. Mengganti grid search epoch dengan Early Stopping secara teori memberikan hasil yang **lebih optimal** karena epoch berhenti tepat di titik terbaik, bukan di angka bulat (3, 4, atau 5).
3. Memisahkan threshold tuning dari training loop menghemat puluhan jam GPU dan justru memberikan granularitas yang lebih halus (step 0.05 vs harus re-train setiap kali).

---

### Ringkasan Tahap 2

| Metrik | Rencana Saat Ini | Rencana Pemangkasan | Penghematan |
|:---|:---:|:---:|:---:|
| **Total run** | 111 run | **6 run** | -105 run |
| **Total GPU** | ~167 jam ❌ | **~9 jam** | ~158 jam |
| **Kualitas hasil** | Brute-force optimal | **Smart optimal** (+ Early Stopping lebih superior) |

---

## TAHAP 3: Ekspansi Multi-Label

### 3.1 Arsitektur Head (H-A / H-B / H-C)

|  | Rencana Saat Ini | Rencana Pemangkasan |
|:---|:---|:---|
| **Skenario** | 3 skenario (H-A, H-B, H-C) | **3 skenario** (H-A, H-B, H-C) — tidak dipangkas |
| **Total run** | 3 run | 3 run |
| **Estimasi GPU** | 3 × ~1.5 jam = ~4.5 jam | **4.5 jam** |

**Apa yang dipangkas:** Tidak ada.

**Justifikasi:**
- Ini adalah inti kontribusi ilmiah terbesar proyek Anda. Perbandingan Flat vs Dual-Head vs Hierarchical pada dataset IndoToxic2024 adalah hal yang **belum pernah dilakukan** oleh paper asli Susanto et al. (2025). Ini adalah nilai jual jurnal Anda.
- Biayanya cukup murah (3 run). Harus dijalankan semua.

**Apakah tetap optimal?** ✅ Ya. Eksperimen ini utuh.

---

### 3.2 Loss Multi-Label (ML-A / ML-B / ML-C)

|  | Rencana Saat Ini | Rencana Pemangkasan |
|:---|:---|:---|
| **Skenario** | 3 skenario (ML-A, ML-B, ML-C) | **2 skenario** (ML-A, ML-C) |
| **Total run** | 3 run | **2 run** |
| **Estimasi GPU** | 3 × ~1.5 jam = ~4.5 jam | **3 jam** |

**Apa yang dipangkas:** ML-B (Focal Loss per label).

**Justifikasi:**
- ML-B (Focal Loss) dan ML-C (ASL / Asymmetric Loss) keduanya bertujuan menangani class imbalance, tapi ASL adalah **generalisasi dan penerus** dari Focal Loss. ASL menambahkan mekanisme *probability shifting* (parameter `clip`) yang secara teoritis dan empiris **selalu >= Focal Loss** pada kasus extreme multi-label imbalance.
- Paper asli ASL (Ridnik et al., 2021) sudah membuktikan bahwa ASL secara konsisten mengungguli Focal Loss pada benchmark multi-label.
- Yang benar-benar perlu dibuktikan: **"Apakah ASL (ML-C) signifikan lebih baik daripada Weighted BCE sederhana (ML-A)?"** Jika ya, itu sudah cukup meyakinkan di jurnal.

**Apakah tetap optimal?** ✅ Ya. Kita mempertahankan baseline (ML-A) dan kandidat terkuat (ML-C).

---

### 3.3 Per-Label Threshold Tuning

|  | Rencana Saat Ini | Rencana Pemangkasan |
|:---|:---|:---|
| **Metode** | Grid search per label | **Grid search per label** — tidak dipangkas |
| **Total run** | **0** (dihitung offline) | **0** (dihitung offline) |
| **Estimasi GPU** | 0 jam | 0 jam |

**Apa yang dipangkas:** Tidak ada (dihitung secara offline di laptop lokal, GPU = 0 jam).

---

### Ringkasan Tahap 3

| Metrik | Rencana Saat Ini | Rencana Pemangkasan | Penghematan |
|:---|:---:|:---:|:---:|
| **Total run** | 6 run | **5 run** | -1 run |
| **Total GPU** | ~9 jam | **~7.5 jam** | ~1.5 jam |

---

## Ringkasan Total Keseluruhan

| Metrik | Rencana Saat Ini | Rencana Pemangkasan | Penghematan |
|:---|:---:|:---:|:---:|
| **Total training run** | **136 run** | **22 run** | -114 run (84%) |
| **Total GPU yang dibutuhkan** | **~184 jam** ❌ | **~21.5 jam** ✅ | ~162.5 jam |
| **Waktu minimum (1 akun Kaggle)** | ~6 minggu penuh | **< 1 minggu** | |
| **Skenario ilmiah yang dibuang** | — | D-B, A-D, ML-B | 3 skenario |
| **Skenario ilmiah yang dipertahankan** | 23 skenario | **20 skenario** | Hanya kehilangan 3 skenario yang hasilnya bisa diprediksi |

---

## Apa yang TIDAK Dipangkas Sama Sekali

| Eksperimen | Alasan tetap utuh |
|:---|:---|
| Preprocessing (P-A/B/C) | Pertanyaan fundamental, biaya murah |
| Tie Handling (T-A/B/C) | Tidak bisa diprediksi, pertanyaan terbuka |
| Loss Biner (L-A/B/C) | Inti Tahap 2, dampak besar pada Recall |
| Arsitektur Head (H-A/B/C) | Kontribusi ilmiah terbesar proyek |
| Per-label Threshold | Gratis (offline computation) |
| Error Analysis setiap tahap | Bukan training run, dilakukan di laptop |
