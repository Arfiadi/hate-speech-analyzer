# Kriteria Penulisan & Desain Presentasi (Murni untuk PPT)

Revisi pada konten presentasi di bawah ini telah disesuaikan berdasarkan **3 Kriteria Utama Desain Presentasi Akademik/Teknis**:

1. **Prinsip *Cognitive Load* (Beban Kognitif):** Menghapus kode rahasia/ID internal seperti (P-A, P-B, H-C, D-B) pada tampilan utama *slide*. Audiens akan kelelahan jika harus mengingat arti dari setiap singkatan huruf. Singkatan diganti dengan **Nama Deskriptif** yang langsung dipahami maknanya (misal: "Arsitektur Hierarki", "Pembersihan Minimal").
2. **Prinsip *Scannability* (Keterbacaan Cepat):** Kalimat panjang dipotong menjadi frasa *punchy* (singkat dan padat). *Slide* berfungsi sebagai papan reklame (kata kunci), sedangkan penjelasan lengkapnya dituturkan melalui ucapan (terdapat di *Speaker Notes*).
3. **Penyelarasan dengan *Smart Tuning***: *Slide* 7, 8, dan 12 dirombak untuk mencerminkan strategi "Pemangkasan Eksperimen" agar tim Anda tidak terlihat *"asal grid search"* yang mustahil dieksekusi secara komputasi. Ini menunjukkan kedewasaan *engineering*.

---

# Draft Konten Presentasi Final (Siap Copy-Paste ke PPT)

## Slide 1: Judul Proyek
**Judul Utama:** Desain Eksperimen & Pipeline ML: XLM-RoBERTa
**Sub-judul:** Deteksi Ujaran Kebencian Bahasa Indonesia Berbasis Hierarki Dua Tahap
**Identitas:** Kelompok 6 — Indonesian Hate Speech Analyzer

**Spesifikasi Teknis:**
*   **Model Inti:** xlm-roberta-base
*   **Dataset:** IndoToxic2024 (28.448 teks, 9 label)
*   **Acuan:** Susanto et al. (2025), Findings of ACL 2025

>**Speaker Notes:** Buka presentasi dengan menegaskan bahwa proyek ini berfokus pada pembangunan pipeline deteksi ujaran kebencian yang terstruktur dan terukur menggunakan fondasi XLM-RoBERTa pada dataset IndoToxic2024.

---

## Slide 2: Filosofi & Objektif Eksperimen
**Prinsip Utama:**
*   Data-Centric lebih dulu $\rightarrow$ Model-Centric kemudian.
*   Task Biner lebih dulu $\rightarrow$ Multi-Label kemudian.
*   Isolasi Variabel Tunggal (hanya ubah 1 parameter per uji).

**4 Objektif Utama:**
1.  Menciptakan **"Golden Dataset"** Biner yang bersih dari konflik label.
2.  Menemukan arsitektur & *loss* biner yang paling optimal.
3.  Ekspansi ke 5 sub-label toksisitas dengan penanganan *extreme imbalance*.
4.  Menjadikan **Error Analysis** sebagai kompas iterasi, bukan sekadar laporan akhir.

>**Speaker Notes:** Tekankan pentingnya disiplin metodologi; kita tidak langsung melompat ke model rumit atau multi-label sebelum kualitas data dan model biner terbukti kokoh.

---

## Slide 3: Arsitektur Task (Dua Tahap Klasifikasi)
**Struktur Pipeline:**
*   **Shared Encoder:** `xlm-roberta-base` sebagai pengekstraksi semantik teks.
*   **Task 1 (Gerbang Biner):** Deteksi *Toxic* vs *Non-Toxic*.
*   **Task 2 (Hierarki Multi-Label):** Evaluasi 5 sub-label spesifik **HANYA JIKA** lolos Task 1.
*   **Pencegahan Anomali:** Jika biner diprediksi non-toxic, semua probabilitas sub-label otomatis menjadi nol (0).

```text
Teks ──> [ XLM-RoBERTa ] ──> [ Task 1: Biner ]
                                /         \
                         (Toxic)           (Non-Toxic)
                           /                 \
            [ Task 2: Multi-Label ]       [ Semua Sub-label = 0 ]
            ├─ identity_attack
            ├─ threat
            └─ ...dsb
```

>**Speaker Notes:** Jelaskan diagram ini dengan menekankan bahwa kegagalan di Task 1 akan merusak Task 2. Model standar sering berhalusinasi (teks non-toxic tapi skor ancaman tinggi). Gerbang biner kita mencegah inkonsistensi logika ini secara arsitektural.

---

## Slide 4: Baseline Model & Metrik Evaluasi
**Konfigurasi Dasar (Dikunci untuk Tahap 1):**
*   **Classification Head:** 1 Linear Layer (768 $\rightarrow$ 1) + Sigmoid.
*   **Max Length:** 256 token (menghemat VRAM, mencakup >96% teks utuh).
*   **Batch Size & Epoch:** 16 *batch size*, dengan batas 3 epoch.
*   **Fungsi Loss:** Naive *BCEWithLogitsLoss*, Threshold 0,5.

**Metrik Keberhasilan:**
*   🔴 **Utama:** Binary F1 (Fokus pada ketepatan kelas minoritas/Toxic).
*   ⚪ **Pendukung:** Precision, Recall, Macro-F1.

>**Speaker Notes:** Seluruh parameter model dikunci mati pada Tahap 1. Ini menjamin bahwa naik-turunnya performa murni karena manipulasi data, bukan karena modelnya yang kebetulan pintar.

---

## Slide 5: Tahap 1.1 — Eksperimen Preprocessing Teks
**Tujuan:** Menentukan pembersihan teks paling efektif pada subset **konsensus mutlak / *unanimous*** (~5.200 baris di mana seluruh annotator 100% sepakat tanpa ada selisih pendapat).

**Perbandingan Skenario:**
| Metode | Detail Operasi | Hipotesis |
|:---|:---|:---|
| **Raw Text** | Teks mentah tanpa ubahan | Batas bawah (Baseline) |
| **Heavy Cleaning** | Hapus tanda baca, angka, kapital, stopwords | Terburuk (Konteks rusak) |
| **Pembersihan Minimal** | Masking `[URL]` & `[USER]`, norm repetisi | **Terbaik (Sinyal utuh)** |

**Bukti Temuan Data (EDA):**
*   Huruf kapital berulang (*shouting*) muncul di 27–37% teks toksik.
*   Ada >11.000 kata negasi kritis (*tidak, bukan*) yang akan hancur jika *stopwords* dihapus.

>**Speaker Notes:** Paparkan bahwa transformer butuh konteks kalimat yang utuh. Pembersihan klasik justru menghapus fitur kritis seperti kapitalisasi (marah) dan negasi.

---

## Slide 6: Tahap 1.2 — Eksperimen Label Engineering
**1. Resolusi Konflik Duplikat**
*   **EDA:** 403 teks duplikat memiliki label saling berlawanan.
*   **Metode:** *Pool & Re-vote* (gabung total suara seluruh kemunculan teks).
*   **Hipotesis:** Akan memulihkan *ground truth* secara optimal tanpa membuang informasi.

**2. Resolusi Label Seri (Ambiguitas)**
*   **EDA:** Tingkat seri cukup tinggi (*Polarized* 9,6%, *Toxic* 6,5%).
*   **Metode:** *Soft-Label* (target 0.5) vs *Drop* (buang data).
*   **Hipotesis:** *Soft-Label* lebih akurat mencerminkan keraguan logis manusia dibanding membuang data.

**3. Penanganan Data Tunggal (55% Dataset)**
*   **EDA:** *Recall* anotator tunggal sangat buruk (50,7%) pada kebencian implisit.
*   **Metode:** *Curriculum Learning* (belajar bertahap) vs *Confidence Weighting*.
*   **Hipotesis:** *Curriculum Learning* mengungguli *Confidence Weighting* untuk menambah kosa kata tanpa merusak data pasti.

🏆 **Keluaran Tahap 1:** Menghasilkan "Golden Dataset" yang dikunci secara permanen.

>**Speaker Notes:** Rincikan bagaimana kita menyelesaikan ambiguitas label dari anotator manusia guna mengunci Golden Dataset sebelum masuk ke optimasi model.

---

## Slide 7: Tahap 2 — Eksperimen Model-Centric (Biner)
**A. Solusi Imbalance Biner (Rasio 1:11)**
*   **EDA:** Kelas non-toxic 11x lipat lebih dominan. Model rentan bias "Aman".
*   **Metode:** *Weighted BCE* vs *Focal Loss*.
*   **Hipotesis:** *Weighted BCE* mendongkrak *Recall* minoritas. *Focal Loss* lebih presisi menekan bobot data "mudah".

**B. Smart Hyperparameter Tuning (Hemat Komputasi)**
*   **Strategi:** Mengunci *Max Length* (256) & *Batch Size* (16). Hanya melakukan penelusuran (*tuning*) pada *Learning Rate*.
*   **Justifikasi:** Bukti EDA menunjukkan >96% teks utuh di bawah 256 token. Mengunci nilai ini akan memangkas ratusan jam komputasi GPU Kaggle tanpa mengorbankan kualitas model.

>**Speaker Notes:** Jelaskan bahwa kita TIDAK menggunakan brute-force grid search karena tidak efisien. Kita mengunci Max Length dan Batch Size berdasarkan EDA untuk menghemat ratusan jam GPU Kaggle.

---

## Slide 8: Tahap 3 — Ekspansi Multi-Label
**A. Eksperimen Arsitektur (Classification Head)**
*   **Masalah Utama:** Model standar sering berhalusinasi (memprediksi skor sub-label tinggi pada teks yang sebenarnya *non-toxic*).
*   **Metode yang Diuji:** Arsitektur Hierarki (*Conditional*) vs *Flat* vs *Dual Head*.
*   **Hipotesis:** Arsitektur Hierarki akan memblokir inkonsistensi logika tersebut sejak awal (mencegah *cascading error*).

**B. Solusi Ketimpangan Label Ekstrem (Imbalance)**
*   **EDA (Tantangan):** Adanya kelas super-langka (*Threat* 1:314, *Sexually Explicit* 1:567).
*   **Metode:** *Asymmetric Loss (ASL)* vs *Weighted BCE*.
*   **Hipotesis:** ASL akan menang karena merupakan *State-of-the-Art* khusus peredam penalti palsu pada kelas ekstrem.

>**Speaker Notes:** Garis bawahi arsitektur Hierarki yang mencegah anomali prediksi. Dan tekankan uji coba fungsi loss ASL yang sangat canggih untuk kasus langka.

---

## Slide 9: Per-Label Threshold & Evaluasi
**Tuning Threshold Independen Per Sub-Label:**

| Sub-Kategori | Rasio | Rentang Threshold | Rasionalisasi |
|:---|:---:|:---:|:---|
| `identity_attack` | 1:34 | Moderat | Imbalance standar |
| `profanity_obscenity` | 1:106 | Rendah | Imbalance tinggi |
| `threat` | 1:314 | Ekstrem | **Prioritas Recall** (Ancaman bahaya) |
| `sexually_explicit` | 1:567 | Ekstrem | Sampel sangat sedikit |

**Metrik Keberhasilan Final:**
*   Skor **Binary F1** tidak boleh anjlok.
*   Diukur menggunakan **Per-label F1, Macro-F1 Multi-label, dan Hamming Loss.**

>**Speaker Notes:** Jelaskan alasan mengapa tiap kelas punya threshold beda. Kelas berisiko tinggi seperti ancaman kekerasan (threat) wajib diprioritaskan ketangkap (recall) dengan menurunkan threshold.

---

## Slide 10: Skema Pembagian Data (Data Split)
**Prinsip Validitas Uji:**
1.  **Multilabel Stratified Split:** Algoritma khusus agar 50 baris *sexually explicit* terdistribusi adil di Train/Val/Test (tidak menumpuk di satu tempat).
2.  **Validasi Bersih:** Test/Val set wajib bersih dari label seri (tie) di seluruh kategori.

```text
Dataset Mentah ──> Deduplikasi (Pool & Re-vote) ──> Golden Data
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        Train Set (80%)         Val (10%) & Test (10%)
  (Diuji dengan Tie Handling)    (Bebas Tie & Anotator Valid)
```

>**Speaker Notes:** Soroti risiko stratifikasi acak biasa yang dapat melenyapkan kelas minoritas secara tak sengaja dari set validasi pengujian.

---

## Slide 11: Error Analysis (Diagnostik Iteratif)
**Peran Kunci:** Dijalankan di akhir setiap eksperimen sebagai pemicu perbaikan kode, bukan sekadar pelaporan statistik buta.

**Taksonomi Error yang Dipantau Manual:**
*   **False Positive (FP):** Bias identitas, kegagalan konteks negasi (*“saya TIDAK benci”*), dan umpatan kasual pertemanan (*slang*).
*   **False Negative (FN):** Kebencian implisit, istilah terselubung (*dogwhistle* seperti “kadrun”/“cebong”), dan penyamaran karakter (*4nj1ng*).
*   **Cascading Error:** Gerbang biner gagal, sehingga seluruh sub-label mati total.

>**Speaker Notes:** Tekankan bahwa metrik rata-rata (angka) sering menyembunyikan bias model. Inspeksi kualitatif 20–50 baris error secara manual dengan mata manusia adalah bagian paling krusial dari pipeline kita.

---

## Slide 12: Roadmap Eksekusi Eksperimen
**Matriks Alur Kerja:**

| No | Fokus Eksperimen | Metode yang Diuji | Target Keluaran |
|:---:|:---|:---|:---|
| 1 | Preprocessing (Biner) | Minimal vs Heavy Cleaning | Fungsi Pembersih Terbaik |
| 2 | Deduplikasi (Biner) | Metode Pool & Re-vote | Reduksi Konflik Label |
| 3 | Tie Handling (Biner) | Drop vs Soft-Label | Penanganan Label 0.5 |
| 4 | Data Tunggal (Biner) | Curriculum vs Confidence Weight | **Golden Dataset Terkunci** |
| 5 | Fungsi Loss (Biner) | Weighted BCE vs Focal Loss | Recall Minoritas Naik |
| 6 | Smart Tuning (Biner) | Tuning LR & Threshold *Offline* | **Konfigurasi Biner Final** |
| 7 | Arsitektur Head (Multi) | Flat vs Dual vs Hierarki | Arsitektur Bebas Anomali |
| 8 | Loss Label (Multi) | Weighted BCE vs ASL | **Model Final Siap Pakai** |

>**Speaker Notes:** Tutup presentasi dengan memperlihatkan peta jalan eksekusi yang disiplin, memprioritaskan kecerdasan komputasi (GPU budget), dan bertahap dari data mentah menuju model multi-label final.
