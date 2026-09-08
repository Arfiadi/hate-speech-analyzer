# Kontrak Kuliah & Kickoff Proyek: Membangun Indonesian Hate Speech Analyzer

**Mata Kuliah**: Workshop Proyek Sistem Cerdas  
**Dosen Pengampu**: Dr. Selvia Ferdiana Kusuma, M.Kom  
**Tahun Akademik**: 2026  
**Struktur Kelas**: 30 Mahasiswa | 6 Kelompok (5 Orang/Kelompok) | 1 Proyek Bersama  

---

## 1. Paradigma & Visi Proyek

Proyek ini tidak hanya sekadar melatih model (*training model*), melainkan merancang alur keputusan berbasis data yang *actionable* hingga sampai ke tangan pengguna akhir.

$$\text{DATA (Teks + Label)} + \text{MODEL (ML/DL/Transformer)} + \text{SISTEM (Prototype Cerdas)} = \text{Keputusan Berbasis Data}$$

### Tiga Pilar Utama:
- **Akurat**: Evaluasi matang dan adil melalui metode komparatif yang ketat.
- **Transparan**: Hasil yang dapat dijelaskan (*explainable*) dari data mentah hingga output akhir.
- **Berguna**: Dibungkus dalam antarmuka cerdas (*intelligent interface*) yang menghasilkan tindakan nyata.

---

## 2. Strategi Kolaborasi & Pembagian Tugas

Proyek dijalankan dengan strategi: **1 Dataset Bersama $\rightarrow$ 6 Eksperimen Berbeda $\rightarrow$ Bukti Komparatif $\rightarrow$ 1 Sistem Bersama (Model Terbaik/Juara Kelas).**

### Matriks Arsitektur Eksperimen Komparatif

| No | Kelompok | Kategori | Metode | Fokus Kontribusi Jurnal |
|---|:---|:---|:---|:---|
| 1 | Kelompok 1 | ML Klasik (Statistik Tradisional) | Naive Bayes + TF-IDF | Sub-bab metode, hasil, & *error analysis* NB |
| 2 | Kelompok 2 | ML Klasik (Statistik Tradisional) | SVM + Random Forest + TF-IDF | Sub-bab metode, hasil, & *error analysis* SVM |
| 3 | Kelompok 3 | Deep Learning (Jaringan Saraf) | LSTM | Sub-bab metode, hasil, & *error analysis* LSTM |
| 4 | Kelompok 4 | Deep Learning (Jaringan Saraf) | CNN for Text | Sub-bab metode, hasil, & *error analysis* CNN |
| 5 | Kelompok 5 | Transformer (Attention Mechanism) | IndoBERT / IndoBERTweet | Analisis domain spesifik (Media Sosial) |
| 6 | Kelompok 6 | Transformer (Attention Mechanism) | XLM-RoBERTa | Pengujian pembanding transformer multibahasa |

---

## 3. Ruang Lingkup Teknis & Standar Evaluasi

- **Dataset**: `indotoxic2024` (Teks media sosial berbahasa Indonesia + label toxicity).
- **Target Klasifikasi**:
  - **Task Utama**: Klasifikasi Biner (*Toxic/Hate Speech* vs *Non-toxic*).
  - **Task Lanjutan**: Klasifikasi Multi-label (*Identity attack*, *violence*, *insult*, *profanity*, *sexually explicit*).
- **Syarat Validasi Model**:
  - Pembagian Data: *Train / Validation / Test split*.
  - Metrik Wajib: *Macro-F1*, *Precision*, *Recall*, *Confusion Matrix*.
  - Penanganan Imbalance: Uji perbandingan antara *Baseline*, *Class-weight / Focal loss*, dan augmentasi data latih secara terbatas.
  - **Syarat Mutlak**: Sepenuhnya reprodusibel dari data mentah hingga hasil akhir.

---

## 4. Anatomi Sistem & Visi Produk

Sistem prototype akhir menterjemahkan teks input dan probabilitas menjadi keputusan yang mudah dipahami:
- **Input**: Teks komentar / unggahan media sosial.
- **Output Prediksi**: Label Klasifikasi (*Hate Speech* / *Non-Hate Speech*).
- **Confidence Score**: Persentase keyakinan model (misal: 91.7%).
- **Severity & Target**: Tingkat keparahan (*Severity: Low/Medium/High*) dan Target (*Individual/Group*).
- **Explainability**: Penandaan kata-kata kunci (*token highlighting*) yang memicu klasifikasi.

---

## 5. Roadmap Pelaksanaan (M1 – M16)

### Fase 1: Membangun Fondasi Pembuktian (Fokus UTS: M1 – M8)
- **M1**: Orientasi (Kontrak, Dataset `indotoxic2024`, GitHub Setup).
- **M2**: *Problem Framing* (Definisi label, etika, *data split*).
- **M3–M4**: Eksplorasi Data (EDA) & Preprocessing Sentral kolaboratif.
- **M5**: Eksekusi Baseline (TF-IDF komparatif).
- **M6–M7**: Eksperimen Kelompok Tahap 1 & Evaluasi Awal.
- **M8 (UTS)**: Presentasi Progres Pipeline & Demo Pemahaman Individu.

### Fase 2: Optimasi & Integrasi Sistem (Fokus UAS: M9 – M16)
- **M9–M10**: Eksperimen Lanjutan (*Hyperparameter Tuning*, *Ablation Studies*, teknik penanganan imbalance).
- **M11**: Implementasi Task Lanjutan (*Multi-label Classification*).
- **M12–M13**: Rekap Perbandingan Kelas & Pemilihan Model Juara untuk Integrasi.
- **M14–M15**: Pengembangan Prototype (Streamlit/Gradio), *UI/UX Refinement*, & Finalisasi Laporan Jurnal.
- **M16 (UAS)**: Presentasi Final Kelayakan Sistem & Rilis Aplikasi.

---

## 6. Syarat Kelulusan Proyek (3 Artefak Utama)

1. **Model**: Model deteksi sesuai arsitektur kelompok yang telah melalui proses *tuning* dan stabil.
2. **Experimental Report**: Dokumentasi artikel ilmiah/jurnal komprehensif mencakup *preprocessing*, *hyperparameter*, metrik evaluasi lengkap, dan *error analysis* mendalam.
3. **Intelligent System Prototype**: Aplikasi *end-to-end* (Streamlit/Gradio) yang reprodusibel di GitHub.

---

## 7. Skema Penilaian & Bobot

| Komponen Penilaian | Bobot | Deskripsi |
|---|:---:|---|
| **Partisipasi** | 10% | Kehadiran, diskusi kelas, dan *peer assessment* |
| **Tugas Mingguan** | 20% | Ketepatan waktu dan kualitas progres mingguan |
| **UTS (Minggu 8)** | 20% | Demo *pipeline* dan ujian pemahaman individu |
| **Implementasi Teknis** | 20% | Kualitas kode GitHub, reprodusibilitas, dan *commit history* |
| **UAS (Minggu 15–16)** | 30% | Presentasi final kelayakan sistem & penulisan jurnal komparatif |
| **Total** | **100%** | |

*Catatan: Nilai akhir individu dipengaruhi secara langsung oleh jejak kontribusi teknis nyata dalam repositori kelompok.*

---

## 8. Checklist Rilis Final & Langkah Awal

### Checklist Rilis Final
- [ ] `README.md` & *environment installation* terdefinisi jelas.
- [ ] Random seed / konfigurasi eksperimen tercatat lengkap.
- [ ] Metrik evaluasi & *error analysis* tuntas.
- [ ] *Commit history* mencerminkan kontribusi aktif & nyata tiap anggota.
- [ ] Prototype berhasil didemokan secara lokal tanpa kendala.

### 4 Langkah Awal (Minggu 1)
1. **Bentuk Tim**: Sepakati 5 anggota per kelompok.
2. **Akuisisi Data**: Unduh dan pelajari struktur dataset `indotoxic2024`.
3. **Infrastruktur**: Setup repositori GitHub kelompok.
4. **Strategi**: Pembagian peran (*Data Engineer*, *ML Engineer*, *UI/System Developer*).
