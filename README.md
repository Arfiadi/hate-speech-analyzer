# Indonesian Hate Speech Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-IndoToxic2024%20(IndoDiscourse)-orange.svg)](https://huggingface.co/datasets/Exqrch/IndoDiscourse)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://choosealicense.com/licenses/apache-2.0/)

Proyek kolaboratif mata kuliah **Workshop Proyek Sistem Cerdas** (Dosen Pengampu: **Dr. Selvia Ferdiana Kusuma, M.Kom**) untuk membangun sistem penganalisis ujaran kebencian (*hate speech*) dan toksisitas (*toxicity*) pada teks media sosial berbahasa Indonesia secara *end-to-end* (data preprocessing, perbandingan model ML/DL/Transformer, hingga prototype cerdas).

---

## 📌 Ringkasan Ruang Lingkup Proyek

$$\text{DATA (Teks + Label)} + \text{MODEL (ML/DL/Transformer)} + \text{SISTEM (Prototype Cerdas)} = \text{Keputusan Berbasis Data}$$

1. **Dataset Utama**: `indotoxic2024` ([`Exqrch/IndoDiscourse`](https://huggingface.co/datasets/Exqrch/IndoDiscourse) di Hugging Face) yang mencakup 28.448 baris teks media sosial berbahasa Indonesia beranotasi multi-label.
2. **Task Klasifikasi**:
   - **Task Utama**: Klasifikasi Biner (*Toxic / Hate Speech* vs *Non-toxic*).
   - **Task Lanjutan**: Klasifikasi Multi-label (*Identity attack*, *violence*, *insult*, *profanity*, *sexually explicit*).
3. **Target Evaluasi**:
   - Evaluasi komparatif (*Macro-F1*, *Precision*, *Recall*, *Confusion Matrix*).
   - Penanganan *imbalance dataset* (Baseline, Class-weight / Focal Loss, Augmentasi).
4. **Artefak Akhir**:
   - Model terlatih (*Trained Model Weights*).
   - Laporan eksperimen ilmiah (*Journal Report*).
   - Aplikasi prototype interaktif (*Streamlit / Gradio*).

---

## 📁 Struktur Direktori

> *Catatan: Struktur direktori ini bersifat sementara untuk tahap awal inisiasi & akuisisi data, serta akan terus berkembang seiring berjalannya fase eksperimen proyek.*

```text
hate-speech-analyzer/
├── data/
│   └── raw/                                                   # Data mentah hasil unduhan
│       ├── indotoxic2024_annotated_data_v2_final.csv         # Data teks utama & label anotasi (28.448 baris)
│       ├── indotoxic2024_annotated_data_v2_final.jsonl        # Format JSON Lines
│       ├── indotoxic2024_annotator_demographic_data_v2_final.csv  # Data demografi 29 annotator
│       ├── indotoxic2024_annotator_demographic_data_v2_final.jsonl
│       └── IndoDiscourse_Toxicity_Related_Experiment_Code.ipynb   # Notebook eksperimen acuan resmi
├── docs/
│   ├── Keterangan_Proyek_Hate_Speech_Analyzer.md             # Kontrak kuliah & detail panduan proyek
│   └── Data_Understanding_Guide.md                           # Panduan problem & pemahaman data untuk tim
├── download_data.py                                          # Script otomatis pengunduh dataset resmi
└── README.md                                                 # Dokumentasi utama repositori
```

---

## 🚀 Panduan Setup & Instalasi

### 1. Prasyarat Sistem
- Python 3.10 atau versi yang lebih baru (disarankan menggunakan virtual environment).

### 2. Instalasi Dependensi
Pastikan dependensi dasar terinstal:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers datasets streamlit
```

### 3. Mengunduh Dataset
Dataset dapat diunduh secara otomatis dengan menjalankan:
```bash
python download_data.py
```
File akan otomatis tersimpan di folder `data/raw/`.

---

## 📊 Detail Kolom Dataset (`indotoxic2024`)

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `text_id` | `string` | ID unik untuk setiap teks |
| `text` | `string` | Konten teks postingan / komentar media sosial |
| `annotators_id` | `list` | ID anonim annotator yang menilai teks |
| `topic` | `string` | Topik teks (misal: Disabilitas, Agama, Tionghoa, dll.) |
| `toxicity` | `list` | Anotasi biner toksisitas (`1` = Toksik, `0` = Non-toksik) |
| `identity_attack` | `list` | Ujaran kebencian bermuatan SARA/identitas |
| `threat_incitement_to_violence` | `list` | Ancaman / ajakan kekerasan |
| `insults` | `list` | Penghinaan / cercaan |
| `profanity_obscenity` | `list` | Kata-kata kasar / kotor |
| `sexually_explicit` | `list` | Konten seksual eksplisit |
| `polarized` | `list` | Anotasi keterkaitan polarisasi opini |
| `related_to_election_2024` | `list` | Relevansi dengan pemilu 2024 |
| `is_noise_or_spam_text` | `list` | Tanda apakah teks merupakan spam/noise |

---

## 👥 Matriks Eksperimen Komparatif Kelompok

| No | Kelompok | Kategori | Metode |
|:---:|:---|:---|:---|
| 1 | Kelompok 1 | ML Klasik | Naive Bayes + TF-IDF |
| 2 | Kelompok 2 | ML Klasik | SVM + Random Forest + TF-IDF |
| 3 | Kelompok 3 | Deep Learning | LSTM |
| 4 | Kelompok 4 | Deep Learning | CNN for Text |
| 5 | Kelompok 5 | Transformer | IndoBERT / IndoBERTweet |
| 6 | Kelompok 6 | Transformer | XLM-RoBERTa |

---

## 📜 Lisensi & Sitasi
Dataset ini dilisensikan di bawah [Apache-2.0 License](https://choosealicense.com/licenses/apache-2.0/).

**Sitasi Paper:**
```bibtex
@misc{susanto2025multilabeleddatasetindonesiandiscourse,
      title={A Multi-Labeled Dataset for Indonesian Discourse: Examining Toxicity, Polarization, and Demographics Information}, 
      author={Lucky Susanto and Musa Wijanarko and Prasetia Pratama and Zilu Tang and Fariz Akyas and Traci Hong and Ika Idris and Alham Aji and Derry Wijaya},
      year={2025},
      eprint={2503.00417},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.00417}, 
}
```
