# Preprocessing Experiment Summary

## 1. Tujuan

Eksperimen preprocessing dilakukan untuk membandingkan tiga perlakuan teks sebelum digunakan pada tahap pemodelan XLM-RoBERTa:

- P-A: Raw Text
- P-B: Heavy Cleaning
- P-C: Minimal Transformer Cleaning

Ketiga varian menggunakan data, label, dan fixed split yang sama sehingga perbedaan hasil eksperimen dapat dikaitkan dengan perlakuan preprocessing.

## 2. Strict Consensus Subset

Dataset preprocessing menggunakan strict consensus dengan kriteria:

- Minimal 2 anotator.
- Seluruh anotator memberikan label toxicity yang sama.
- Label akhir hanya 0 atau 1.
- Teks tidak boleh kosong.

Jumlah data hasil strict consensus:

**8.842 data**

Output:

`data/preprocessing_consensus_subset.csv`

## 3. Fixed Split

Dataset dibagi menjadi fixed split:

| Split | Jumlah |
|---|---:|
| Train | 7.073 |
| Validation | 884 |
| Test | 885 |
| Total | 8.842 |

Output:

```text
data/preprocessing_split/
├── train.csv
├── validation.csv
└── test.csv

## 4. Preprocessing Variants

### P-A — Raw Text

Teks dipertahankan dalam bentuk aslinya tanpa perubahan isi teks.

Output:

data/preprocessing_variants/p_a_raw/
├── train.csv
├── validation.csv
└── test.csv

### P-B — Heavy Cleaning

Preprocessing dilakukan secara lebih agresif, meliputi:
- lowercase
- menghapus URL
- menghapus mention
- mengubah hashtag menjadi kata
- menghapus emoji
- menghapus punctuation
- menghapus angka
- menghapus stopword

Output:

data/preprocessing_variants/p_b_heavy/
├── train.csv
├── validation.csv
└── test.csv

### P-C — Minimal Transformer Cleaning

Preprocessing minimal untuk mempertahankan informasi penting pada teks:
- URL → [URL]
- mention → [USER]
- mengurangi karakter berulang
- mengurangi punctuation berulang
- merapikan whitespace
- mempertahankan kapitalisasi
- mempertahankan emoji
- mempertahankan angka
- mempertahankan negasi
- tanpa stemming atau lemmatization

Output:

data/preprocessing_variants/p_c_minimal/
├── train.csv
├── validation.csv
└── test.csv

## 5. Quality Assurance

QA dilakukan untuk memastikan:
- jumlah row P-A, P-B, dan P-C konsisten
- text_id konsisten
- toxicity_label konsisten
- fixed split konsisten
- aturan preprocessing P-C terpenuhi

Hasil QA:

**STATUS: PASS**

Output:

reports/preprocessing_quality_check.csv
reports/preprocessing_before_after_samples.csv
reports/preprocessing_statistics.csv
reports/p_c_rule_check.csv

## 6. Script

Script yang digunakan:

scripts/make_consensus_subset.py
scripts/make_fixed_split.py
scripts/make_preprocessing_variants.py
scripts/qa_preprocessing.py
scripts/qa_p_c_rules.py

## 7. Status Akhir

Preprocessing experiment dan quality assurance telah selesai.

Dataset P-A, P-B, dan P-C telah dibuat dengan fixed split yang sama dan siap digunakan untuk tahap eksperimen pemodelan XLM-RoBERTa.

Seluruh hasil telah di-commit dan di-push ke branch:

feat/preprocessing-golden-dataset

Commit:

e136183 — Add preprocessing consensus and variants