# Dokumen Resmi Pemilihan Tech Stack & Metodologi

**Proyek:** Indonesian Hate Speech Analyzer — Kelompok 6  
**Mata Kuliah:** Workshop Proyek Sistem Cerdas  
**Dosen Pengampu:** Dr. Selvia Ferdiana Kusuma, M.Kom  

Dokumen ini merangkum keputusan resmi terkait arsitektur teknologi (tech stack) dan pustaka (library) yang digunakan dalam proyek ini, beserta kelebihan dan justifikasi pengambilan keputusannya.

---

## 1. Ringkasan Tech Stack Final

| Bagian | Teknologi Terpilih | Status |
|:---|:---|:---:|
| Bahasa pemrograman | **Python** | ✅ Dipertahankan |
| Pengolahan data | **Pandas, NumPy, Regex** | ✅ Dipertahankan |
| Pembagian data | **Scikit-learn, Scikit-multilearn** | ✅ Dipertahankan |
| Model utama | **XLM-RoBERTa Base** | ✅ Dipertahankan |
| Framework model | **PyTorch** | ✅ Dipertahankan |
| Library transformer | **Hugging Face Transformers, Datasets, Accelerate** | 🆕 Ditambahkan |
| Optimasi | **AdamW** | ✅ Dipertahankan |
| Evaluasi | **Scikit-learn & HuggingFace Evaluate** | 🔄 Diubah |
| Interpretasi model | **Captum (Integrated Gradients)** | 🔄 Diubah |
| Pencatatan eksperimen| **Weights & Biases (wandb)** | 🔄 Diubah |
| Penyimpanan model | **Hugging Face Hub + safetensors** | 🆕 Ditambahkan |
| Frontend / Prototype | **Streamlit** | 🆕 Ditambahkan |

*(Catatan: Penggunaan FastAPI, Docker, dan deployment cloud backend secara eksplisit dihapus dari lingkup proyek untuk menghindari over-engineering).*

---

## 2. Analisis Lengkap per Teknologi (Kelebihan, Justifikasi, Kegunaan, & Implementasi)

### A. Core & Data Processing
#### 1. Python (Bahasa Pemrograman)
* **Kelebihan:** Standar industri untuk Data Science dan AI/ML. Memiliki ekosistem pustaka open-source terlengkap.
* **Justifikasi:** Seluruh ekosistem machine learning modern (PyTorch, Hugging Face, Scikit-learn) dibangun di atas Python. Tidak ada alternatif lain yang rasional untuk proyek NLP berbasis Transformer.
* **Kegunaan:** Bahasa dasar untuk menulis instruksi logika, pemrosesan data, pelatihan model, hingga merancang antarmuka aplikasi.
* **Implementasi:** Digunakan secara menyeluruh di semua skrip `.py` (seperti `download_data.py`, `train.py`, `app.py`) dan notebook eksperimen `.ipynb` di seluruh proyek.

#### 2. Pandas, NumPy, Regex (Pengolahan Data)
* **Kelebihan:** Operasi vektorisasi cepat untuk dataset tabular; Regex sangat andal untuk text-cleaning.
* **Justifikasi:** Wajib digunakan karena dataset `indotoxic2024` memiliki format tabular dan berisi teks media sosial kotor yang butuh pembersihan tingkat karakter.
* **Kegunaan:** Membaca data mentah, mengubah format string label ke list (*parsing*), menghitung *majority voting*, statistik deskriptif EDA, dan membersihkan teks (menghapus URL, emoji, mention, dll).
* **Implementasi:** Diimplementasikan masif pada **Fase 1 (Tahap Data-Centric)** oleh *Data Engineer* (Anggota 1), dikhususkan pada file notebook EDA dan skrip fungsi pembersihan teks (`clean_text()`).

#### 3. Scikit-learn & Scikit-multilearn (Pembagian Data)
* **Kelebihan:** `scikit-multilearn` menyediakan algoritma pembagian data berstratifikasi untuk kasus multi-label.
* **Justifikasi:** Dataset `indotoxic2024` memiliki *class imbalance* ekstrem, terutama pada sub-label langka seperti `sexually_explicit` (hanya 50 sampel positif). *Random split* biasa berisiko menumpuk kelas ini hanya di satu tempat.
* **Kegunaan:** Membagi data menjadi himpunan *Train*, *Validation*, dan *Test* dengan proporsi kelas minoritas yang adil dan merata.
* **Implementasi:** Dieksekusi sejak **Tahap 1 (Data-Centric)** untuk membagi data eksperimen, dan dikunci secara permanen menjelang **Tahap 2** setelah *Golden Dataset* terbentuk. Dijalankan oleh *ML Engineer* (Anggota 2).

### B. Modeling & Training
#### 4. XLM-RoBERTa Base (Model Utama)
* **Kelebihan:** Transformer multibahasa tangguh (278M parameter), pre-trained di 100 bahasa (termasuk Indonesia), memberikan representasi kontekstual teks yang sangat kaya.
* **Justifikasi:** Ini adalah spesifikasi arsitektur yang diwajibkan oleh dosen untuk Kelompok 6 guna diadu performanya. Varian "Base" dipilih untuk menghemat memori (VRAM) GPU saat *training*.
* **Kegunaan:** Bertindak sebagai *shared encoder* dan "otak" sistem yang mampu mencerna nuansa, sarkasme, dan slang bahasa Indonesia.
* **Implementasi:** Di-*load* menggunakan modul Transformers. Pada **Tahap 1**, digunakan dengan konfigurasi dibekukan sebagai *evaluator* kualitas data. Pada **Tahap 2 dan 3**, menjadi inti komputasi yang di-*fine-tune* secara penuh.

#### 5. PyTorch (Framework Model)
* **Kelebihan:** Dinamis (dynamic computation graph), sangat *pythonic*, dan menjadi standar utama di riset AI akademis.
* **Justifikasi:** Memberikan kebebasan mutlak untuk merancang arsitektur kelas kustom (seperti klasifikasi bertingkat/hierarkis) yang tidak bisa dilakukan oleh modul standar.
* **Kegunaan:** Mendefinisikan lapisan jaringan saraf (Neural Network), menghitung fungsi *Loss*, *backpropagation*, dan mengolah data latih ke dalam *DataLoader*.
* **Implementasi:** Menjadi fondasi pada pembuatan file `dataset.py` (untuk kustomisasi input) dan di dalam skrip arsitektur klasifikasi *classification head*.

#### 6. Hugging Face Transformers, Datasets, Accelerate (Library Transformer)
* **Kelebihan:** Abstraksi tingkat tinggi, mempercepat eksperimen, dan mengoptimalkan manajemen memori (VRAM/RAM) secara otomatis.
* **Justifikasi:** Mustahil melatih model berukuran ratusan juta parameter pada GPU gratis (Kaggle/Colab) tanpa manajemen memori dan *mixed precision* dari ekosistem Hugging Face.
* **Kegunaan:** Mengunduh bobot XLM-RoBERTa, melakukan tokenisasi teks agar dipahami model, memuat data tanpa membanjiri RAM, dan melakukan *training loop* secara efisien (*Trainer API*).
* **Implementasi:** Merupakan pustaka (*library*) utama yang diimpor pada seluruh skrip *training* di Fase 2 dan 3 oleh Anggota 2 dan 3.

#### 7. AdamW (Optimasi)
* **Kelebihan:** Menggabungkan adaptivitas kecepatan belajar (Adam) dengan *Weight Decay* yang terpisah (*decoupled*).
* **Justifikasi:** Secara empiris dan teoretis merupakan optimisator *golden standard* yang menghasilkan konvergensi tercepat pada arsitektur Transformer.
* **Kegunaan:** Mengoreksi parameter (bobot) dari XLM-RoBERTa pada setiap putaran belajar (epoch) agar tingkat kesalahan prediksi (loss) semakin menurun.
* **Implementasi:** Diatur sebagai nilai pada argumen `TrainingArguments` saat memanggil kelas `Trainer` dari Hugging Face.

### C. Evaluasi & Observasi
#### 8. Scikit-learn & Hugging Face Evaluate (Evaluasi)
* **Kelebihan:** Pustaka terpercaya yang menjamin akurasi dan legitimasi penghitungan matematika statistik (F1-score, presisi, dll).
* **Justifikasi:** Laporan akhir dan presentasi UAS wajib menyertakan matriks performa komparatif yang ketat dan terstandarisasi.
* **Kegunaan:** Mengevaluasi seberapa pintar model; menerjemahkan prediksi mentah menjadi skor kuantitatif terukur dan matriks kebingungan (Confusion Matrix).
* **Implementasi:** **HF Evaluate** digunakan pada fungsi `compute_metrics` di dalam siklus *training* tiap epoch. **Scikit-learn** diimplementasikan pasca-training saat melakukan *Error Analysis* mendalam oleh Anggota 3.

#### 9. Captum / Integrated Gradients (Interpretasi Model)
* **Kelebihan:** Metode Explainable AI (XAI) berbasis gradien yang menjamin integritas kalkulasi, lebih cepat, dan hemat memori dibanding SHAP.
* **Justifikasi:** Antarmuka (prototype akhir) diwajibkan transparan. Model ML tidak boleh menjadi "kotak hitam" (*black-box*); manusia harus tahu kenapa model menebak suatu kalimat toksik.
* **Kegunaan:** Menganalisis kata per kata (token) dan memberi warna (highlight) pada kata yang paling berkontribusi pada keputusan deteksi *hate speech*.
* **Implementasi:** Diimplementasikan oleh tim *ML/UI Developer* (Anggota 4) pada *backend inference script*, dan hasilnya diumpankan langsung ke desain UI untuk fitur *Token Highlighting*.

#### 10. Weights & Biases / wandb (Pencatatan Eksperimen)
* **Kelebihan:** Papan hub (*dashboard*) berbasis web yang indah, kolaboratif, dan otomatis memonitor penggunaan *hardware*.
* **Justifikasi:** Eksperimen tersebar pada banyak notebook oleh 5 anggota tim berbeda di platform terpisah. *wandb* menyatukan seluruh log riwayat eksperimen dalam satu *link* publik terpusat.
* **Kegunaan:** Menampilkan grafik garis untuk melihat apakah *loss* turun (model membaik) atau naik (*overfitting*), mendokumentasikan hiperparameter eksperimen, dan melakukan *ablation study*.
* **Implementasi:** Cukup dengan menambahkan parameter `report_to="wandb"` pada konfigurasi *training*. Log otomatis dikirim ke akun W&B proyek dari tiap skrip *training*.

### D. Penyimpanan & Delivery
#### 11. Hugging Face Hub + safetensors (Penyimpanan Model)
* **Kelebihan:** `safetensors` cepat saat di-*load* dan aman dari eksekusi kode jahat. HF Hub adalah *cloud* penyimpanan gratis untuk ML.
* **Justifikasi:** Menyediakan "jembatan emas" antara tim Machine Learning dan tim UI. Tim UI (Anggota 4) tidak perlu menjalankan script pelatihan yang makan waktu berjam-jam.
* **Kegunaan:** Sebagai tempat penyimpanan permanen bobot akhir model XLM-RoBERTa hasil *fine-tuning* terbaik (*checkpoint*).
* **Implementasi:** Diimplementasikan dengan kode `model.push_to_hub()` di baris paling akhir skrip *training*. Tim UI memanggilnya dari internet saat aplikasi `app.py` dimulai.

#### 12. Streamlit (Frontend / Prototype)
* **Kelebihan:** Pembuatan UI *dashboard* secara cepat (*rapid prototyping*) menggunakan sintaks Python, tanpa perlu HTML/CSS.
* **Justifikasi:** Sangat efektif dan bebas *layout* untuk membuat purwarupa (prototype) yang memenuhi kriteria ujian akhir.
* **Kegunaan:** Menyediakan antarmuka visual (layar UI) di mana dosen/pengguna bisa mengetik teks, menekan tombol, lalu melihat label prediksi, level keparahan (*severity*), dan alasan prediksi.
* **Implementasi:** Merupakan kerangka kerja (*framework*) utama pada berkas kode `app.py` yang dibangun oleh *System/UI Developer* untuk demonstrasi UAS di Fase 2 akhir.

---

## 3. Fitur/Teknologi yang Dieksklusi (Anti-Pattern)

Dalam perencanaannya, opsi-opsi *deployment backend* seperti **FastAPI, Docker, dan infrastruktur cloud deployment** sengaja dihapus.
* **Kelebihan Peniadaan (Lean Approach):** Memastikan tim Machine Learning dan tim UI tetap fokus mengeksekusi *pipeline* yang berdampak langsung pada laporan riset (Data Engineering, Tuning Transformer, dan UX analisis error). 
* **Justifikasi:** Berdasarkan kontrak kuliah, proyek ini mensyaratkan prototipe didemokan **secara lokal**. Ekosistem *Streamlit* sudah mencakup inferensi model (*backend logic*) sekaligus tampilan layar interaktif (*frontend*) yang berjalan bersama pada satu proses Python. Membangun *microservices REST API* (FastAPI) dalam wadah *container* (Docker) merupakan praktik keahlian spesialisasi *MLOps* (Over-engineering), yang dapat menghabiskan waktu estimasi berminggu-minggu tanpa memberikan bobot nilai tambahan yang sepadan.
