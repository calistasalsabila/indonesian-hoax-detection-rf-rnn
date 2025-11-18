# Evaluasi Kinerja RNN dan Random Forest Berbasis TF-IDF untuk Klasifikasi Berita Hoaks Politik Berbahasa Indonesia

# Definisi Masalah

## 1. Latar Belakang
Penyebaran berita hoaks di Indonesia terus meningkat seiring dengan tingginya penggunaan media sosial dan platform digital. Informasi yang tidak akurat dapat membentuk opini publik secara keliru, menimbulkan keresahan, hingga menurunkan kualitas literasi digital masyarakat. Proses verifikasi berita secara manual membutuhkan waktu, ketelitian, dan pengalaman yang memadai, sehingga kurang efektif ketika volume informasi sangat besar. Pendekatan berbasis pemrosesan bahasa alami (NLP) dan pembelajaran mesin menawarkan solusi otomatis untuk mengelompokkan berita ke dalam kategori hoaks atau fakta melalui analisis teks.

## 2. Konteks
Dataset yang digunakan terdiri dari berita berbahasa Indonesia yang telah diberi label sebagai *hoaks* atau *fakta*. Teks pada setiap berita diubah menjadi representasi numerik menggunakan TF-IDF, yang memberikan bobot pada kata-kata berdasarkan tingkat kepentingannya. Penelitian ini membandingkan tiga model utama:

- **LSTM**   
- **BiLSTM**  
- **Random Forest**  

Tujuan utama perbandingan ini adalah membandingkan kinerja model LSTM, BiLSTM, dan Random Forest dalam menangani karakteristik teks Bahasa Indonesia yang beragam dan dinamis.

## 3. Masalah
Metode deteksi hoaks secara tradisional memiliki sejumlah keterbatasan. Pemeriksaan manual memakan waktu, sulit dilakukan dalam skala besar, serta rentan terhadap perbedaan penilaian antar pemeriksa. Selain itu, pola bahasa pada berita hoaks kerap berubah sehingga sulit diidentifikasi tanpa alat bantu otomatis. Penelitian ini bertujuan mengembangkan dan mengevaluasi model klasifikasi berbasis TF-IDF dengan pendekatan pembelajaran mesin dan deep learning untuk menghasilkan deteksi hoaks yang lebih cepat, akurat, dan konsisten.

## 4. Tujuan
- Mengembangkan model klasifikasi berita hoaks menggunakan LSTM, BiLSTM, dan Random Forest berbasis TF-IDF.  
- Membandingkan performa ketiga model melalui metrik evaluasi seperti akurasi, precision, recall, dan F1-score.  
- Mengidentifikasi fitur dan pola teks yang paling memengaruhi prediksi model.  
- Memberikan analisis komprehensif tentang keunggulan dan keterbatasan setiap model.

## 5. Ruang Lingkup

### Lingkup Termasuk
- Pemrosesan teks berita berbahasa Indonesia yang telah diberi label sebagai hoaks atau fakta.
- Penerapan metode representasi teks menggunakan TF-IDF sebagai fitur utama untuk seluruh model.
- Pengembangan dan evaluasi model LSTM, BiLSTM, dan Random Forest untuk tugas klasifikasi biner.
- Analisis performa model berdasarkan metrik akurasi, precision, recall, dan F1-score.

### Lingkup Tidak Termasuk
- Penggunaan model transformer modern seperti BERT atau IndoBERT.
- Analisis mendalam terhadap aspek linguistik pragmatik atau semantik di luar pemrosesan teks berbasis fitur TF-IDF.
- Penanganan data multimodal seperti gambar, komentar pengguna, atau metadata tautan berita.

## 6. Dampak
**Pengguna:** Peneliti NLP, lembaga pemeriksa fakta, dan pengembang sistem deteksi hoaks.  
**Sasaran Manfaat:** Masyarakat, organisasi pemerhati literasi digital, dan perusahaan teknologi.  
**Pengaruh:** Peningkatan akurasi identifikasi hoaks, pengurangan penyebaran informasi palsu, serta perbaikan kualitas ekosistem informasi digital di Indonesia.

## 7. Kriteria keberhasilan
- Model mencapai tingkat akurasi yang memenuhi standar evaluasi sistem klasifikasi berita hoaks.
- Model mampu meminimalkan kesalahan prediksi positif dengan menghasilkan presisi yang tinggi.
- Model dapat mendeteksi sebanyak mungkin berita hoaks yang benar tanpa banyak terlewat (false negative minimal).
- Model menunjukkan keseimbangan optimal antara presisi dan recall melalui pencapaian F1-score yang tinggi.
  

## 8. Risiko dan Asumsi Dasar

### Risiko
- Ketidakseimbangan kelas pada dataset dapat menyebabkan bias prediksi terhadap salah satu kategori.
- Variasi gaya bahasa dan ejaan pada berita dapat menurunkan kualitas representasi fitur TF-IDF.
- Model RNN seperti LSTM dan BiLSTM berpotensi mengalami overfitting apabila ukuran dataset terbatas.

### Asumsi Dasar
- Teks dalam dataset dianggap telah melalui proses pelabelan yang valid dan dapat dipertanggungjawabkan.
- Representasi TF-IDF dinilai cukup untuk menangkap informasi penting dari berita berbahasa Indonesia.
- Lingkungan eksperimen (hyperparameter, pembagian data, dan preprocessing) diasumsikan stabil dan konsisten.

## 9. Keterbatasan
- Ketergantungan pada TF-IDF membuat model kurang efektif dalam menangkap konteks semantik mendalam dibandingkan metode berbasis embedding modern.
- Ukuran dataset yang terbatas dapat membatasi kemampuan model RNN dalam mempelajari pola bahasa yang kompleks.
- Hasil evaluasi hanya berlaku pada dataset yang digunakan dan tidak serta-merta mencerminkan performa model pada data dunia nyata yang lebih beragam.
