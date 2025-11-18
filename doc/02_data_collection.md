# Evaluasi Kinerja RNN dan Random Forest Berbasis TF-IDF untuk Klasifikasi Berita Hoaks Politik Berbahasa Indonesia

# Data Collection
Data collection merupakan tahap penting dalam membangun sistem klasifikasi berita hoaks politik. Dataset yang digunakan pada penelitian ini berisi kumpulan berita politik berbahasa Indonesia, baik yang telah terverifikasi sebagai hoaks maupun non-hoaks. Seluruh data ini akan menjadi dasar untuk proses preprocessing, ekstraksi fitur TF-IDF, serta pelatihan model Random Forest, LSTM, dan BiLSTM.

## 1. Tujuan Pengumpulan Data
Tujuan dari proses pengumpulan data pada penelitian ini adalah untuk memperoleh dataset teks yang:
- Mewakili karakteristik umum berita politik berbahasa Indonesia.
- Mengandung variasi gaya bahasa, sumber, dan konteks politik agar model mampu melakukan generalisasi.
- Memiliki label yang akurat (hoaks /non-hoaks) sehingga dapat digunakan untuk supervised learning.
- Mencukupi kebutuhan volume data agar model berbasis RNN dapat belajar secara efektif.
- Mendukung evaluasi perbandingan performa antara metode klasik (Random Forest + TF-IDF) dan metode deep learning (RNN berbasis TF-IDF + embedding).
  
## 2. Sumber Data
Dataset dalam penelitian ini diperoleh dari platform Kaggle, yang menyediakan kumpulan berita hoaks dan non-hoaks dalam bahasa Indonesia. Dataset tersebut telah tersedia dalam format CSV sehingga memudahkan proses pemuatan, analisis, serta integrasi ke dalam pipeline machine learning.

## 3. Kebutuhan Data
Sebelum pengumpulan data dilakukan, beberapa kebutuhan dataset telah ditentukan, yaitu:
- **Jenis Data :** Teks berita politik berbahasa Indonesia.
- **Label :**
  - 1 = Hoaks
  - 0 = Non-Hoaks
- **Jumlah Data :** Dataset idealnya terdiri dari puluhan ribu teks berita untuk mendukung pelatihan model deep learning.
- **Format Data :** Disimpan dalam format CSV atau teks terstruktur sebelum masuk ke tahap pembersihan dan preprocessing.
- **Kualitas Data :**
  - Tidak mengandung duplikasi yang signifikan
  - Memiliki label yang jelas
  - Memiliki struktur kalimat yang dapat diolah oleh tokenizer dan TF-IDF
  
## 4. Proses Pengumpulan Data

Proses pengumpulan dataset dilakukan melalui beberapa langkah berikut:

**4.1 Mengidentifikasi sumber data**

Menentukan dari mana data akan diambil. Pada proyek ini, dataset berita hoaks/non-hoaks diperoleh dari platform Kaggle yang menyediakan data publik.

**4.2 Mengambil data dari sumber**

Dataset kemudian diunduh dari Kaggle dalam format CSV sehingga mudah diproses lebih lanjut.

**4.3 Menyusun dataset dalam format tersetruktur**
   
  Data dikompilasi menjadi format CSV dengan kolom:
   - text : isi berita
   - label : hoaks/non-hoaks
   - source : asal berita

**4.4 Pengecekan kualitas awal**
   - Mengindentafikasi data kosong
   - Menghapus entri duplikat
   - Mengecek panjang teks (terlalu pendek dihapus)

**4.5 Penyimpanan dataset**
   
   Dataset disimpan sebagai:
   - raw.csv : data asli sebelum preprocessing
   - cleaned.csv : data setelah pembersihan dasar
## 5. Tantangan dalam Pengumpulan Data
Selama proses pengumpulan data, ada beberapa tantangan yang muncul, terutama karena topik yang diangkat adalah berita hoaks politik:

**5.1 Keterbatasan dataset berkualitas**

Dataset hoaks politik berbahasa Indonesia tidak sebanyak dataset teks umum. Banyak dataset berisi jumlah data yang sedikit atau tidak seimbang antara hoaks dan non-hoaks.

**5.2 Variasi gaya bahasa**

Berita politik sering menggunakan bahasa yang berbeda-beda: formal, semi-formal, hingga bahasa media sosial. 

**5.3 Duplikasi konten**

Beberapa berita ditemukan muncul berulang dengan judul berbeda atau isi hampir sama, sehingga perlu pembersihan manual agar tidak memengaruhi distribusi data.

## 6. Praktik terbaik 
Untuk memastikan kualitas dataset, dilakukan beberapa praktik terbaik:
- Menyimpan dataset mentah dan dataset yang sudah dibersihkan secara terpisah
- Menggunakan sumber hoaks yang kredibel untuk memastikan validitas label
- Melakukan deduplication dan quality check sebelum preprocessing
- Menetapkan format dataset yang konsisten
  
## 7. Hasil Pengumpulan data
Pada akhir tahap pengumpulan data, diperoleh dataset yang:
- Berisi puluhan ribu teks berita politik, baik hoaks maupun non-hoaks.
- Memiliki format terstruktur dalam file CSV yang siap diproses.
- Telah melalui pemeriksaan awal kualitas (panjang teks, duplikasi, kelengkapan label).
- Mampu mendukung tahap-tahap berikutnya yaitu:
    - Preprocessing teks
    - TF-IDF vectorization
    - Pelatihan model Random Forest
    - Pelatihan model LSTM dan BiLSTM
    - Evaluasi performa model berdasarkan metrik akurasi, presisi, recall, dan f1-score
