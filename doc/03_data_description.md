# Evaluasi Kinerja RNN dan Random Forest Berbasis TF-IDF untuk Klasifikasi Berita Hoaks Politik Berbahasa Indonesia

# Dataset Description
Bagian ini menjelaskan secara detail variabel-variabel yang terdapat dalam dataset yang digunakan untuk klasifikasi berita hoaks politik berbahasa Indonesia. Pemahaman setiap variabel penting untuk proses preprocessing, ekstraksi fitur TF-IDF, dan evaluasi model RNN maupun Random Forest.

## 1. Fitur Teks Utama

### **a. text**
Isi berita dalam bentuk teks mentah (raw text).  
Berisi paragraf atau kalimat berita politik yang diklasifikasikan sebagai hoaks atau non-hoaks.

**Contoh isi:**

> "Eks politikus pdip laksamana sukardi gabung pkn jelang pemilu eks politikus senior pdip laksamana sukardi bergabung dengan Partai Kebangkitan Nusantara (pkn)"

Fitur ini menjadi input utama untuk proses TF-IDF, tokenisasi, dan embedding untuk model RNN/LSTM/BiLSTM.

### **b. length**
Jumlah karakter atau jumlah token pada teks.  
Digunakan untuk memeriksa kualitas awal, seperti mendeteksi berita yang terlalu pendek atau tidak informatif.

## 2. Fitur Metadata

### **a. source**
Sumber asal berita, seperti portal berita online, media sosial, atau website tertentu.  
Informasi ini membantu melihat keragaman asal data, meskipun tidak selalu dipakai dalam pemodelan.

### **b. date**
Tanggal publikasi atau tanggal pengambilan data.  
Berguna untuk analisis konteks politik dan memastikan bahwa data sesuai dengan rentang waktu yang dibutuhkan.

### **c. url**
Tautan asli berita (jika ada).  
Disimpan hanya sebagai informasi tambahan dan tidak digunakan dalam proses pelatihan model.

## 3. Variabel Target

Variabel ini menunjukkan kategori dari setiap berita dalam dataset, apakah termasuk hoaks atau bukan hoaks.  
Label digunakan sebagai target pada proses pelatihan model klasifikasi.

**Keterangan nilai:**
- **0 = berita valid (non-hoaks)**  
  Berita yang isinya sesuai fakta, diverifikasi, dan tidak mengandung informasi palsu.
- **1 = berita hoaks**  
  Berita yang mengandung disinformasi, klaim palsu, atau manipulasi konten, khususnya terkait isu politik.

Variabel ini menjadi dasar evaluasi kinerja model TF-IDF + Random Forest serta model RNN, karena keduanya dirancang untuk memprediksi nilai label ini secara akurat.
