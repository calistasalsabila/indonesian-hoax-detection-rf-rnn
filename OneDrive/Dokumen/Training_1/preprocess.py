import pandas as pd
from sklearn.model_selection import train_test_split

# ===============================
# 1. DEFINISI KEYWORDS POLITIK
# ===============================
POLITIK_KEYWORDS = [
    "politik", "pemilu", "pemilihan umum", "pilkada", "pileg", "pilpres", "kampanye",
    "partai", "partai politik", "caleg", "capres", "cawapres", "calon presiden",
    "calon wakil presiden", "calon legislatif", "suara", "TPS", "DPT", "KPU", "Bawaslu",
    "politikus", "politisi", "anggota DPR", "DPR", "DPRD", "MPR", "lembaga legislatif",
    "parlemen", "ormas", "LSM", "koalisi", "oposisi", "kubu", "kabinet", "menteri",
    "reshuffle", "pemerintahan", "kekuasaan", "pemangku kebijakan", "kepala daerah",
    "gubernur", "bupati", "wali kota", "presiden", "wakil presiden", "sidang paripurna",
    "perppu", "peraturan", "undang-undang", "RUU", "RUU KUHP", "UU ITE", "konstitusi",
    "amandemen", "demokrasi", "otoriter", "otoritarian", "diktator", "sistem politik",
    "ideologi", "pancasila", "reformasi", "orba", "orde baru", "orde lama", "kebijakan",
    "anggaran", "APBN", "APBD", "korupsi", "nepotisme", "kolusi", "KKN", "KPK", "MK",
    "MA", "hukum tata negara", "pelanggaran HAM", "demonstrasi", "aksi", "unjuk rasa",
    "aktivis", "suara rakyat", "politik identitas", "politik uang", "black campaign",
    "hoaks politik", "buzzer", "opini publik"
]

# ===============================
# 2. FUNGSI UTILITY DASAR
# ===============================

def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return set([w.strip() for w in f.readlines() if w.strip()])

def clean_text(text):
    return text.lower()

def is_politik(text):
    """
    Mendeteksi apakah teks mengandung setidaknya satu kata kunci politik.
    Catatan: Teks harus sudah dibersihkan (lowercase dan tanpa stopwords) 
    sebelum fungsi ini dipanggil untuk hasil yang optimal.
    """
    if pd.isna(text) or not text:
        return 0
    
    # Memeriksa kata kunci di dalam teks
    return int(any(kata in text.split() for kata in POLITIK_KEYWORDS))

def load_filter_and_split_datasets(paths, stopwords_path):
    stopwords = load_stopwords(stopwords_path)
    all_dfs = []

    for path in paths:
        df = pd.read_csv(path)

        # ===============================
        # DETEKSI KOLOM TEKS
        # ===============================
        text_col = next((col for col in ["text_new", "Clean Narasi", "FullText", "Narasi"] if col in df.columns), None)
        if text_col is None:
                raise ValueError(f"Tidak ada kolom teks yang cocok di file: {path}")
        
        # Deteksi Kolom Label
        label_col = next((col for col in ["hoax", "label"] if col in df.columns), None)
        if label_col is None:
            raise ValueError(f"Tidak ada kolom label (hoax/label) di file: {path}")
        
        # Seleksi dan rename kolom penting
        df = df[[text_col, label_col]].rename(columns={text_col: 'text_raw', label_col: 'label'})

        # 1. Lowercase
        df['text_clean'] = df['text_raw'].astype(str).apply(clean_text)

        # 2. Hapus stopwords (dilakukan di sini agar filter politik bekerja pada teks yang bersih)
        df['text_clean'] = df['text_clean'].apply(
            lambda x: " ".join([w for w in x.split() if w not in stopwords])
        )

        all_dfs.append(df)
    
    df_gabungan = pd.concat(all_dfs, ignore_index=True)

    # --- Tahap B: Filter Politik dan Penyeimbangan Kelas ---
    
    # 1. Deteksi Politik
    df_gabungan['is_politik'] = df_gabungan['text_clean'].apply(is_politik)

    # 2. Filter hanya data yang berbau politik
    df_politik = df_gabungan[df_gabungan['is_politik'] == 1].copy()

    if df_politik.empty:
         raise ValueError("Tidak ada data yang mengandung kata kunci politik setelah penyaringan.")
    
    # 3. Penyeimbangan Kelas (Undersampling)
    df_hoaks = df_politik[df_politik['label'] == 1]
    df_non_hoax = df_politik[df_politik['label'] == 0]

    min_len = min(len(df_hoaks), len(df_non_hoax))

    # Undersample mayoritas untuk mencapai keseimbangan
    df_hoaks = df_hoaks.sample(n=min_len, random_state=42)
    df_non_hoax = df_non_hoax.sample(n=min_len, random_state=42)

    df_final = pd.concat([df_hoaks, df_non_hoax]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n--- Data Akhir (Politik & Seimbang) ---")
    print(f"Total Sampel: {len(df_final)}")
    print(df_final['label'].value_counts())

    # --- Tahap C: Data Split ---
    
    X_all = df_final['text_clean'].tolist() # Menggunakan teks yang sudah di-clean/stopword-removed
    y_all = df_final['label'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, shuffle=True
    )
    
    return X_train, X_test, y_train, y_test