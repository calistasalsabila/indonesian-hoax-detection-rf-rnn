# type: ignore

import os
import numpy as np
from preprocess import load_filter_and_split_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Input, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

DATASET_PATHS = [
    r"C:\Users\liali\OneDrive\Dokumen\Training_1\data\dataset_cnn_10k_cleaned.csv",
    r"C:\Users\liali\OneDrive\Dokumen\Training_1\data\dataset_kompas_4k_cleaned.csv",
    r"C:\Users\liali\OneDrive\Dokumen\Training_1\data\dataset_tempo_6k_cleaned.csv",
    r"C:\Users\liali\OneDrive\Dokumen\Training_1\data\dataset_turnbackhoax_10_cleaned.csv"
]

STOPWORDS_PATH = r"C:\Users\liali\OneDrive\Dokumen\Training_1\data\stopwords_id.txt"
MODEL_SAVE_PATH = "models/bilstm_model.h5"

# ========================
# PREPROCESS DATA
# ========================
X_train, X_test, y_train, y_test = load_filter_and_split_datasets(DATASET_PATHS, STOPWORDS_PATH)

# --- Definisikan Hyperparameter ---
VOCAB_SIZE = 5000
TFIDF_FEATURES = 5000
MAX_LEN = 200

# ========================
# TF-IDF
# ========================
tfidf = TfidfVectorizer(max_features=TFIDF_FEATURES)
tfidf.fit(X_train)

X_train_tfidf = tfidf.transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# ========================
# Tokenizer untuk BiLSTM
# ========================
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

# ========================
# Konversi Label ke NumPy Array (Sangat Dianjurkan)
# ========================
y_train = np.array(y_train, dtype="float32")
y_test = np.array(y_test, dtype="float32")

# Periksa dimensi array untuk konfirmasi
print(f"Shape X_train_pad: {X_train_pad.shape}")
print(f"Shape X_train_tfidf: {X_train_tfidf.shape}")
print(f"Shape y_train: {y_train.shape}")
print(f"Shape y_test: {y_test.shape}")

# ====================================
# MEMBANGUN MODEL BiLSTM BERBASIS TF-IDF (MODEL MULTIMODAL)
# ====================================

# --- Input 1: Sequence (Untuk BiLSTM) ---
input_seq = Input(shape=(MAX_LEN,), name='input_sequence')
x = Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN)(input_seq)
x = Bidirectional(LSTM(64))(x)
# Output BiLSTM (Vektor konteks)

# --- Input 2: Vektor TF-IDF ---
input_tfidf = Input(shape=(TFIDF_FEATURES,), name='input_tfidf')
# TF-IDF bisa langsung dimasukkan ke lapisan Dense atau langsung digabungkan
y = Dense(32, activation='relu')(input_tfidf)

# --- Gabungan (Concatenate) ---
# Menggabungkan output dari BiLSTM (x) dan output dari TF-IDF (y)
combined = Concatenate()([x, y])

# --- Output Layer ---
z = Dense(64, activation="relu")(combined)
output_layer = Dense(1, activation="sigmoid")(z)

# Definisikan Model dengan dua input
model = Model(inputs=[input_seq, input_tfidf], outputs=output_layer)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model([X_train_pad[:1], X_train_tfidf[:1]]) 
print(model.summary())

# ========================
# Definisi Callback
# ========================
# Hentikan pelatihan jika val_loss tidak membaik setelah 3 epoch (patience=3)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True, # <== Simpan bobot terbaik
    verbose=1
)

# ========================
# Training
# ========================
history = model.fit(
    [X_train_pad, X_train_tfidf],
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# ========================
# Evaluasi
# ========================
loss, acc = model.evaluate([X_test_pad, X_test_tfidf], y_test, verbose=1)
print(f"\nAkurasi Test (BiLSTM + TF-IDF): {acc:.4f}")

# ========================
# Simpan model
# ========================
os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print("Model saved to:", MODEL_SAVE_PATH)
