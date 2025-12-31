# Indonesian Political Hoax Detection

### Performance Evaluation of BiLSTM, LSTM, and Random Forest

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-green?logo=scikit-learn)


## Project Overview

The escalation of political disinformation on social media poses a serious challenge to democratic stability. This project presents a **Hybrid Deep Learning** approach for detecting Indonesian political hoaxes with high precision.

The proposed method uses a **multi-input feature concatenation** strategy that integrates:

1. **Contextual Features** – sequential text representations processed using **LSTM / BiLSTM**.
2. **Lexical Features** – keyword-based importance extracted using **TF-IDF**.

Model performance is evaluated against a **Random Forest baseline** using a curated dataset consisting of **20,890 Indonesian political news samples**.

---

## Key Features and Methodology

* **Hybrid Architecture**: Combines contextual understanding from RNN-based models (LSTM/BiLSTM) with statistical text representations from TF-IDF vectors using a `Concatenate` layer.
* **Regularization Strategy**: Applies **L2 Regularization** and **Dropout (0.5)** to mitigate overfitting.
* **Optimization**: Trained using the **AdamW** optimizer with a low learning rate (`1e-5`) to ensure stable convergence.
* **Data Handling**: Addresses class imbalance through **Random Oversampling** and monitors training using **Early Stopping**.

---

## Experimental Results

The **Hybrid BiLSTM** model demonstrates the strongest overall performance, particularly in terms of precision, indicating a reduced risk of misclassifying factual news as hoaxes.

| Model                    | Accuracy   | Precision  | Recall | F1-Score   |
| :----------------------- | :--------- | :--------- | :----- | :--------- |
| Random Forest (Baseline) | 97.13%     | 75.37%     | 93.87% | 0.8361     |
| Hybrid LSTM              | 97.61%     | 83.04%     | 87.12% | 0.8503     |
| Hybrid BiLSTM (Proposed) | **98.52%** | **91.77%** | 88.96% | **0.9034** |

**Key Finding**: The BiLSTM-based hybrid model improves precision by approximately **16%** compared to the Random Forest baseline, highlighting the importance of bidirectional contextual modeling in political hoax detection.

---

## Project Structure

```bash
├── dataset/
│   ├── raw/                # Original Kaggle dataset (31k samples)
│   └── processed/          # Curated and split data (Train/Val/Test)
├── models/
│   ├── bilstm_model.keras  # Saved BiLSTM model
│   ├── lstm_model.keras    # Saved LSTM model
│   └── rf_model.pkl        # Saved Random Forest model
├── notebooks/
│   ├── 01_Preprocessing.ipynb       # Cleaning, stopwords, stemming
│   ├── 02_Feature_Engineering.ipynb # TF-IDF and tokenizer/padding
│   ├── 03_Modeling_BiLSTM.ipynb     # BiLSTM + TF-IDF training
│   ├── 04_Modeling_LSTM.ipynb       # LSTM + TF-IDF training
│   └── 05_Modeling_RF.ipynb         # Random Forest baseline
└── README.md
```

---

## Contributors

* [Calista Salsabila](https://github.com/calistasalsabila)
* [Aprillia Wulandari](https://github.com/aprliwln)
* [Saskya Aliya Azizah](https://github.com/askyaya31)
* [Baihaqi Hakim Abdullah](https://github.com/baihaqihakim77)
