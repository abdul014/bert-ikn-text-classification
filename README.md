<h1 align="center">🤖 BERT Text Classification — Sentiment Analysis on Ibu Kota Nusantara (IKN)</h1>

<p align="center">
  <img src="https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcR9ATm2P7Q0VnITMM8OCWfHl2illHgcHgEicA8pBaFOWttEYi6-2QQj4jBedBrWv35ZcDlNCZBN_s6V0JAwse4t837_AIe6Hp-vo_oZ0vg" width="700" alt="BERT IKN Banner">
</p>

<p align="center">
  <b>Fine-tuned BERT model for sentiment classification of news and narratives about Indonesia’s new capital city — Ibu Kota Nusantara (IKN).</b><br>
  Built using <code>cahya/bert-base-indonesian-1.5G</code> and TensorFlow 2.x.
</p>

---

## 📘 Project Overview
This project implements a **text classification model** using **BERT for Bahasa Indonesia** to analyze public sentiment toward the **Ibu Kota Nusantara (IKN)** development project.  
The model classifies texts into **three sentiment categories**:
- 🟩 **Positive**
- 🟨 **Neutral**
- 🟥 **Negative**

The dataset was generated from narratives discussing IKN’s **history**, **initiators**, **development process**, and **challenges**.  
The model is fine-tuned using TensorFlow and achieves strong performance on both training and validation datasets.

---

---

## ⚙️ Workflow Summary
1. **Data Preparation**  
   The corpus consists of Indonesian sentences about IKN. Each sentence was labeled automatically based on keyword matching using lists of positive and negative words.
2. **Tokenization**  
   Tokenization was done using `BertTokenizer` from `transformers`.
3. **Model Training**  
   Fine-tuning was performed with the following setup:
   ```python
   optimizer = tf.keras.optimizers.Adam(learning_rate=2e-7, epsilon=1e-8)
   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
   metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
   ```
4. **Evaluation & Visualization**  
   Accuracy and loss were tracked throughout 10 epochs.

---

## 🧩 Dataset Example
| Sentence | Label |
|-----------|--------|
| “Ibu Kota Nusantara is a monumental project aimed at relocating Indonesia’s capital to reduce overpopulation in Jakarta.” | Positive |
| “The project faces major environmental and funding challenges.” | Negative |
| “The planning phase of IKN started in 2019.” | Neutral |

Dataset split:
- **Training set:** 80%
- **Testing set:** 20%

---

## 📈 Training Results

| Metric | Training | Validation |
|--------|-----------|-------------|
| Accuracy | **94.0%** | **85.71%** |
| Loss | **0.1309** | **0.6575** |

---

### 🧾 Model Accuracy & Loss Graphs

<p align="center">
  <img src="https://github.com/abdul014/bert-ikn-text-classification/blob/main/resulst.png?raw=true" width="700" alt="Model Accuracy and Loss">
</p>

The graphs above visualize the training and validation performance over 10 epochs:  
- The **accuracy curve** shows steady improvement, reaching ~94% on training and 85.7% on validation.  
- The **loss curve** indicates consistent convergence with a stable validation loss (~0.65).

---

## 🧮 Sentiment Prediction Example
```python
sentence = "IKN aims to create a smart and green city with modern infrastructure."
predict_sentence(sentence, tokenizer, model)
# Output: Positive (2)
```

**Prediction Distribution:**
| Label | Meaning | Percentage |
|--------|----------|-------------|
| 0 | Negative | 34.45% |
| 1 | Neutral | 34.45% |
| 2 | Positive | 31.25% |

🧭 Overall Sentiment Result → **Negative**

---

---

## 🧰 Requirements
```bash
tensorflow==2.15.0
transformers==4.36.2
pandas==2.1.3
scikit-learn==1.3.2
matplotlib==3.8.2
```

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/abdul014/bert-ikn-text-classification.git
   cd bert-ikn-text-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook or Python script:
   ```bash
   python ikn_sentiment_classification.py
   ```

---

## 🧾 Folder Structure
```
bert-ikn-text-classification/
├── data/
│   └── ikn_sentences.csv
├── models/
│   └── bert_ikn_finetuned.h5
├── images/
│   ├── model_accuracy_loss.png
│   └── ikn_sentiment_chart.png
├── notebooks/
│   └── IKN_TextClassification.ipynb
└── README.md
```

---

## 🧪 Key Insights
- BERT successfully captures contextual nuances in Bahasa Indonesia.
- Majority of IKN-related texts carry **negative** or **neutral** sentiments.
- The model demonstrates **strong generalization** with minimal overfitting.  

> 🎯 This demonstrates how pre-trained language models can effectively analyze sociopolitical discourse in local languages.

---

## 👨‍💻 Author
**Abd Rahman**  
_M.Sc. Student in Statistics & Data Science, IPB University_  
📧 [GitHub](https://github.com/abdul014) • [LinkedIn](https://linkedin.com/in/abd-rahman-ysf)

---

## 📄 License
This project is licensed under the **MIT License** — feel free to use, modify, and share with proper attribution.
