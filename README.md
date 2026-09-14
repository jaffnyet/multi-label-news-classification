# 📰 Multi-Label News Classification: Handling Long-Tail Distributions with DistilBERT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4)

## 📌 Project Overview
This project tackles the challenge of **Multi-Label Text Classification** on highly imbalanced datasets. By combining the classic **Reuters-21578** financial corpus with the diverse **Argilla Synthetic News** dataset, we classify news articles into 97 distinct categories. 

The core challenge of this dataset is its severe **long-tail distribution**—where a few categories dominate the data, leaving most classes with very few examples.

## 🚀 The Challenge & Solution
Standard Binary Cross-Entropy (BCE) loss treats all classes equally, which causes deep learning models to ignore rare labels (resulting in a poor Macro F1-Score). 

To solve this, we implemented and compared custom loss functions in PyTorch:
1. **TF-IDF + Logistic Regression** (Baseline)
2. **DistilBERT with Default BCE Loss**
3. **DistilBERT with Weighted BCE Loss** (Scales the loss gradient based on the inverse frequency of the class)
4. **DistilBERT with Asymmetric Loss (ASL)** (Down-weights the penalty of easy negative samples, forcing the model to focus on rare positive labels)

## 📊 Final Optimized Performance
We optimized the inference thresholds for each model to maximize the Macro F1-score. The results show that **Asymmetric Loss (ASL)** significantly outperforms the default BCE, achieving a **+14% improvement in Macro F1-Score**.

| Metric | TF-IDF + LogReg | DistilBERT (Default BCE) | DistilBERT (Weighted BCE) | DistilBERT (Asymmetric Loss) |
| :--- | :---: | :---: | :---: | :---: |
| **Optimal Threshold** | 0.50 | 0.15 | 0.80 | **0.45** |
| **Micro F1-Score** | 0.8426 | 0.8892 | 0.8888 | **0.9058** |
| **Macro F1-Score** | 0.5393 | 0.4735 | 0.5534 | **0.6143** |
| **Precision** | 0.7720 | 0.8595 | 0.8553 | **0.8702** |
| **Recall** | 0.9273 | 0.9210 | 0.9251 | **0.9445** |

## 💻 Interactive Dashboard (Gradio)
We wrapped the fine-tuned modeling pipeline into an interactive **Gradio web application**. Users can paste raw news wires into the UI and watch the four models classify the text in real-time.
> `Gradio_Demo.png`

## 🛠️ Tech Stack
* **Modeling:** PyTorch, Hugging Face `transformers`, `datasets`
* **Machine Learning:** Scikit-learn (TF-IDF, Logistic Regression)
* **Frontend/UI:** Gradio
* **Data Processing:** Pandas, NumPy, NLTK

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/](https://github.com/)jaffnyet/multi-label-news-classification.git
cd multi-label-news-classification