
# 🎌 Anime Recommendation System

A Machine Learning-powered system designed to deliver **personalized anime recommendations** by combining collaborative, content-based, and clustering techniques. This hybrid model enhances user engagement by recommending shows based on preferences, popularity, and narrative themes.

---

## 📌 Table of Contents

* [📖 Overview](#-overview)
* [🧠 Features](#-features)
* [📊 Tech Stack & Tools](#-tech-stack--tools)
* [📁 Datasets](#-datasets)
* [🔍 Methodology](#-methodology)
* [🚀 Implementation Highlights](#-implementation-highlights)
* [📈 Model Evaluation](#-model-evaluation)
* [🧪 How to Run](#-how-to-run)
* [📌 Future Improvements](#-future-improvements)
* [📚 References](#-references)

---

## 📖 Overview

This project implements a hybrid anime recommendation system using various **machine learning** and **natural language processing** techniques. It provides personalized recommendations by analyzing:

* User preferences
* Anime metadata (genre, studio, type)
* Narrative content via **DistilBERT embeddings**
* Community popularity metrics


---

## 🧠 Features

✅ Hybrid Recommendation System
✅ Collaborative Filtering using KNN
✅ Content-Based Filtering using `DistilBERT` and metadata
✅ Clustering via K-Means & Agglomerative methods
✅ Popularity-Based Filtering
✅ Text preprocessing & BERT-based embeddings
✅ Cross-validation & hyperparameter tuning

---

## 📊 Tech Stack & Tools

* **Languages**: Python
* **ML Libraries**: scikit-learn, pandas, NumPy, seaborn, matplotlib
* **NLP**: `transformers`, `DistilBERT`, `BERT`
* **Vectorization**: CountVectorizer, Cosine Similarity
* **Clustering**: K-Means, Agglomerative Clustering
* **Recommendation Techniques**:

  * Collaborative Filtering (KNN)
  * Content-Based Filtering (BERT, DistilBERT)
  * Popularity-Based Filtering

---

## 📁 Datasets

1. **anime.csv** – Contains anime ID, name, genre, type, episode count, rating, and popularity
2. **anime\_dataset.csv** – Used for collaborative filtering with user\_id and rating info
3. **popular\_anime.csv** – Anime popularity metadata based on ratings and views

---

## 🔍 Methodology

* **Data Preprocessing**:

  * Null handling, typecasting, vector encoding
  * Stopword removal and synopsis cleaning
* **Filtering Techniques**:

  * Popularity-based for cold-start users
  * KNN for behavior-based similarity
  * BERT/DistilBERT for deep content understanding
* **Clustering**:

  * K-Means for genre-based groups
  * Agglomerative for hierarchical theme classification
* **Hyperparameter Optimization**: Grid/Random Search
* **Model Evaluation**: Precision\@K, Recall\@K, F1-Score, RMSE, MAP

---

## 🚀 Implementation Highlights

* Anime descriptions vectorized using **DistilBERT** to understand narrative themes
* Clustering used to group similar genres and themes
* Collaborative filtering based on cosine similarity of user ratings
* Tag-based CountVectorizer + Cosine Similarity for content matching
* Models trained and evaluated with k-fold cross-validation
* Visual performance charts (ROC, Confusion Matrix, Bar Graphs)

---

## 🧪 How to Run

1. **Clone this repo**

   ```bash
   git clone https://github.com/yourusername/anime-recommendation-system.git
   cd anime-recommendation-system
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook or script**

   * Use Jupyter or VS Code to run `main.ipynb`
   * Ensure data files (`anime.csv`, `anime_dataset.csv`, `popular_anime.csv`) are in the working directory

4. **Get Recommendations**

   * Input an anime name to receive top-k similar recommendations

---

## 📌 Future Improvements

* Add user demographics and watch history
* Real-time recommendation deployment via Flask or FastAPI
* Integrate external anime APIs (e.g., MyAnimeList, AniList)
* Improve efficiency of large-scale BERT inference
* Build a simple web UI with React or Streamlit

---

## 📚 References

* Koren et al., *Matrix Factorization for Recommender Systems*
* Bansal & Gupta, *Content-based recommendation for anime*
* Yoon & Song, *Hybrid recommendation for anime*
* Transformers Library by HuggingFace
* MyAnimeList datasets

---

> 👨‍💻 Developed by Akshat Pratap Singh and Team | VIT-AP | 2024-25
> 📩 For queries or collaborations, feel free to reach out!

