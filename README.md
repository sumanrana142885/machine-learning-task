# Zomato Restaurant Machine Learning Hub

An end-to-end Machine Learning project developed for the **Cognifyz Technologies Machine Learning Development Internship**. This repository implements a series of predictive modeling, recommendation engine, cuisine classification, geospatial clustering, and natural language sentiment analysis tasks using the Zomato Restaurant Dataset.

An interactive, high-fidelity dark-themed **Streamlit Web Dashboard** is provided to demonstrate the real-time inference of these models.

---

## 🚀 Live Dashboard Demonstration
The dashboard provides a tabbed, visually rich environment containing:
1. **Performance Overview:** Comparison charts showing model accuracies and R² scores.
2. **Ratings & Recommendations:** Real-time restaurant aggregate rating predictor and content-based recommendation filter using Cosine Similarity.
3. **Cuisines & Clusters:** A Random Forest classifier to predict restaurant cuisines and interactive spatial mapping using K-Means coordinates clustering.
4. **Sentiment & Feature Engineering:** A TF-IDF Naive Bayes/Logistic Regression classifier to predict review sentiments and evaluations of engineered features.

---

## 📈 Model Performance & Benchmarks

### 1. Restaurant Rating Prediction (Regression)
- **Random Forest Regressor:** **R² = 0.9582** (Selected Model)
- **Decision Tree Regressor:** R² = 0.9187
- **Linear Regression:** R² = 0.2773

### 2. Cuisine Classification (Classification)
- **Random Forest Classifier:** **Accuracy = 0.3950** (Predicting the top 15 primary cuisines)

### 3. Review Sentiment Analysis (NLP Classification)
- **TF-IDF + Logistic Regression:** **Accuracy = 1.0000** (Trained on preprocessed text reviews matched to rating categories)

---

## 📁 Repository Structure
```bash
├── Dataset.csv             # Raw Zomato Restaurant dataset (9,551 records)
├── ml_models.py            # Preprocessing, training, and pickle model serialization script
├── app.py                  # Streamlit Web App dashboard source code
├── internship_tasks.ipynb  # Comprehensive Jupyter Notebook submission containing outputs and charts
├── internship_tasks.html   # Standalone HTML export of the executed Jupyter Notebook
├── models/                 # Serialized pickle files for trained models
│   ├── rating_regressor.pkl
│   ├── cuisine_classifier.pkl
│   ├── kmeans_clustering.pkl
│   ├── sentiment_classifier.pkl
│   ├── tfidf_recommend.pkl
│   ├── processed_data.pkl
│   └── evaluation_metrics.pkl
└── README.md               # Repository documentation
```

---

## 🛠️ Setup and Installation

### Prerequisites
Make sure you have Python 3.8+ installed. You can install all required libraries using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit nltk
```

### Steps to Run the Project

1. **Preprocess and Train Models:**
   Run the machine learning pipeline script to process the dataset, benchmark models, and save them:
   ```bash
   python ml_models.py
   ```

2. **Launch the Streamlit Web Application:**
   Run the local web dashboard:
   ```bash
   streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501`.

3. **Open the Submission Notebook:**
   Explore the step-by-step notebook execution:
   ```bash
   jupyter notebook internship_tasks.ipynb
   ```
