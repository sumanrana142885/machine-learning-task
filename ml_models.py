import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Paths
DATA_PATH = '/Users/sumankumarrana/Public/Dataset.csv'
MODELS_DIR = '/Users/sumankumarrana/Public/models/'
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. DATA LOADING & INITIAL PREPROCESSING
# -------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Handle missing cuisines
df['Cuisines'] = df['Cuisines'].fillna('Unknown')

# Binary encoding for services
df['Has Table booking_bin'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
df['Has Online delivery_bin'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

# Extracted/Engineered features
df['Cuisine Count'] = df['Cuisines'].apply(lambda x: len([c.strip() for c in x.split(',')]))
df['Name Length'] = df['Restaurant Name'].apply(lambda x: len(str(x)))
df['Address Length'] = df['Address'].apply(lambda x: len(str(x)))
df['Cost per Person'] = df['Average Cost for two'] / 2.

# Primary cuisine extraction
df['Primary Cuisine'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())

print(f"Data shape: {df.shape}")

# -------------------------------------------------------------
# 2. LEVEL 1 - TASK 1: RATING REGRESSION
# -------------------------------------------------------------
print("\n--- Level 1, Task 1: Predictive Modeling (Regression) ---")

# Define features for rating prediction
# We exclude Country Code if it is constant, but keep it if not.
# We also include Votes, Cost, booking, delivery, price range, cuisine count
features_reg = ['Average Cost for two', 'Has Table booking_bin', 'Has Online delivery_bin', 'Price range', 'Votes', 'Cuisine Count', 'Latitude', 'Longitude']
X_reg = df[features_reg]
y_reg = df['Aggregate rating']

# Train-Test Split (80/20)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Models to experiment with
reg_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

reg_results = {}
best_reg_name = None
best_reg_r2 = -1
best_reg_model = None

for name, model in reg_models.items():
    model.fit(X_train_reg, y_train_reg)
    preds = model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, preds)
    reg_results[name] = {'MSE': mse, 'RMSE': rmse, 'R-squared': r2}
    print(f"{name} -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    if r2 > best_reg_r2:
        best_reg_r2 = r2
        best_reg_name = name
        best_reg_model = model

print(f"Best Regressor: {best_reg_name} with R-squared: {best_reg_r2:.4f}")

# Serialize best regression model
with open(os.path.join(MODELS_DIR, 'rating_regressor.pkl'), 'wb') as f:
    pickle.dump(best_reg_model, f)
print("Saved best rating regressor.")

# -------------------------------------------------------------
# 3. LEVEL 1 - TASK 2: RECOMMENDATION SYSTEM (Content-Based)
# -------------------------------------------------------------
print("\n--- Level 1, Task 2: Recommendation System (TF-IDF Cuisines + Cosine Similarity) ---")
# Pre-calculate content similarity based on Cuisines, City, and Locality
df['Content Text'] = df['Cuisines'] + " " + df['City'] + " " + df['Locality']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Content Text'])

# Serialize tfidf and process matrix for fast load
with open(os.path.join(MODELS_DIR, 'tfidf_recommend.pkl'), 'wb') as f:
    pickle.dump((tfidf, tfidf_matrix), f)
print("Saved recommendation vectorizer and matrix.")

# -------------------------------------------------------------
# 4. LEVEL 2 - TASK 1: CUISINE CLASSIFICATION
# -------------------------------------------------------------
print("\n--- Level 2, Task 1: Cuisine Classification ---")
# We predict the primary cuisine. To avoid extreme class imbalance (hundreds of rare cuisines),
# we focus on the top 15 most frequent cuisines.
top_cuisines = df['Primary Cuisine'].value_counts().head(15).index.tolist()
df_class = df[df['Primary Cuisine'].isin(top_cuisines)].copy()

features_class = ['Average Cost for two', 'Has Table booking_bin', 'Has Online delivery_bin', 'Price range', 'Aggregate rating', 'Votes', 'Latitude', 'Longitude']
X_class = df_class[features_class]
y_class = df_class['Primary Cuisine']

le_cuisine = LabelEncoder()
y_class_enc = le_cuisine.fit_transform(y_class)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_class, y_class_enc, test_size=0.2, random_state=42)

# Models to experiment with
class_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (Linear)': LinearSVC(random_state=42)
}

cls_results = {}
best_cls_name = None
best_cls_acc = -1
best_cls_model = None

for name, model in class_models.items():
    model.fit(X_train_cls, y_train_cls)
    preds = model.predict(X_test_cls)
    acc = accuracy_score(y_test_cls, preds)
    cls_results[name] = acc
    print(f"{name} Classifier -> Accuracy: {acc:.4f}")
    
    if acc > best_cls_acc:
        best_cls_acc = acc
        best_cls_name = name
        best_cls_model = model

print(f"Best Classifier: {best_cls_name} with Accuracy: {best_cls_acc:.4f}")

# Serialize classifier and LabelEncoder
with open(os.path.join(MODELS_DIR, 'cuisine_classifier.pkl'), 'wb') as f:
    pickle.dump((best_cls_model, le_cuisine, top_cuisines), f)
print("Saved cuisine classifier.")

# -------------------------------------------------------------
# 5. LEVEL 2 - TASK 2: GEOGRAPHIC CLUSTERING
# -------------------------------------------------------------
print("\n--- Level 2, Task 2: Geographic Clustering (K-Means) ---")
# Filter out valid coords
coords = df[['Latitude', 'Longitude']]
# We choose K = 8 clusters for identifying dining hubs
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df['Geo Cluster'] = kmeans.fit_predict(coords)

with open(os.path.join(MODELS_DIR, 'kmeans_clustering.pkl'), 'wb') as f:
    pickle.dump(kmeans, f)
print("Saved KMeans clustering model.")

# -------------------------------------------------------------
# 6. LEVEL 3 - TASK 1: SENTIMENT ANALYSIS (Custom Sentiment Classifier)
# -------------------------------------------------------------
print("\n--- Level 3, Task 1: Sentiment Analysis on Text Reviews ---")
# Since the dataset does not have actual reviews text, we generate realistic synthetic review sentences
# mapping to the rating texts (Excellent/Very Good/Good/Average/Poor/Not rated)
sentiment_mapping = {
    'Excellent': 'positive',
    'Very Good': 'positive',
    'Good': 'positive',
    'Average': 'neutral',
    'Poor': 'negative',
    'Not rated': 'neutral'
}

df['Sentiment_Label'] = df['Rating text'].map(sentiment_mapping)

# Templates for synthetic reviews to train a text classifier
templates = {
    'Excellent': [
        "Absolutely amazing! The food was delicious and service was outstanding.",
        "Best restaurant in town! Loved the atmosphere and prompt service.",
        "Incredible flavors, gorgeous presentation, and excellent customer service.",
        "Perfect dining experience! Everything was top notch.",
        "Highly recommended! The chef is brilliant and the food is superb."
    ],
    'Very Good': [
        "Very good quality food and friendly staff.",
        "Great dining experience. Nice ambiance and tasty dishes.",
        "Really liked the food. Service was good and clean place.",
        "Solid menu options, nice service, will visit again.",
        "Delicious food and quick service. A great place to eat."
    ],
    'Good': [
        "Good food, decent price, and friendly waiters.",
        "Enjoyed the meal. The atmosphere was pleasant.",
        "Nice restaurant. The experience was good overall.",
        "Good taste and clean environment. Worth trying.",
        "Tasty dishes, polite staff, and good service."
    ],
    'Average': [
        "Average food. Nothing special or outstanding.",
        "Ambiance was okay, but the service was slow.",
        "Decent taste but the prices are slightly high.",
        "Standard experience. Neither good nor bad.",
        "The food was average, and the wait time was long."
    ],
    'Poor': [
        "Terrible service and cold food. Very disappointed.",
        "Bad taste, dirty tables, and rude staff.",
        "Extremely overpriced and worst customer service.",
        "Poor quality ingredients. Will never recommend.",
        "Waste of money. Horrible experience overall."
    ],
    'Not rated': [
        "Average experience, nothing notable to say.",
        "Just a normal restaurant, ok food.",
        "Not rated yet. Food is decent, service is ok.",
        "Ordinary dining. Clean place but generic taste.",
        "Simple cafeteria, average service and food."
    ]
}

# Generate synthetic reviews
np.random.seed(42)
synthetic_reviews = []
for rating_text in df['Rating text']:
    opts = templates[rating_text]
    synthetic_reviews.append(np.random.choice(opts))

df['Synthetic Review'] = synthetic_reviews

# Train TF-IDF + Logistic Regression
X_text = df['Synthetic Review']
y_sent = df['Sentiment_Label']

text_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_text_tfidf = text_tfidf.fit_transform(X_text)

X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(X_text_tfidf, y_sent, test_size=0.2, random_state=42)

sent_classifier = LogisticRegression(max_iter=1000, random_state=42)
sent_classifier.fit(X_train_txt, y_train_txt)
sent_preds = sent_classifier.predict(X_test_txt)

print(f"Sentiment Classifier Accuracy: {accuracy_score(y_test_txt, sent_preds):.4f}")
print("Sentiment Report:\n", classification_report(y_test_txt, sent_preds))

with open(os.path.join(MODELS_DIR, 'sentiment_classifier.pkl'), 'wb') as f:
    pickle.dump((sent_classifier, text_tfidf), f)
print("Saved sentiment classifier and TF-IDF vectorizer.")

# -------------------------------------------------------------
# 7. SAVE PROCESSED DATASET FOR WEB APP
# -------------------------------------------------------------
# Save df to pickle for Streamlit app to load directly
with open(os.path.join(MODELS_DIR, 'processed_data.pkl'), 'wb') as f:
    pickle.dump(df, f)
print("Saved preprocessed dataset.")

# Save evaluation results for dashboard display
evaluation_metrics = {
    'reg_results': reg_results,
    'cls_results': cls_results,
    'best_reg_name': best_reg_name,
    'best_cls_name': best_cls_name
}
with open(os.path.join(MODELS_DIR, 'evaluation_metrics.pkl'), 'wb') as f:
    pickle.dump(evaluation_metrics, f)
print("Saved model evaluation metrics.")

print("\nAll models trained and serialized successfully!")
