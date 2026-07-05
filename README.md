# shopper_spectrum_analysis
# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations

An end-to-end data science project analyzing e-commerce transactions to uncover purchasing patterns, segment customers using RFM analysis, and build a collaborative filtering product recommendation system.

## 🚀 Key Features
*   **Customer Segmentation:** Groups customers into 4 actionable clusters (High-Value, Regular, Occasional, At-Risk) using RFM metrics and K-Means clustering.
*   **Product Recommendation Engine:** Item-based collaborative filtering using Cosine Similarity to recommend the top 5 related products based on purchase history.
*   **Interactive Web Dashboard:** A Streamlit application featuring EDA visualizations, a live recommendation engine, and a real-time customer segment predictor.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Data Processing:** Pandas
*   **Machine Learning:** Scikit-Learn (K-Means, StandardScaler, Cosine Similarity)
*   **Visualization:** Matplotlib, Seaborn
*   **Deployment:** Streamlit, Joblib

## 📂 Project Structure
*   `data/online_retail.csv`: The raw transaction dataset.
*   `shopper_spectrum_eda_modeling.ipynb`: Jupyter Notebook containing data cleaning, exploratory data analysis (EDA), RFM engineering, and model training.
*   `models/kmeans_rfm_model.pkl`: Saved K-Means clustering model.
*   `app.py`: Streamlit web application script.

## ⚙️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone <your-github-repo-url>
   cd shopper-spectrum
