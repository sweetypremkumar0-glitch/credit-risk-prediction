# credit-risk-prediction
An end-to-end Machine Learning pipeline on the German Credit Dataset to predict defaulter risk. Handled severe class imbalance using SMOTE to optimize minority recall. Built a Voting Ensemble (Gradient Boosting + Random Forest + Logistic Regression) and deployed it via a live Streamlit dashboard.

# Credit Risk Analysis & Prediction System

An end-to-end Machine Learning pipeline built on the German Credit Dataset to predict financial defaulter risk and assist stakeholders with real-time risk scoring.

## 🚀 Key Features & Pipeline
* **Feature Engineering:** Systematically selected 9 high-impact features from 20+ raw attributes using feature importance analysis.
* **Handling Class Imbalance:** Applied Synthetic Minority Over-sampling Technique (SMOTE) to handle severe class imbalance, successfully optimizing minority-class recall and mitigating false negatives on potential defaulters.
* **Ensemble Modeling:** Developed a Voting Ensemble Classifier combining Gradient Boosting, Random Forest, and Logistic Regression, achieving a realistic 72% evaluation accuracy benchmark.
* **Interactive UI:** Wrapped the trained ensemble model into an interactive Streamlit dashboard for real-time customer risk visualization.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-learn, SMOTE (Imbalanced-learn)
* **Web Framework:** Streamlit
* **Data Analysis:** Pandas, NumPy

## 📂 Project Structure
* `app.py` / `main.py` - Streamlit application deployment file
* `model.py` / `notebook.ipynb` - Data preprocessing, training, and model evaluation pipeline
* `requirements.txt` - Python project dependencies

## 📊 Results & Business Impact
By prioritizing minority-class recall using SMOTE, this system directly aligns with real-world credit risk business logic—ensuring that high-risk applicants are accurately flagged, thereby reducing potential financial credit loss.
