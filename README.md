# CRP Scatter Estimation and Analysis Dashboard

A professional data analytics tool designed to visualize and predict C-Reactive Protein (CRP) levels based on nutritional intake data from the NHANES (2017-2023) dataset.

## 📊 Overview
This dashboard provides a multivariate analysis of how dietary components correlate with systemic inflammation (CRP). It features a population-level explorer and an individual regression-based predictor.

## 🤖 Algorithm
The tool utilizes a **Random Forest Regressor** (`scikit-learn` implementation) for its analytical engine.
- **Ensemble Learning**: Uses 200 decision trees to ensure stable and robust predictions.
- **Multivariate Analysis**: Captures complex, non-linear relationships between multiple nutritional variables and inflammation markers.
- **Classification**: Derived from the regression estimate using a population mean threshold (3.863 mg/L) to categorize results into "High" or "Low" inflammation status.

## 🧬 Features (All in Grams)
The dataset has been refined and engineered into 6 key analytical features:
1.  **Proteins**: Total protein intake.
2.  **Carbs**: Total carbohydrate intake.
3.  **Fibre**: Dietary fiber intake.
4.  **Fats**: Total fats (Saturated + Monounsaturated + Polyunsaturated).
5.  **Water**: Total moisture/water weight from food and beverages.
6.  **Micronutrients**: A composite mass sum of sugars, cholesterol, vitamins, and minerals.

## 🚀 Installation & Local Run
1. **Clone the repository** (or download the files).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

## 🌐 Deployment
This project is optimized for deployment on **Streamlit Community Cloud** or **Hugging Face Spaces**. 
Ensure the `dataset/` folder and `requirements.txt` are included in your repository to allow the remote server to build the environment and load the reference data.

## 📚 Data Source
The model is trained on the [NHANES CVD Raw Data 2017-23](https://www.kaggle.com/datasets/ahiduzzaman28/nhanes-cvd-raw-data-2017-23) dataset available on Kaggle.
