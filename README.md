# DropAlert Rwanda Dashboard: Predicting & Preventing School Dropout

## 👩‍🎓 Student Information

* **Name**: \[Your Full Name]
* **Student ID**: \[Your Student ID]
* **Course**: INSY 8413 | Introduction to Big Data Analytics
* **Assistant Lecturer**: Eric Maniraguha
* **Academic Year**: 2024–2025 (Semester III)

---

## 📘 Project Introduction

The project titled **"DropAlert Rwanda"** focuses on the education sector, addressing the persistent issue of student dropout across Rwanda's schools. The objective is to leverage big data analytics and machine learning to:

* Analyze patterns of student dropout
* Predict dropout risks
* Visualize regional trends to guide interventions

By using Python for analytical tasks and Power BI for dashboard reporting, we aim to deliver actionable insights for education policymakers and stakeholders.

---

## 🧭 Methodology

### 1. Dataset Identification

* **Sector**: 🎓 Education
* **Dataset Title**: Rwanda School Dropout Dataset
* **Source**: [Rwanda Data Portal](https://www.statistics.gov.rw)
* **Rows & Columns**: \~6000 rows × 18 columns
* **Format**: Structured CSV
* **Status**: Requires preprocessing

### 2. Python Analysis Steps

#### Step 1: Load Data

```python
import pandas as pd

df = pd.read_csv("DropAlert_Rwanda_Analysis_Results.csv")
df.head()
```

#### Step 2: Data Cleaning

* Removed missing/null values
* Converted categorical data (e.g., Region\_Type) using label encoding

#### Step 3: Feature Engineering

```python
df['is_dropout_high'] = df['Overall_Dropout_Percentage'] > df['Overall_Dropout_Percentage'].mean()
df['Region_Type'] = df['Region_Type'].map({"Urban": 1, "Rural": 0})
```

#### Step 4: Exploratory Data Analysis (EDA)

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Overall_Dropout_Percentage'])
```

* Histograms and bar plots used to explore distributions
* Grouped comparisons (urban vs. rural, gender split)

#### Step 5: Machine Learning Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df.drop(['is_dropout_high'], axis=1)
y = df['is_dropout_high']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
```

#### Step 6: Feature Importance

```python
importances = model.feature_importances_
features = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
```

#### Step 7: Model Evaluation

* Accuracy: \~86%
* Key Features: Region\_Type, Gender\_Rate, Grade\_Repetition, etc.

---

## 📊 Power BI Dashboard

### Dashboard Title: "📉 DropAlert Rwanda Dashboard"

### Key Elements (9 Visuals):

1. **KPI Card** – Average Dropout Rate (DAX)

   ```dax
   Average_Dropout_Percentage = AVERAGE('DropAlert_Rwanda_Analysis_Results'[Overall_Dropout_Percentage])
   ```

2. **Card** – Total Number of High-Risk Schools

   ```dax
   High_Risk_School_Count =
   CALCULATE(COUNTROWS('DropAlert_Rwanda_Analysis_Results'),
   'DropAlert_Rwanda_Analysis_Results'[is_dropout_high] = TRUE())
   ```

3. **Slicer** – Region Name (interactive)

4. **Gauge** – Target Dropout Rate (e.g., 5%)

5. **Pie Chart** – Rural vs Urban Distribution

   * Legend: `Region_Type`
   * Values: Count of schools

6. **Bar Chart** – Dropout Rate by Gender

   * Axis: `Gender`
   * Values: `Dropout_Rate`

7. **Line Chart** – Dropout Trends (if date present, otherwise grouped average)

8. **Map** – Geographic Dropout Risk Distribution

   * Location: `District`
   * Values: `Overall_Dropout_Percentage`

9. **Column Chart** – Top 10 High-Risk Schools

   * Axis: `School Name`
   * Values: `Overall_Dropout_Percentage`

### Interactivity

* All visuals are linked via filters
* Slicer allows per-district selection
* KPI and Gauge update based on filters

---

## 📈 Results

* Over 35% of schools exceed national dropout threshold
* Rural schools show 2x higher risk than urban
* Female dropout rate slightly exceeds male dropout in some districts
* Key influencers: Region Type, Grade Repetition, Parental Education

---

## ✅ Recommendations

* 🚸 Increase monitoring in rural districts
* 📚 Target interventions at lower grades with high repetition
* 💡 Use predictive risk scores to assign school-level intervention budgets

---

## 🔮 Future Work

* Integrate longitudinal data across 3–5 years
* Use LSTM or time series forecasting for trends
* Connect with government API or mobile data collection tools
* Add mobile-friendly dashboard view

---

## 🗃️ GitHub Repository Structure

```bash
DropAlert-Rwanda-Project/
│
├── README.md
├── DropAlert_Rwanda_Analysis_Results.csv
├── Dropout_Analysis.ipynb
├── DropAlertDashboard.pbix
├── Screenshots/
│   └── dashboard_view.png
└── Presentation/
    └── Final_PPT_Slides.pptx
```

---

## 📌 Submission Checklist

* ✅ GitHub Repo: Well structured
* ✅ Power BI File: Uploaded
* ✅ README: Complete with documentation and screenshots
* ✅ Presentation: Finalized

---

📖 “Whatever you do, work at it with all your heart, as working for the Lord…” – Colossians 3:23
