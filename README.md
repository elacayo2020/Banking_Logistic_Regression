## Project Summary: Bank Marketing Response Prediction

### Overview
This project applies data mining and machine learning techniques to a bank marketing dataset (https://www.kaggle.com/competitions/marketing-data/overview) to predict whether a customer will respond to a marketing campaign. The goal is to identify high-potential customers while balancing model performance with interpretability and real-world usability.

---

### Data Understanding & EDA
Initial exploratory data analysis (EDA) focused on understanding customer behavior and dataset structure:

- Handled special values (e.g., `pdays = 999` indicating no prior contact) by:
  - Creating a binary feature for prior contact
  - Cleaning the numeric feature for modeling
- Analyzed distributions and relationships using:
  - Histograms and boxplots
  - Correlation analysis for numeric features
- Evaluated class imbalance, confirming that the target variable is skewed toward the negative class (non-response)

---

### Data Preprocessing
A preprocessing pipeline was built to ensure consistent and reproducible transformations:

- **Numerical Features**
  - Scaled using MinMaxScaler
- **Categorical Features**
  - One-hot encoded using OneHotEncoder
- Combined using a **ColumnTransformer**

This allowed seamless integration with machine learning models.

---

### Feature Engineering & Selection
Multiple approaches were explored to improve model performance and reduce dimensionality:

- Created engineered features (e.g., prior contact indicator)
- Applied **SelectKBest (ANOVA F-test)** to identify top predictive features
- Tested **Sequential Feature Selection (SFS)** for model-driven feature selection
- Compared performance between full-feature and reduced-feature models

---

### Models Built
Several classification models were developed and compared:

#### 1. Logistic Regression
- Primary model used due to interpretability and effectiveness
- Applied **class weighting** to address imbalance
- Evaluated both:
  - Full feature set
  - Feature-selected versions (SequentialFeatureSelector and SelectKBest)

#### 2. Support Vector Machine (LinearSVC)
- Tested as an alternative linear classifier
- Compared performance with logistic regression

---

### Model Evaluation
Models were evaluated using multiple metrics to account for class imbalance:

- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, F1-score**
- **ROC-AUC**
- **Matthews Correlation Coefficient (MCC)**

Key findings:
- Class weighting significantly improved minority class detection
- The best model based on MCC (0.5611) was the Logistic Regression model with weights {0: 1, 1: 10} and SelectKBest feature selection with k=38
- Tradeoff observed between **recall** (capturing responders) and **precision** (avoiding false positives)

---

### Final Model
The final model was implemented as a **scikit-learn pipeline**, combining:

- Preprocessing (scaling + encoding)
- Feature selection
- Logistic regression with class weighting

The pipeline was retrained on the full dataset and serialized using `joblib` for deployment.

---

### Key Insights
- Customers previously contacted behave significantly differently from those never contacted
- Feature selection helps simplify the model without sacrificing performance
- Handling class imbalance is critical for meaningful predictions
- Logistic regression provides a strong balance of performance and interpretability for this problem

---

### Conclusion
This project demonstrates a full machine learning workflow, from EDA and preprocessing to model selection and deployment. The final model provides a practical solution for identifying high-value customers in marketing campaigns while maintaining transparency and efficiency.
