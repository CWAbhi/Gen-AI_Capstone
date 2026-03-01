# INTELLIGENT CREDIT RISK SCORING & AGENTIC LENDING DECISION SUPPORT

##  Project Overview

This project involves the design and implementation of an system that evaluates borrower credit risk and evolves in decision support assistant. AI-driven credit analytics of an agentic AI lending.

---

##  Problem Statement

Financial institutions face significant challenges in assessing borrower creditworthiness accurately. Manual risk evaluation processes are often time consuming, inconsistent, and prone to human bias.

This project addresses this problem by implementing an automated credit risk scoring system that uses machine learning algorithms to analyze borrower data and classify applicants into risk categories.


##  Key Features

- Upload borrower dataset through an interactive UI  
- Automatic data preprocessing pipeline  
- Support for categorical encoding and feature scaling  
- Training and comparison of multiple ML models  
- Real-time credit risk prediction  
- Visualization of evaluation metrics  
- Clean and user-friendly Streamlit interface  


##  Machine Learning Models Used

The following supervised learning models were implemented:

**Logistic Regression**
- Used for probabilistic classification
- Estimates default likelihood

**Decision Tree Classifier**
- Rule-based classification model
- Identifies important risk driving features


##  Evaluation Metrics

Model performance is evaluated using:

- Accuracy Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve Visualization
- Feature Importance Analysis


## Installation and Setup Instructions

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/CWAbhi/Gen-AI_Capstone.git
cd Gen-AI_Capstone
```
### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```
### Step 3: Launch the Application

Start the Streamlit server:
``` bash
streamlit run app.py
```
The application will open automatically in your browser.


## Team Contribution

| Member                     | Contribution                                                              |
| -------------------------- | ------------------------------------------------------------------------- |
| Anshika Seth (2401010080)  | Data Cleaning & EDA, Complete Model Development, Streamlit UI, Deployment |
| Abhijeet Dey (2401010014)  | Helped Model Development, Deployment                                      |
| Aditya Ranjan (2401010035) | Documentation & Testing                                                   |


## Conclusion

The Credit Risk Prediction System successfully demonstrates how Machine Learning can automate loan risk assessment. The trained model achieved strong performance and can assist financial institutions in making reliable lending decisions.