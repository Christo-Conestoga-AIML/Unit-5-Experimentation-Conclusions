# DDxPlus Dataset Analysis & AI Chatbot

## Overview
This project demonstrates a complete **data analysis and AI/ML pipeline** on the DDxPlus dataset to predict the most likely differential diagnosis for patients. The project follows a **Results-driven Analytical Method (RDAM)** and includes experimentation, model evaluation, and development of an AI chatbot tool.

---

## Dataset
The DDxPlus dataset contains patient records with the following key features:

- `AGE`: Patient age (numeric)  
- `SEX`: Patient gender (categorical: M/F)  
- `DIFFERENTIAL_DIAGNOSIS`: List of possible diagnoses with associated probabilities  
- `PATHOLOGY`: Confirmed pathology for the patient  
- `EVIDENCES`: List of evidences (symptoms, lab tests, etc.)  
- `INITIAL_EVIDENCE`: The first observed evidence for the patient  

**Note:**  
- The full DDxPlus dataset is approximately 1 GB in size. To manage this, we have used **Git LFS** (Large File Storage) to include it in the repository.  
- For faster analysis and quicker insight generation, the **analysis in this project was performed using a smaller test dataset** instead of the full dataset.


## Project Components

### 1. Exploratory Data Analysis (EDA)
- Dataset inspection and statistical summaries
- Parsing and visualization of differential diagnosis and evidences
- Analysis of distributions:
  - Age distribution
  - Sex distribution
  - Pathology distribution
  - Top differential diagnosis distribution

### 2. Results-driven Analytical Method (RDAM)
- Problem identification: Predict top differential diagnosis
- Data exploration and visualization
- Feature preparation: numeric, categorical, text
- Model building and experimentation
- Evaluation of model performance
- Integration of the best model into a chatbot

### 3. Data Preparation & Feature Engineering
- Handling numeric (`AGE`), categorical (`SEX`), and text (`TOP_DIAGNOSIS`, `EVIDENCES_TEXT`) features
- Scaling, one-hot encoding, TF-IDF vectorization
- Train-test split and label encoding

### 4. Experiment / A/B Testing
- **Pipeline A**: Logistic Regression using numeric + categorical features
- **Pipeline B**: Logistic Regression using numeric + categorical + text features
- Comparison of accuracy, macro F1-score, and confusion matrices
- Pipeline B shows improved performance due to inclusion of textual features

### 5. AI/ML Tool (Chatbot)
- **Input**: Patient features (`AGE`, `SEX`) and symptom evidences (`EVIDENCES_TEXT`)
- **Processing**: Pipeline B predicts top differential diagnosis
- **Output**: Top predicted diagnosis and probabilities
- **Use Case**: Assists medical professionals in suggesting likely conditions and guiding further investigation

### 6. Demo Predictions
- Sample predictions for new patients
- Visualization of top predicted probabilities for interpretability

### 7. Conclusions & Recommendations
- Pipeline B outperforms Pipeline A in all metrics
- Future improvements:
  - Incorporate more clinical features (lab tests, imaging)
  - Fine-tune text vectorization using domain-specific embeddings (BioBERT)
  - Provide ranked multiple likely diagnoses for better decision support

---

## Dependencies
The project uses the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Steps to Run
1. Clone the repository:
```bash
https://github.com/Christo-Conestoga-AIML/Unit-5-Experimentation-Conclusions.git
```
2. Navigate to the project folder
```bash
cd Unit-5-Experimentation-Conclusions
```
3. Install required packages:
```bash
pip install -r requirements.txt
```
4. Run the training script:
```bash
python lib/main.ipynb
```
5. EDA plots and logs will be displayed and in the shell output.
