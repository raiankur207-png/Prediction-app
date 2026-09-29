# Employee Retention Prediction

A Streamlit web application for predicting whether an employee is likely to stay or leave an organization based on employee profile information.

## Project Overview

This project uses a trained XGBoost classifier to assess employee retention risk. The application provides a simple interactive interface where users input employee details and receive:

- retention prediction
- probability of leaving
- probability of staying
- risk interpretation

## Features

- Interactive employee form
- Machine-learning-based prediction
- Real-time probability display
- Clear risk guidance and dashboard layout

## Project Files

- `app.py` – Streamlit web application
- `aug_train.csv` – training dataset
- `aug_test.csv` – testing dataset
- `employee_retention_xgboost_model.pkl` – trained model
- `feature_columns.pkl` – required model input columns
- `requirements.txt` – Python dependencies

## Local Setup

1. Open a terminal in the project folder.
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it:

   - Windows:

     ```bash
     .venv\Scripts\activate
     ```

   - macOS/Linux:

     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the app:

   ```bash
   streamlit run app.py
   ```

## GitHub Upload Steps

1. Initialize a Git repository:

   ```bash
   git init
   ```

2. Add all project files:

   ```bash
   git add .
   ```

3. Commit the files:

   ```bash
   git commit -m "Initial commit"
   ```

4. Create a repository on GitHub.
5. Link the repo and push:

   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repository-name>.git
   git push -u origin main
   ```

## Streamlit Cloud Deployment

This project is ready for Streamlit Community Cloud.

1. Push the project to GitHub.
2. Go to https://streamlit.io/cloud
3. Sign in with your GitHub account.
4. Click “Deploy app”.
5. Select the repository and branch.
6. Set the main file to `app.py`.
7. Deploy.

Streamlit Cloud will install the libraries from `requirements.txt` and run the app automatically.

## Model Notes

The application expects the following files in the project directory:

- `employee_retention_xgboost_model.pkl`
- `feature_columns.pkl`

These files are necessary for prediction and are included with the project.
