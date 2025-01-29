Heart Disease Prediction Project

Overview

This project uses machine learning to predict the presence of heart disease based on medical data. It includes data preprocessing, model training, evaluation, and saving the trained model for deployment.

Dataset

The dataset should be placed in the project directory as Dataset Heart Disease.csv.

The dataset contains patient data including age, cholesterol levels, blood pressure, and other medical indicators.

Requirements

Ensure you have Python installed along with the following dependencies:

pip install pandas numpy seaborn matplotlib scikit-learn joblib

Usage

Clone this repository:

git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

Place the dataset (Dataset Heart Disease.csv) in the project directory.

Run the script to train and evaluate the model:

python heart_disease_project.py

The trained model (heart_disease_model.pkl) and scaler (scaler.pkl) will be saved for later use.

Model Performance

The script trains a Random Forest classifier and evaluates its performance using:

Accuracy Score

Confusion Matrix

Classification Report

Deployment

You can use the saved heart_disease_model.pkl and scaler.pkl files to make predictions in a separate application.

Contributing

Feel free to contribute by adding more models, improving data preprocessing, or enhancing the deployment process.

License

This project is open-source and available under the MIT License.