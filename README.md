
# Student Loan Risk Prediction with Deep Learning

This project demonstrates the use of a deep learning model to predict student loan repayment success based on various features such as payment history, GPA ranking, financial aid score, and more. The project uses TensorFlow's Keras library to build, train, and evaluate a neural network model.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

## Overview
The goal of this project is to predict whether a student will successfully repay their loan based on their financial and academic profile. The project involves data preprocessing, feature scaling, model training, evaluation, and saving the trained model for future use.

## Dataset
The dataset used in this project is `student-loans.csv`, which contains the following features:
- `payment_history`
- `location_parameter`
- `stem_degree_score`
- `gpa_ranking`
- `alumni_success`
- `study_major_code`
- `time_to_completion`
- `finance_workshop_score`
- `cohort_ranking`
- `total_loan_score`
- `financial_aid_score`
- `credit_ranking` (target variable)

The dataset is preprocessed to handle missing values, scale features, and split into training and testing sets.

## Project Workflow
1. **Data Preparation**: Load and preprocess the dataset.
2. **Feature Scaling**: Use `StandardScaler` to scale the features.
3. **Model Building**: Create a neural network with two hidden layers.
4. **Model Training**: Train the model using the training dataset.
5. **Model Evaluation**: Evaluate the model's performance on the test dataset.
6. **Model Saving**: Save the trained model for future use.
7. **Prediction**: Use the saved model to make predictions on new data.

## Model Architecture
The neural network model consists of:
- Input Layer: Number of input features = 11
- Hidden Layer 1: 6 neurons with ReLU activation
- Hidden Layer 2: 3 neurons with ReLU activation
- Output Layer: 1 neuron with sigmoid activation

## Results
The model achieved the following performance metrics:
- **Loss**: 0.5102
- **Accuracy**: 75.99%

## Usage
1. Clone the repository.
2. Install the required Python libraries:
    ```bash
    pip install pandas tensorflow scikit-learn
    ```
3. Run the Jupyter Notebook to train and evaluate the model.
4. Use the saved model (`student_loans_model.keras`) to make predictions on new data.

## Future Work
- Improve the model by experimenting with different architectures and hyperparameters.
- Incorporate additional features to enhance prediction accuracy.
- Develop a recommendation system for student loan options.

## License
This project is licensed under the MIT License. See the LICENSE file for details.