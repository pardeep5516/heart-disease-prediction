import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

# Load the heart disease data
heart_disease_data = pd.read_csv('heart-disease.csv')  # Replace with your actual data file

# Split the data into features (X) and target variable (y)
X = heart_disease_data.drop('target', axis=1)
y = heart_disease_data['target']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Logistic Regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
cv_accuracy = np.mean(cv_scores)

# Define the feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Set random initial values for the input fields
initial_values = [np.random.choice(X[feature].dropna().unique()) for feature in feature_names]

# Main application
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Heart Disease Prediction")
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7Prwr83EazqsPWzN0C7k1bmRY9RaoZWEK7A&usqp=CAU', use_column_width=True)
    st.markdown("---")

    # Display model accuracy and cross-validation accuracy
    col1, col2 = st.columns(2)
    with col1:
        st.info("Model Accuracy")
        st.write(f"{accuracy:.2f}")
    with col2:
        st.info("Cross-Validation Accuracy")
        st.write(f"{cv_accuracy:.2f}")
    st.markdown("---")

    # Sidebar with user input
    st.sidebar.title("User Input")
    user_inputs = get_user_inputs()
    
    # Check if inputs are provided
    if user_inputs:
        # Convert user inputs to a DataFrame
        input_data = pd.DataFrame([user_inputs], columns=feature_names)
        
        # Make predictions
        prediction = model.predict(input_data)
        
        # Display prediction
        if prediction[0] == 1:
            st.success("The patient is predicted to have heart disease.")
        else:
            st.success("The patient is predicted to not have heart disease.")
    
    # Data visualization
    st.markdown("---")
    st.subheader("Data Visualization")
    visualize_data()

    # Confusion matrix
    st.markdown("---")
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(model, X_test, y_test, display_labels=["No Heart Disease", "Heart Disease"])
    st.pyplot()

    # Footer
    st.markdown("---")
    st.write("Made with ❤️ by Your Name")

# Create a function to get user inputs
def get_user_inputs():
    inputs = []
    for i, feature in enumerate(feature_names):
        value = st.sidebar.text_input(f"{feature.capitalize()}", value=initial_values[i])
        if is_float(value):
            inputs.append(float(value))
        elif is_integer(value):
            inputs.append(int(value))
        else:
            inputs.append(None)
    return inputs

# Check if the value is a float
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Check if the value is an integer
def is_integer(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

# Data visualization
def visualize_data():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4)
    sns.countplot(x='sex', data=heart_disease_data, ax=axes[0, 0])
    sns.countplot(x='cp', data=heart_disease_data, ax=axes[0, 1])
    sns.countplot(x='fbs', data=heart_disease_data, ax=axes[1, 0])
    sns.countplot(x='restecg', data=heart_disease_data, ax=axes[1, 1])
    axes[0, 0].set_title('Sex')
    axes[0, 1].set_title('Chest Pain Type')
    axes[1, 0].set_title('Fasting Blood Sugar')
    axes[1, 1].set_title('Resting ECG')
    st.pyplot(fig)

if __name__ == "__main__":
    main()

