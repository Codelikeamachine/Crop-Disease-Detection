
Plant Disease Detection using Convolutional Neural Networks: 
This project aims to detect plant diseases using Convolutional Neural Networks (CNNs). The model is trained on a dataset containing images of various plant diseases and healthy plants. The trained model can then be used to predict the presence of diseases in plants based on input images.

Dataset: 
The dataset used in this project is obtained from Kaggle: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset. It contains images of healthy plants as well as plants affected by various diseases.

Project Structure: 
train folder: Contains training images categorized into different disease classes.
valid folder: Contains validation images for model evaluation.
main_code.py: Python script for the main functionality of the web application using Streamlit.
README.md: This file providing an overview of the project.
trained_plant_disease_model.keras: Saved trained model file.
training_hist.json: JSON file containing training history.
Dependencies: 
Ensure you have the following dependencies installed:

TensorFlow
Matplotlib
Pandas
Seaborn
Scikit-learn
Streamlit
You can install these dependencies using pip:

Copy code
pip install tensorflow matplotlib pandas seaborn scikit-learn streamlit

Usage: 
Data Preprocessing: The dataset is preprocessed to prepare it for training. This includes resizing images and splitting them into training and validation sets.

Model Building: A CNN model is constructed using TensorFlow's Keras API. The model architecture consists of several convolutional layers followed by max-pooling layers, dropout layers to prevent overfitting, and dense layers for classification.

Model Training: The model is trained using the training set and validated using the validation set. Training parameters such as learning rate, number of epochs, and batch size can be adjusted as needed.

Model Evaluation: After training, the model's performance is evaluated using various metrics such as accuracy, precision, recall, and F1-score. Confusion matrix visualization helps in understanding the model's performance across different classes.

Saving Model: Once trained, the model is saved for future use.

Web Application: The trained model can be integrated into a web application using Streamlit. Users can upload images of plants, and the application will predict whether the plant is healthy or affected by any disease.

Additional Notes: 
It's recommended to fine-tune hyperparameters and experiment with different architectures to improve model performance.
Ensure proper data augmentation techniques are applied to handle class imbalances and improve model generalization.
Continuously monitor and evaluate the model's performance on new data to maintain its accuracy and reliability.
References
TensorFlow Documentation
Kaggle
Streamlit Documentation

Author: 
Armaan Saraswat, 
Pragya, 
Akash Raj, 
Aadya Gupta, 
Nikhil.

License: 
This project is licensed under the GPL License.
