from flask import Flask, render_template, request
import pickle

# Load the trained model from the pickle file
with open('iris.pkl','rb') as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

# Define the route for the home page (form page)
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction result page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the web form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])
    feature4 = float(request.form['feature4'])

    # Make the prediction using the loaded model
    input_features = [[feature1, feature2, feature3, feature4]]  # Create a 2D array
    prediction = model.predict(input_features)
    predicted_class = int(prediction[0])
    print("prediction is",prediction)
    print("predicted class",predicted_class)

    # Get the predicted class label (0, 1, or 2)
    predicted_label = ''
    if predicted_class == 0:
        predicted_label = 'setosa'
    elif predicted_class == 1:
        predicted_label = 'versicolor'
    elif predicted_class == 2:
        predicted_label = 'virginica'

    # Print the input features and prediction for debugging
    print("Input features:", [feature1, feature2, feature3, feature4])
    print("Predicted Label:", predicted_label)

    # Return the prediction result page
    return render_template('index.html', label=predicted_label)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
