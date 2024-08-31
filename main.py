from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model, encoder, and vectorizer
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_medicine(gender, symptoms, causes, disease):
    # Create a DataFrame for new data
    new_data = pd.DataFrame({
        'Gender': [gender],
        'Symptoms': [symptoms],
        'Causes': [causes],
        'Disease': [disease]
    })
    
    # Prepare the new data
    new_data['Symptoms'] = new_data['Symptoms'].fillna('')
    new_data['Causes'] = new_data['Causes'].fillna('')
    new_data['Disease'] = new_data['Disease'].fillna('')
    new_data['Features'] = new_data[['Symptoms', 'Causes', 'Disease']].apply(lambda x: ' '.join(map(str, x)), axis=1)
    
    # Encode the new data's Gender feature
    new_gender_encoded = encoder.transform(new_data[['Gender']])
    
    # Vectorize the new data's Features
    new_X_features = vectorizer.transform(new_data['Features'])
    
    # Combine the encoded gender with vectorized features
    new_X = pd.concat([pd.DataFrame(new_gender_encoded), pd.DataFrame(new_X_features.toarray())], axis=1)
    
    # Predict the medicine
    predictions = model.predict(new_X)
    return predictions[0]

@app.route('/predict3', methods=['POST'])
def predict():
    data = request.get_json()
    gender = data.get('Gender')
    symptoms = data.get('Symptoms')
    causes = data.get('Causes')
    disease = data.get('Disease')
    
    if not all([gender, symptoms, causes, disease]):
        return jsonify({'error': 'Missing data'}), 400

    predicted_medicine = predict_medicine(gender, symptoms, causes, disease)
    return jsonify({'Predicted Medicine': predicted_medicine})

if __name__ == "__main__":
    app.run(debug=True)
