from flask import Flask, request, jsonify
import joblib

# Flask app start
app = Flask(__name__)

# Load trained model (use apna trained model .pkl file)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    # Text ko vectorize karo
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0].max()

    return jsonify({
        'label': 'FAKE' if prediction == 1 else 'REAL',
        'confidence': float(confidence),
        'top_tokens': []
    })

if __name__ == '__main__':
    app.run(debug=True)
