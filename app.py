from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('trained model/model.h5')
with open('trained model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']

    if not review.strip():
        return render_template('index.html', prediction="\u26A0 Please enter a review.")

    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]
    sentiment = 'Positive \U0001F60A' if prediction >= 0.5 else 'Negative \U0001F641'
    confidence = f"{prediction:.2f}" if sentiment == 'Positive \U0001F60A' else f"{1 - prediction:.2f}"

    return render_template('index.html',
                           prediction=f"{sentiment} (Confidence: {confidence})",
                           original_text=review)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    review = data.get('review', '')

    if not review.strip():
        return jsonify({'error': 'Empty review text'}), 400

    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]
    sentiment = 'positive' if prediction >= 0.5 else 'negative'
    confidence = float(prediction if sentiment == 'positive' else 1 - prediction)

    return jsonify({
        'review': review,
        'sentiment': sentiment,
        'confidence': round(confidence, 3)
    })

if __name__ == '__main__':
    app.run(debug=True)
