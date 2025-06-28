# Movie Review Sentiment Analysis Web App

This project is a sentiment analysis web application that classifies movie reviews as either **positive** or **negative** using a deep learning model built with TensorFlow. It provides both a browser-based interface and a REST API for interacting with the model. The model was trained on the IMDb dataset and deployed using Flask.


---

## Features

- Deep learning-based sentiment prediction using TensorFlow
- Clean web interface built with Flask and HTML/CSS
- REST API endpoint for programmatic access
- Input validation and basic error handling
- Confidence score displayed with each prediction
- Deployed on Render (Free Tier)

---

## Model Overview

- **Architecture**: LSTM-based binary classifier
- **Input**: Raw movie review text
- **Preprocessing**: Tokenization and sequence padding
- **Dataset**: IMDb 50,000 labeled reviews
- **Accuracy**: Achieved ~97% on validation data

The model was trained in Google Colab and saved in `.h5` format, with a tokenizer saved separately in `.pkl` format.

---

## Example Predictions

| Review Text                           | Expected Output |
|--------------------------------------|-----------------|
| This movie was fantastic             | Positive        |
| Could have been better               | Negative        |
| It was amazing                       | Positive        |
| A complete waste of time             | Negative        |

---

## Known Limitations

While the model performs well on standard reviews, it sometimes misclassifies short or comparative phrases such as:

- "Better than expected"
This behavior is likely due to:
- Limited contextual understanding in short or ambiguous inputs

Improvements could involve using a contextual language model like BERT or expanding the dataset to include more diverse short reviews.


---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ak-1203/sentiment-analysis-flask.git
cd sentiment-analysis-flask

### 2. Install Dependencies

```bash
pip install -r requirements.txt

### 3. Run the Application
```bash
python app.py
