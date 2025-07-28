from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models
classifier = joblib.load('placement_classifier.pkl')
regressor = joblib.load('placement_regressor.pkl')
company_map = joblib.load('company_map.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None, ctc=None, company=None, year=None, error=None)
@app.route('/predict', methods=['POST'])
def predict():
    company_name = request.form['company'].strip()
    year = int(request.form['year'])

    if company_name not in company_map:
        return render_template('index.html', error="Sorry .",
                               prediction=None, ctc=None, company=None, year=None)

    company_encoded = company_map[company_name]
    input_data = np.array([[company_encoded, year]])

    will_visit = classifier.predict(input_data)[0]
    expected_ctc = 0.0

    if will_visit == 1:
        expected_ctc = regressor.predict(input_data)[0]

    return render_template('index.html',
                           prediction=will_visit,
                           ctc=expected_ctc,
                           company=company_name,
                           year=year,
                           error=None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

