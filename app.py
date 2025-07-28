from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load models once at startup
classifier = joblib.load('placement_classifier.pkl')
regressor = joblib.load('placement_regressor.pkl')
company_map = joblib.load('company_map.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None, ctc=None, company=None, year=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company_name = request.form['company'].strip()
        year = int(request.form['year'])

        if company_name not in company_map:
            return render_template('index.html',
                                   error="Sorry, this company has no records.",
                                   prediction=None, ctc=None, company=None, year=None)

        # Encode input
        company_encoded = company_map[company_name]
        input_data = np.array([[company_encoded, year]])

        # Predict classification (1 = will visit, 0 = won't)
        will_visit = int(classifier.predict(input_data)[0])

        expected_ctc = float(regressor.predict(input_data)[0]) if will_visit == 1 else None

        return render_template('index.html',
                               prediction=will_visit,
                               ctc=expected_ctc,
                               company=company_name,
                               year=year,
                               error=None)

    except Exception as e:
        print(f"[ERROR] Prediction Failed: {e}")
        return render_template('index.html',
                               error="An error occurred while processing your request.",
                               prediction=None, ctc=None, company=None, year=None)

if __name__ == '__main__':
    # Disable debug and enable threading for better production performance
    app.run(host='0.0.0.0', port=10000, debug=False, threaded=True)
