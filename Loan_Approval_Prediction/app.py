from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model & scaler
model = joblib.load("final_model/loan_model.pkl")
scaler = joblib.load("final_model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        Gender = int(request.form["Gender"])
        Married = int(request.form["Married"])
        Dependents = int(request.form["Dependents"])
        Education = int(request.form["Education"])
        Self_Employed = int(request.form["Self_Employed"])
        ApplicantIncome = float(request.form["ApplicantIncome"])
        CoapplicantIncome = float(request.form["CoapplicantIncome"])
        LoanAmount = float(request.form["LoanAmount"])
        Loan_Amount_Term = float(request.form["Loan_Amount_Term"])
        Credit_History = float(request.form["Credit_History"])
        Property_Area = int(request.form["Property_Area"])

        # Create numpy array from inputs
        input_data = np.array([[Gender, Married, Dependents, Education, Self_Employed,
                                ApplicantIncome, CoapplicantIncome, LoanAmount,
                                Loan_Amount_Term, Credit_History, Property_Area]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict class & probability
        prediction_class = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][prediction_class] * 100

        # Result text
        result = "Approved" if prediction_class == 1 else "Not Approved"

        return render_template("result.html",
                               prediction=f"{result} — {prediction_proba:.2f}% probability")

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
