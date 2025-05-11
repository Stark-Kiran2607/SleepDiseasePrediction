from flask import Flask, url_for, redirect, render_template, request, session, make_response
import mysql.connector, os, re
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from rag_engine import RAGEngine  

app = Flask(__name__)
app.secret_key = 'admin'

# Connect to MySQL
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database=""
)
mycursor = mydb.cursor()

# Database functions
def executionquery(query, values):
    mycursor.execute(query, values)
    mydb.commit()
    return

def retrivequery1(query, values):
    mycursor.execute(query, values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

# Initialize RAG
rag = RAGEngine('sleep_disease_docs')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = [i[0] for i in email_data]
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID already exists!")
        return render_template('register.html', message="Confirm password does not match!")
    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = [i[0] for i in email_data]
        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password_data = retrivequery1(query, values)
            if password.upper() == password_data[0][0]:
                session['user_email'] = email
                return render_template('home.html')
            return render_template('login.html', message="Invalid Password!")
        return render_template('login.html', message="This email ID does not exist!")
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ✅ Prediction route
@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    result = None
    report = None
    if request.method == "POST":
        patient_name = request.form.get('Patient_Name')  # New Patient Name field
        
        Gender = request.form['Gender']
        Age = int(request.form['Age'])
        Occupation = request.form['Occupation']
        Sleep_Duration = float(request.form['Sleep_Duration'])
        Quality_of_Sleep = int(request.form['Quality_of_Sleep'])
        Physical_Activity_Level = int(request.form['Physical_Activity_Level'])
        Stress_Level = int(request.form['Stress_Level'])
        BMI_Category = request.form['BMI_Category']
        systolic = int(request.form['systolic'])
        diastolic = int(request.form['diastolic'])
        Heart_Rate = int(request.form['Heart_Rate'])
        Daily_Steps = int(request.form['Daily_Steps'])

        Blood_Pressure = f"{systolic}/{diastolic}"

        with open('Models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('Models/k_best_selector.pkl', 'rb') as f:
            k_best = pickle.load(f)

        with open('Models/stacking_classifier_k_best.pkl', 'rb') as f:
            stacking_classifier = pickle.load(f)

        single_input = {
            'Gender': Gender,
            'Age': Age,
            'Occupation': Occupation,
            'Sleep Duration': Sleep_Duration,
            'Quality of Sleep': Quality_of_Sleep,
            'Physical Activity Level': Physical_Activity_Level,
            'Stress Level': Stress_Level,
            'BMI Category': BMI_Category,
            'Blood Pressure': Blood_Pressure,
            'Heart Rate': Heart_Rate,
            'Daily Steps': Daily_Steps
        }
        input_df = pd.DataFrame([single_input])
        
        input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})
        input_df['Occupation'] = pd.Categorical(input_df['Occupation']).codes
        input_df['BMI Category'] = pd.Categorical(input_df['BMI Category']).codes
        input_df['Blood Pressure'] = input_df['Blood Pressure'].str.split('/').apply(lambda x: int(x[0]))

        input_df['Is_Overweight'] = (input_df['BMI Category'] >= 2).astype(int)
        input_df['Is_Hypertensive'] = (input_df['Blood Pressure'] > 130).astype(int)
        input_df['Is_High_Heart_Rate'] = (input_df['Heart Rate'] > 80).astype(int)
        input_df['Is_Short_Sleeper'] = (input_df['Sleep Duration'] < 6.5).astype(int)
        input_df['Is_Stressed'] = (input_df['Stress Level'] > 5).astype(int)
        input_df['Is_Low_Steps'] = (input_df['Daily Steps'] < 5000).astype(int)
        input_df['Overweight_and_Hypertensive'] = (
            (input_df['Is_Overweight'] == 1) & (input_df['Is_Hypertensive'] == 1)
            ).astype(int)
        input_df['HeartStress_and_ShortSleep'] = (
        (input_df['Is_High_Heart_Rate'] == 1) & (input_df['Is_Short_Sleeper'] == 1)
        ).astype(int)
        input_df = input_df.fillna(0)

        all_features = [
            'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
            'Physical Activity Level', 'Stress Level', 'BMI Category', 'Blood Pressure',
            'Heart Rate', 'Daily Steps','Is_Overweight', 'Is_Hypertensive', 'Is_High_Heart_Rate',
            'Is_Short_Sleeper', 'Is_Stressed', 'Is_Low_Steps',
            'Overweight_and_Hypertensive', 'HeartStress_and_ShortSleep'
        ]
        for col in all_features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[all_features]

        input_scaled = scaler.transform(input_df)
        input_k_best = k_best.transform(input_scaled)

        prediction = stacking_classifier.predict(input_k_best)

        class_labels = {'No Disorder': 'No Disorder', 'Sleep Apnea': 'Sleep Apnea', 'Insomnia': 'Insomnia', 'Narcolepsy': 'Narcolepsy', 'Restless Leg Syndrome': 'Restless Leg Syndrome'}  
        predicted_class = class_labels[prediction[0]]

        if predicted_class.lower() != "no disorder":
            report = rag.generate_report(patient_name=patient_name, disease_name=predicted_class)
        else:
            report = f"✅ {patient_name} shows no signs of a sleep disorder."

        session['report'] = report
        return render_template('results.html', prediction=predicted_class, report=report)

    # Prepare data for dropdowns
    df = pd.read_csv(r"Dataset/Sleep_disease_dataset.csv")

    columns_to_drop = ['Sleep Disorder', 'Blood Pressure']
    df = df.drop(columns=columns_to_drop)

    df.columns = [re.sub(r'\s+', '_', col) for col in df.columns]

    object_columns = df.select_dtypes(include=['object']).columns
    labels = {col: df[col].value_counts().to_dict() for col in object_columns}

    data = {}
    for key in labels.keys():
        data[key] = []
        for value, count in labels[key].items():
            data[key].append((value, count))

    # Ensure required fields
    required_fields = ['Gender', 'Occupation', 'BMI_Category']
    for field in required_fields:
        if field not in data:
            if field == "Gender":
                data[field] = [('Male', 0), ('Female', 1)]
            elif field == "Occupation":
                data[field] = [('Doctor', 0), ('Engineer', 1), ('Teacher', 2)]
            elif field == "BMI_Category":
                data[field] = [('Underweight', 0), ('Normal', 1), ('Overweight', 2), ('Obese', 3)]

    return render_template('prediction.html', data=data, prediction=result)

@app.route('/download_report', methods=["POST"])
def download_report():
    report = session.get('report', '')
    response = make_response(report)
    response.headers['Content-Disposition'] = 'attachment; filename=patient_report.txt'
    response.headers['Content-Type'] = 'text/plain'
    return response

if __name__ == '__main__':
    app.run(debug=True)
