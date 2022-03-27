
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('test.html')

@app.route('/train_dataset')
def train_dataset():
    df = pd.read_csv("fuel_data.csv")
    x = df.filter(['drivenKM'])
    y = df.filter(['fuelAmount'])
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=42)
    model=LinearRegression()
    model.fit(x_train, y_train)
    pickle.dump(model, open('model.pkl','wb'))
    return render_template('train_dataset.html')


@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    model = pickle.load(open('model.pkl', 'rb'))
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('test.html', prediction_text=f'fuel price for kilometer driven is :{prediction}')

if __name__ == '__main__':
    app.run()