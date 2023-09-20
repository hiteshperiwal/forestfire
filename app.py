from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("goldprice.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[x for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    output='{}'.format(prediction)
    return render_template('goldprice.html',pred='Predicted price of gold is {}'.format(output),bhai="kuch karna hain iska ab?")
if __name__ == '__main__':
    app.run(debug=True)
