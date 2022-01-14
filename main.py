import pickle
import numpy as np

from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
app=Flask(__name__)


@app.route('/', methods=['GET','POST'])
@cross_origin()
def index():
    #print('This is homepage')
    return render_template("Homepage.html")
@app.route('/predict', methods=['GET','POST'])
@cross_origin()
def predict():
    if request.method=='POST':
        try:
            Pclass=float(request.form["pclass"])
            #print(type(Pclass))
            Sex=float(request.form["sex"])
            Age=float(request.form["age"])
            Parch=float(request.form["Parch"])
            SibSp=float(request.form["SibSp"])
            Fare=float(request.form["fare"])


            filename='MLAssign_DecisionTree.pickle'
            model=pickle.load(open(filename,"rb"))
            #parr=np.array([[Pclass,Sex,Age,Parch,SibSp,Fare]])
            #parr=parr.reshape(1,-1)
            predy=model.predict([[Pclass,Sex,Age,Parch,SibSp,Fare]])
            print(predy)
            if predy[0]==0:
                predy='N'
            else:
                predy='Y'
            #print('Passenger will be Survived(Y/N)',predy)
            return render_template('results.html', prediction=predy)
        except Exception as e:
            print ("Exception is ",e)
            return "Something is wrong"
        else:
            return render_template('Homepage.html')

if __name__=="__main__":
    app.run(debug=True)

