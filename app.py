from flask import Flask,request,url_for,render_template
#import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from numpy import asarray
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import tensorflow as tf
# import requests
heart_model_path="models/heart.pkl"
heart_model=pickle.load(open(heart_model_path,'rb'))

# diabetes_model_path="models/diabates.h5"
# diabetes_model=load_model(diabetes_model_path)

diabetes_model_path="models/dia.pkl"
diabetes_model=pickle.load(open(diabetes_model_path,'rb'))

#liver_model_path="models/liver.h5"
#liver_model=load_model(liver_model_path)
#liver_model_path="models/RandomForest.pkl"
#liver_model=pickle.load(open(liver_model_path,'rb'))

liver_model_path="models/RandomForest1.pkl"
liver_model=pickle.load(open(liver_model_path,'rb'))

#malaria_model_path="models/mal2.h5"
#malaria_model=load_model(malaria_model_path)

malaria_model=tf.keras.models.load_model('models/mal-036.model')

mri_model_path="models/RandomForest1.pkl"
mri_model=pickle.load(open(mri_model_path,'rb'))

ct_model=tf.keras.models.load_model('models/cov-003.model')


app = Flask(__name__)
 
@app.route('/')
def hello_world():
    return render_template("index.html")
@app.route("/heart")
def heart():
    return render_template("heart.html")
@app.route("/heartpredict",methods=["POST"])
def heart_predict():
    age=int(request.form["age"])
    trestbps=int(request.form["trestbps"])
    gender=int(request.form["gender"])
    cp=int(request.form["cp"])
    chol=int(request.form["chol"])
    fbs=int(request.form["fbs"])
    exang=int(request.form["exang"])
    thalach=int(request.form["thalach"])
    restecg=int(request.form["restecg"])
    oldpeak=float(request.form["oldpeak"])
    slope=int(request.form["slope"])
    thal=int(request.form["thal"])
    ca=int(request.form["ca"])
    features=np.array([[age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    #features=np.array([[44,1,1,120,263,0,1,173,0,0,2,0,3]])
    #features=np.array([[60,0,0,150,258,0,0,157,0,2.6,1,2,3]])
    #features=np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
    features=np.array([[57,0,0,120,354,0,1,163,1,0.6,2,0,2]])
    predict=heart_model.predict(features)
    if(predict>=1):
        return render_template("heart.html",pred="You may have Heart Disease")
    else:
        return render_template("heart.html",pred="You may not have Heart Disease")
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")
@app.route("/diabetespredict",methods=["POST"])
def diabetes_predict():
    num_preg=int(request.form["num_preg"])
    glucose_conc=int(request.form["glucose_conc"])
    diastolic_bp=int(request.form["diastolic_bp"])
    thickness=float(request.form["thickness"])
    insulin=int(request.form["insulin"])
    bmi=float(request.form["bmi"])
    diab_pred=float(request.form["diab_pred"])
    age=int(request.form["age"])
    #skin=float(request.form["thickness"])
    features=np.array([[num_preg,glucose_conc,diastolic_bp,insulin,bmi,diab_pred,age,thickness]])
    #features=np.array([[0,137,40,168,43.1,2.288,33,1.379]]) #true
    #features=np.array([[1,89,66,94,28.1,0.167,21,0.9062]]) #false
    predict=diabetes_model.predict(features)
    if(predict>=1):
         return render_template("diabetes.html",pred="The person is likely to have diabetes")
    else:
         return render_template("diabetes.html",pred="The person is not likely to have diabetes")

@app.route("/liver")
def liver():
    return render_template("liver.html")
@app.route("/liverpredict",methods=["POST"])
def liver_predict():
    age=int(request.form["age"])
    gender=int(request.form["gender"])
    total_bilirubin=float(request.form["total_bilirubin"])
    direct_bilirubin=float(request.form["direct_bilirubin"])
    alkaline_phosphotase=int(request.form["alk"])
    alamine_aminotransferase=int(request.form["alm"])
    aspartate_aminotransferase=int(request.form["asp"])
    total_proteins=float(request.form["total_proteins"])
    albumin=float(request.form["albumin"])
    albumin_globulin_ratio=float(request.form["agr"])
    features=np.array([[age,gender,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_globulin_ratio]])
    #features=np.array([[65,1,0.7,0.1,187,16,18,6.8,3.3,0.90]])
    #features=np.array([[63,1,0.9,0.2,194,52,45,6,3.9,1.85]])
    #features=np.array([[84,0,0.2,188,13,21,6,3.2,1.1]])
    #features=np.array([[47,1,2.7,1.3,275,123,73,6.2,3.3,1.1]])
    #features=np.array([[50,1,1.2,415,407,576,6.4,3.2,1]]) #has disease
    #features=np.array([[65,1,0.7,0.1,187,16,18,6.8,3.3,0.90]]) 
    #features=np.array([[84,0,0.2,188,13,21,6,3.2,1.1]]) #No disease
    predict=liver_model.predict(features)
    if(predict==1):
        return render_template("liver.html",pred="The person may have Liver Disease")
    else:
        return render_template("liver.html",pred="The person may not have Liver Disease")

@app.route("/ct")
def ct():
    return render_template("ct.html")
@app.route("/ctpredict",methods=["POST"])
def ct_predict():
    filestr = request.files['ctscan'].read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
    resized=cv2.resize(gray,(100,100))
    r=resized.reshape(1,100,100,1)
    predict=np.argmax(ct_model.predict(r), axis=1)
    if(predict==0):
        return render_template("ct.html",pred="The person has COVID-19")
    elif(predict==1):
        return render_template("ct.html",pred="The person does not have trace of COVID or Pneumonia")
    else:
        return render_template("ct.html",pred="The person has Pneumonia")
@app.route("/malaria")
def malaria():
    return render_template("malaria.html")
@app.route("/malariapredict",methods=["POST"])
def malaria_predict():
    filestr = request.files['malaria'].read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
    resized=cv2.resize(gray,(100,100))
    r=resized.reshape(1,100,100,1)
    predict=np.argmax(malaria_model.predict(r), axis=1)
    if(predict==1):
        return render_template("malaria.html",pred="The person has malaria.")
    else:
       return render_template("malaria.html",pred="The person does not have malaria")
@app.route("/mri")
def mri():
    return render_template("mri.html")
@app.route("/mripredict",methods=["POST"])
def mri_predict():
    filestr = request.files['mriscan'].read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
    resized=cv2.resize(gray,(100,100))
    r=resized.reshape(1,100,100,1)
    predict=mri_model.predict(r)
    if(predict[0][0]==1):
        return render_template("mri.html",pred="The person does not have brain disease")
    else:
        return render_template("mri.html",pred="The person has brain disease")
if __name__ == '__main__':
    app.run(debug=True)