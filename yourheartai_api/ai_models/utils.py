            
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import load_model
from tensorflow import keras
from keras.layers import Dense
from numpy import loadtxt
from keras.models import Sequential, load_model
import pandas as pd
from pandas import read_csv


            
def getCancerPrediction(filename):
    
    classes = ['Actinic keratoses', 'Basal cell carcinoma', 
               'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 
               'Melanocytic nevi', 'Vascular lesions']
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
    
    
    #Load model
    my_model=load_model("model/cancer/HAM10000_100epochs.h5")
    
    SIZE = 32 #Resize to same size as training images
    img_path = 'yourheartai_api/static/ai_images/cancer/'+filename
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
    img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    pred = my_model.predict(img) #Predict                    
    
    #Convert prediction to class name
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Diagnosis is:", pred_class)
    return pred_class


def getCHDPrediction(patientData): 
    #load model
    model_mlp = load_model("model/cvd/chd_mlp/CHD-MLP-Regression_300e.h5")

    #load Datafram with mean and standard devs to scale patientData - df.describ()
    Desc_Data_file = "model/cvd/chd_mlp/CHD-DataFrame-Desc.csv"
    desc_df = read_csv(Desc_Data_file, delim_whitespace=False)

    ### Dataframe row descriptions strs ### 
    #row 0 - count
    #row 1 - mean
    #row 2 - std
    #row 3 - min
    #row 4 - 25%
    #row 5 - 50%
    #row 6 - 75%
    #row 7 - max
    ### Dataframe row descriptions ends ###

    mean_n_train = desc_df.loc[1,:]
    std_n_train = desc_df.loc[2,:]

    #add patient data array to panda
    patientData_np = np.array(patientData)
    # Transform the data
    patientData_scaled = (patientData_np-mean_n_train)/std_n_train
    ## Reshape input data
    patientData_scaled_rs = patientData_scaled.values.reshape(1,12) 

    ## Predict Patient input data
    patientPred = model_mlp.predict(patientData_scaled_rs)
    print("Patient Predicted values are: \n", patientPred)    

    ##### Threshold for CHD risk #####
    chd_pred = patientPred[0][0]

    if chd_pred > 1:
        chd_pred = .999 #risk prediction cannot be 100%
    elif chd_pred < 0:
        chd_pred = 0

    #CHD Risk Grading
    risk_category = ""

    if 0 <= chd_pred <= 0.25: # and round(value,2)==value:
        risk_category = "low risk"      
    elif 0.25 <= chd_pred <= 0.5: # and round(value,2)==value:
        risk_category = "mid risk"      
    elif 0.5 <= chd_pred <= 0.75: # and round(value,2)==value:
        risk_category = "high risk"      
    elif 0.75 <= chd_pred <= 1: # and round(value,2)==value:
        risk_category = "Very high risk"      

    chd_pred = round(chd_pred, 4)
    patient_pred_dict = {
        "CHD Probability": chd_pred,
        "Risk Category": risk_category
        }
    
    return patient_pred_dict
