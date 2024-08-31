            
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense
from numpy import loadtxt
from keras.models import Sequential, load_model
import pandas as pd
from pandas import read_csv

##mask r-cnn imports
# evaluate the mask rcnn model on the stenosis dataset
import mrcnn
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config

import os
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean

from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import cv2    
import skimage
import matplotlib.pyplot as plt
from mrcnn.visualize import save_image
from flask import url_for, current_app, jsonify


def getStenosisPrediction(filename):   
    
    LOGS_DIR = "./model/logs"
    # H5_FILENAME = "Stenosis_mrcnn_train-6661s-832v-1000spe-10e.h5"
    H5_FILENAME = "Stenosis_mrcnn_train-6661s-832v-1000spe-10e.h5"
    MODEL_DIR = os.path.join(current_app.root_path,"ai_models/cvd/stn_m_rcnn",H5_FILENAME) 
    print("################################################################################")
    print("MODEL_DIR: ",MODEL_DIR)
    print("################################################################################")
    IMAGE_DIR = os.path.join(current_app.root_path,'static/ai_images/chd-mrcnn/raw/')
    SAVE_RES_DIR = os.path.join(current_app.root_path,"static/ai_images/chd-mrcnn/results")

    # load the class label names from disk, one label per line
    CLASS_NAMES = ['BG', 'stenosis']


    print("Setting up config: StenosisPredConfig")
    class StenosisPredConfig(mrcnn.config.Config):
        # Give the configuration a recognizable name
        NAME = "stenosis_inference"
        BATCH_SIZE = 1

        # set the number of GPUs to use along with the number of images per GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
        NUM_CLASSES = len(CLASS_NAMES)
        # USE_MINI_MASK = False
        DETECTION_MIN_CONFIDENCE = 0.6

    # Create Prediction Config
    predCfg = StenosisPredConfig()
    # predCfg.display()

    # Initialize the Mask R-CNN model for inference and then load the weights.
    # This step builds the Keras model architecture. USE CPU for inference
    # DEVICE = "/cpu:0"
    # with tf.device(DEVICE):
        # model = mrcnn.model.MaskRCNN(mode="inference",
        #                             config=predCfg,
        #                             model_dir=LOGS_DIR)
    
    print("Generating inference model")
    model = mrcnn.model.MaskRCNN(mode="inference",
                                    config=predCfg,
                                    model_dir=LOGS_DIR)
    
    print("Loading weights")
    # Load the weights into the model.
    model.load_weights(MODEL_DIR, by_name=True)
    
    # load the input image, convert it from BGR to RGB channel
    # image = cv2.imread(IMAGE_DIR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Beginning Inference")
    #Save Output Images
    ## See solution here: https://github.com/matterport/Mask_RCNN/pull/38
    ## Implemented save_image visualize.py
    ## To correctly install m-rcnn, see notes - mask-rcnn dependecies notes.txt
    
    SIZE = 1024 #Resize to same size as training images
    img_path = IMAGE_DIR+filename
    print("################################################################################")
    print("##### img_path: ",img_path)
    print("################################################################################")
    # image = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    # image = skimage.io.imread(os.path.join(img_path)).resize((SIZE,SIZE))
    image = skimage.io.imread(os.path.join(img_path))
    image_name = filename[:-4]+"_res"

    # Perform a forward pass of the network to obtain the results
    # r = model.detect([image], verbose=0) 
    print("Performing forward pass of the network")   
    results = model.detect([image], verbose=1)
    # # Get the results for the first image.
    r = results[0]

    print("Visualize & save image")
    mrcnn.visualize.save_image(
        image, 
        image_name, 
        r['rois'], 
        r['masks'],
        r['class_ids'],
        r['scores'],
        CLASS_NAMES,
        scores_thresh=0.6,
        save_dir=SAVE_RES_DIR,
        mode=0)

    result_img_url = os.path.join(  "static/ai_images/chd-mrcnn/results",image_name) 

    print("################################################################################")
    # print("result_img_url: ",result_img_url)
    print("################################################################################")

    patient_sten_pred_dict = image_name
    return patient_sten_pred_dict



def getCHDPrediction(patientData): 
    age = patientData["age"]
    gender = patientData["gender"]
    chestPain = patientData["chestPain"]
    restingBP = patientData["restingBP"]
    serumCholestrol = patientData["serumCholestrol"]
    fastingBloodSugar = patientData["fastingBloodSugar"]
    restingRElectro = patientData["restingRElectro"]
    maxHeartRate = patientData["maxHeartRate"]
    exerciseAngia = patientData["exerciseAngia"]
    oldPeak = patientData["oldPeak"]
    slope = patientData["slope"]
    noOfMajorVessels = patientData["noOfMajorVessels"]

    patientData_arr = []
    patientData_arr.append(age)
    patientData_arr.append(gender)
    patientData_arr.append(chestPain)
    patientData_arr.append(restingBP)
    patientData_arr.append(serumCholestrol)
    patientData_arr.append(fastingBloodSugar)
    patientData_arr.append(restingRElectro)
    patientData_arr.append(maxHeartRate)
    patientData_arr.append(exerciseAngia)
    patientData_arr.append(oldPeak)
    patientData_arr.append(slope)
    patientData_arr.append(noOfMajorVessels)

    print("################################################################################")
    print("patientData: ",patientData_arr)
    print("################################################################################")

    #load model
    KERAS_FILENAME = "CHD-MLP-Regression_250e.keras"
    MODEL_DIR = os.path.join(current_app.root_path,"ai_models/cvd/chd_mlp",KERAS_FILENAME) 
    model_mlp = load_model(MODEL_DIR)

    TFLITE_FILE = "CHD-MLP-Regression_250e.tflite"
    TFMODEL_DIR = os.path.join(current_app.root_path,"ai_models/cvd/chd_mlp",TFLITE_FILE) 
    
    #load Dataframe with mean and standard devs to scale patientData - df.describ()
    DATAFRAME_FILE = "CHD-DataFrame-Desc.csv"
    Desc_Data_file = os.path.join(current_app.root_path,"ai_models/cvd/chd_mlp",DATAFRAME_FILE)
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
    patientData_np = np.array(patientData_arr)
    # Transform the data
    patientData_scaled = (patientData_np-mean_n_train)/std_n_train
    ## Reshape input data
    patientData_scaled_rs = patientData_scaled.values.reshape(1,12) 

    ################################################################################
    #TFLite Inference
    interpreter = tf.lite.Interpreter(model_path=TFMODEL_DIR)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    input_data = np.array(patientData_scaled_rs, dtype=np.float32)
    print("patientData_scaled_rs:",patientData_scaled_rs)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    ## TFlite Predict Patient input data
    patientPred_tf = interpreter.get_tensor(output_index)
    # print("TFlite - Patient Predicted values are: \n", patientPred_tf)
    ################################################################################

    # Predict Patient input data
    # patientPred = model_mlp.predict(patientData_scaled_rs)
    # print("Patient Predicted values are: \n", patientPred)    

    ##### Threshold for CHD risk #####
    # chd_pred = patientPred[0][0]
    chd_pred = patientPred_tf[0][0]

    if chd_pred > 1:
        chd_pred = .999 #risk prediction cannot be 100%
    elif chd_pred < 0:
        chd_pred = 0

    #CHD Risk Grading
    # https://my.clevelandclinic.org/health/articles/17085-heart-risk-factor-calculators
    risk_category = ""

    if 0 <= chd_pred <= 0.25: # and round(value,2)==value:
        risk_category = "Low"      
    elif 0.25 <= chd_pred <= 0.5: # and round(value,2)==value:
        risk_category = "Borderline"      
    elif 0.5 <= chd_pred <= 0.75: # and round(value,2)==value:
        risk_category = "Intermediate"      
    elif 0.75 <= chd_pred <= 1: # and round(value,2)==value:
        risk_category = "High"      

    chd_pred = round(chd_pred, 4)
    patient_pred_dict = {
        "chd_probability": chd_pred,
        "risk_Category": risk_category
        }
    return patient_pred_dict


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


