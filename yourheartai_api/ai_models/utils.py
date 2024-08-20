            
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

##mask r-cnn imports
# evaluate the mask rcnn model on the stenosis dataset
import mrcnn
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import cv2    

            
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

def getStenosisPrediction(filename):
    LOGS_DIR = "model\logs"
    MODEL_DIR = "model\cvd\stenosis_mask_r-cnn\Stenosis_mrcnn_train-6661s-832v-100spe-10e.h5"
    IMAGE_DIR = ""
    # load the class label names from disk, one label per line
    CLASS_NAMES = ['BG', 'stenosis']

    class StenosisPredConfig(mrcnn.config.Config):
        # Give the configuration a recognizable name
        NAME = "stenosis_inference"

        # set the number of GPUs to use along with cthe number of images per GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
        NUM_CLASSES = len(CLASS_NAMES)
        USE_MINI_MASK = False
        DETECTION_MIN_CONFIDENCE = 0.6

    # Create Prediction Config
    predCfg = StenosisPredConfig()
    # predCfg.display()C

    # Initialize the Mask R-CNN model for inference and then load the weights.
    # This step builds the Keras model architecture.
    model = mrcnn.model.MaskRCNN(mode="inference",
                                config=predCfg,
                                model_dir=LOGS_DIR)
    # Load the weights into the model.
    model.load_weights(MODEL_DIR, by_name=True)
    
    # load the input image, convert it from BGR to RGB channel
    image = cv2.imread(IMAGE_DIR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform a forward pass of the network to obtain the results
    r = model.detect([image], verbose=0)
    # Get the results for the first image.
    r = r[0]

    #Save Output Images
    ## See solution here: github.com/matterport/Mask_RCNN/issues/134

    patient_sten_pred_dict = {}
    return patient_sten_pred_dict



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
