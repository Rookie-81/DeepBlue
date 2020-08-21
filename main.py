from functions import *
from sklearn.model_selection import train_test_split

path = 'DataSet'   #Setting .CSV File and IMG file path
###########################
data = importdataset(path)    #Loading data into pandas format
###########################
DataBalancing(data,display=False)   #Skimming through the captured images to reduce  dataset size and optimizing the dataset
###########################
imgpath, steering =  DataLoading(path,data)   # Converting intno numpy arrays and unpacking data

##########################
xTrain , xVal , yTrain , yVal = train_test_split(imgpath,steering,test_size=0.2,random_state=5)        #Splitting data into training and validation

##########################

