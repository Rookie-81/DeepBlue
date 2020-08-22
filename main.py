from functions import *

print('Loading...')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
######################################################################
path = 'DataSet'  # Setting .CSV File and IMG file path
###########################
data = importDataInfo(path)  # Loading data into pandas format

###########################
data = balanceData(data,display=False)    # Skimming through the captured images to reduce  dataset size and optimizing the dataset
###########################
imagesPath, steerings = DataLoading(path,data)


##########################
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))
model = createModel()
model.summary()
#########################
history = model.fit(batchGen(xTrain, yTrain, 100, 1),
                    steps_per_epoch=300,
                    epochs=10,
                    validation_data=batchGen(xVal, yVal, 100, 0),
                    validation_steps=200)
################################
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val-loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
