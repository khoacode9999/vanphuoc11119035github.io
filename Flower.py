#%%
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#%matplotlib inline  
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)
#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
#%%
Path = "D:\\Flower recognition\\flowers"
print(os.listdir("D:\\Flower recognition\\flowers"))

# %%
X = []
Z = []

flower_daisy = Path + "\\daisy"
flower_sunflower =  Path + "\\sunflower"
flower_tulip = Path + "\\tulip"
flower_dandelion = Path + "\\dandelion"
flower_rose = Path + "\\rose"

# %%
def assign_label(img, flower_type):
    return flower_type
    
# %%
def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
        img = cv2.resize(img, (150,150))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        X.append(np.array(img))
        Z.append(str(label))
    
# %%
make_train_data("Daisy",flower_daisy)
print(len(X))
print(len(Z))

make_train_data("Sunflower",flower_sunflower)
print(len(X))
print(len(Z))

make_train_data("Tulip",flower_tulip)
print(len(X))
print(len(Z))

make_train_data("Dandelion",flower_dandelion)
print(len(X))
print(len(Z))
 
make_train_data("Rose",flower_rose)
print(len(X))
print(len(Z))
# %%
f,ax = plt.subplots(7,5)
f.set_size_inches(15,15)
for i in range(7):
    for j in range(5):
        l = rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title("Flower: " + Z[l])

        
plt.tight_layout()
# %%
# Label Encoding the Y array (i.e. Daisy->0, Rose->1 etc...) & then One Hot Encoding
labelEncoder = LabelEncoder()
Y = labelEncoder.fit_transform(Z)
Y = to_categorical(Y,5)
X = np.array(X)
X = X/ 255
#%%


# %%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)
# %%
# modelling 
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), padding = "Same", activation="relu", input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding  = "Same", activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3), padding  = "Same", activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3), padding  = "Same", activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(5, activation="softmax"))
# %%
epochs = 25
batch_size = 32

# %%
# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,   # set each sample mean to 0
    featurewise_std_normalization=False,   # divide inputs by std of the dataset
    samplewise_std_normalization= False,   # divide each input by its std
    zca_whitening=False,   # dimesion reduction
    rotation_range=10,    # randomly rotate images in the range 10 degrees
    zoom_range=0.1,      # Randomly zoom image 10%
    width_shift_range=0.2,   # randomly shift images horizontally 20%
    height_shift_range=0.2,   # randomly shift images vertically 20%
    horizontal_flip=True,     # randomly flip images
    vertical_flip=False    # randomly flip images
)
datagen.fit(x_train)
# %%
# Compiling the Keras Model 

model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics = ["accuracy"])
# %%
# summary

model.summary()
# %%
# Fitting on the Training set and making predcitons on the Validation set
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size = batch_size), 
                              epochs= epochs, 
                              validation_data=(x_test,y_test), 
                              verbose = 1, 
                              steps_per_epoch=x_train.shape[0] // batch_size)
# %%
from tensorflow.keras.models import save_model
save_model(model, "Flower_detection.h5")

# %%
from numpy import loadtxt
from tensorflow.keras.models import load_model
model = load_model("Flower_detection.h5")
#%%
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["train","test"])
plt.show()
# %%
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["train","test"])
plt.show()
# %%
# getting predictions on validation set
pred = model.predict(x_test)
predict_digits = np.argmax(pred, axis = 1)
# %%
# We will create two arrays to sort the correct and incorrect predictions and throw them in
i = 0
right_class = []
wrong_class = []

for i in range(len(y_test)):
    if(np.argmax(y_test[i]) == predict_digits[i]):
        right_class.append(i)
    if(len(right_class) == 10):
        break

i = 0

for i in range(len(y_test)):
    if(np.argmax(y_test[i]) != predict_digits[i]):
        wrong_class.append(i)
    if(len(wrong_class) == 10):
        break
# %%
count = 0
f,ax = plt.subplots(3,2)
f.set_size_inches(15,15)
for i in range(3):
    for j in range(2):
        ax[i,j].imshow(x_test[right_class[count]])
        ax[i,j].set_title("Predicted Watch :"+str(labelEncoder.inverse_transform([predict_digits[right_class[count]]]))
                          +"\n"+"Actual Watch : "+str(labelEncoder.inverse_transform([np.argmax(y_test[right_class[count]])])))

        plt.tight_layout()
        count += 1
# %%
count = 0
f,ax = plt.subplots(3,2)
f.set_size_inches(15,15)
for i in range(3):
    for j in range(2):
        ax[i,j].imshow(x_test[wrong_class[count]])
        ax[i,j].set_title("Predicted Watch :"+str(labelEncoder.inverse_transform([predict_digits[wrong_class[count]]]))
                          +"\n"+"Actual Watch : "+str(labelEncoder.inverse_transform([np.argmax(y_test[wrong_class[count]])])))

        plt.tight_layout()
        count += 1
# %% - Manual testing
X_Manual = []
Path_test = "D:\\Flower recognition\\Test\Test.jpg"
img_test = cv2.imread(Path_test, cv2.IMREAD_COLOR)
img_test = cv2.resize(img_test, (150,150))
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
X_Manual.append(np.array(img_test))
X_Manual = np.array(X_Manual)
X_Manual = X_Manual/ 255
pred_manual = model.predict(X_Manual)
predict_digits_manual = np.argmax(pred_manual, axis = 1)

def number_to_string(argument):
    match argument:
        case 1:
            return "Daisy"
        case 2:
            return "Sunflower"
        case 3:
            return "Tulip"
        case 4:
            return "Dandelion"
        case 5:
            return "Rose"
        case default:
            return "something"

print(number_to_string(predict_digits_manual))


# %%
