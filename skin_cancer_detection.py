
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
path="D:\exceed\archive\hmnist_28_28_RGB.csv"
df=pd.read_csv("D:\\exceed\\archive\\hmnist_28_28_RGB.csv")
df.tail() 
fractions=np.array([0.8,0.2])
df=df.sample(frac=1)
train_set, test_set = np.array_split(df, (fractions[:-1].cumsum() * len(df)).astype(int))
df.label.unique()
classes={
    0:('akiec', 'actinic keratoses and intraepithelial carcinomae'),
         
    1:('bcc' , 'basal cell carcinoma'),
         
    2:('bkl', 'benign keratosis-like lesions'),
         
    3:('df', 'dermatofibroma'),
         
    4:('nv', ' melanocytic nevi'),
         
    5:('vasc', ' pyogenic granulomas and hemorrhage'),
         
    6:('mel', 'melanoma'),
}
y_train=train_set['label']
x_train=train_set.drop(columns=['label'])
y_test=test_set['label']
x_test=test_set.drop(columns=['label'])
columns=list(x_train)
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import seaborn as sns
sns.countplot(train_set['label'])
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
train_data = np.column_stack((x_train, y_train))
majority_class = train_data[train_data[:, -1] == 0]
minority_class = train_data[train_data[:, -1] == 1]
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
upsampled_data = np.concatenate((majority_class, minority_upsampled), axis=0)
x_train_resampled = upsampled_data[:, :-1]
y_train_resampled = upsampled_data[:, -1]
x_combined = np.concatenate((x_train_resampled, x_val), axis=0)
y_combined = np.concatenate((y_train_resampled, y_val), axis=0)
sns.countplot(y_train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
import tensorflow as tf
get_ipython().run_line_magic('time', '')
model = Sequential()
model.add(Conv2D(16, 
                 kernel_size = (3,3), 
                 input_shape = (28, 28, 3), 
                 activation = 'relu', 
                 padding = 'same'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(tf.keras.layers.BatchNormalization())

model.add(Conv2D(32, 
                 kernel_size = (3,3), 
                 activation = 'relu'))

model.add(Conv2D(64, 
                 kernel_size = (3,3), 
                 activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(tf.keras.layers.BatchNormalization())

model.add(Conv2D(128, 
                 kernel_size = (3,3), 
                 activation = 'relu'))

model.add(Conv2D(256, 
                 kernel_size = (3,3), 
                 activation = 'relu'))

model.add(Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(7,activation='softmax'))
model.summary()
callback = tf.keras.callbacks.ModelCheckpoint(filepath="D:\\pbl\\Skin_Cancer_Detection_MNIST-main\\Skin_Cancer_Detection_MNIST-main\\best_model.h5",
                                              monitor='val_acc', 
                                              mode='max',
                                              verbose=1,
                                              save_best_only=True)
get_ipython().run_line_magic('time', '')
from tensorflow.keras import optimizers
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
from datetime import datetime
start_time = datetime.now()
history = model.fit(x_train,
                    y_train,
                    validation_split=0.2,
                    batch_size = 128,
                    epochs = 150,
                    shuffle=True,
                    callbacks=[callback])
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
model.load_weights("D:\\pbl\\Skin_Cancer_Detection_MNIST-main\\Skin_Cancer_Detection_MNIST-main\\best_model.h5")
x_test=np.array(x_test).reshape(-1,28,28,3)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
import PIL
image=PIL.Image.open("D:\\exceed\\archive\\HAM10000_images_part_1\\ISIC_0025628.jpg")
image=image.resize((28,28))
img=x_test[1]
img=np.array(image).reshape(-1,28,28,3)
result=model.predict(img)
print(result[0])
result=result.tolist()
max_prob=max(result[0])
class_ind=result[0].index(max_prob)
print(classes[class_ind])



