import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import keras
from tensorflow.keras import layers

#1 load the data and check missing value

df = pd.read_csv('datasets/raw_dataset.csv', header= 0)
print(df.head())
print(df.isnull().sum())

#2 process the firmness, normalize to 0 ~ 1 by using z-score

mu_firmness = df.firmness.mean()
stddev_firmness = df.firmness.std()
df['firmness'] = ((df['firmness'] - mu_firmness) / stddev_firmness).round(4)

#3 process the hue, using sin-cos encoding

radians = np.deg2rad(df['hue'])

df['hue_sin'] = np.sin(radians)
df['hue_cos'] = np.cos(radians)

df.drop(columns=['hue'], inplace=True)

#4 process the saturation and brightness, transform into real percentage value

df['saturation'] = df['saturation'] / 100
df['brightness'] = df['brightness'] / 100

#5 process the categorical value: color, using the one-hot encoding

df['is_darkGreen'] = (df['color_category'] == 'dark green').astype(int)
df['is_green'] = (df['color_category'] == 'green').astype(int)
df['is_purple'] = (df['color_category'] == 'purple').astype(int)
df['is_black'] = (df['color_category'] == 'black').astype(int)

df.drop(columns=['color_category'], inplace=True)

#6 process sound_db, weight_g, and size_cm3, using z-score to normalize

mu_sound = df.sound_db.mean()
mu_weight = df.weight_g.mean()
mu_size = df.size_cm3.mean()
stddev_sound = df.sound_db.std()
stddev_weight = df.weight_g.std()
stddev_size = df.size_cm3.std()

df['sound_db'] = (df['sound_db'] - mu_sound) / stddev_sound
df['weight_g'] = (df['weight_g'] - mu_weight) / stddev_weight
df['size_cm3'] = (df['size_cm3'] - mu_size) / stddev_size

#7 transform the y(target) into numbers(0 ~ 4)
df['ripeness'] = df['ripeness'].replace({'hard': 0,
                                         'pre-conditioned': 1,
                                         'breaking': 2,
                                         'firm-ripe': 3,
                                         'ripe': 4})
#8 store the final dataframe

print(df)
df.to_csv('datasets/processed_dataset.csv', index=False, encoding='utf-8-sig')

#9 divide the dataframe

np.random.seed(10)
df = df.sample(frac=1).reset_index(drop=True)

total = len(df)

train_size = int(len(df) * 0.6)
validation_size = int(len(df) * 0.2)
test_size = len(df) - train_size - validation_size

train_df = df.iloc[:train_size]
validation_df = df.iloc[train_size:(train_size+validation_size)]
test_df = df.iloc[train_size+validation_size:]

train_df.to_csv('datasets/train_dataset.csv', index=False, encoding='utf-8-sig')
validation_df.to_csv('datasets/validation_dataset.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('datasets/test_dataset.csv', index=False, encoding='utf-8-sig')

#10 separate the x and y
train_y = train_df['ripeness']
train_x = train_df.drop(columns=['ripeness'])
validation_y = validation_df['ripeness']
validation_x = validation_df.drop(columns=['ripeness'])
test_y = test_df['ripeness']
test_x = test_df.drop(columns=['ripeness'])

#11 build the neuron network

model = keras.Sequential(name='project_model')
model.add(keras.layers.Dense(16, activation='relu', input_shape=(12,)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(5, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#12 set models callback
model_path = 'project_models/'
os.makedirs(model_path, exist_ok=True)
log_path = 'project_models/project_logs/'
os.makedirs(log_path, exist_ok=True)

tensorboard = keras.callbacks.TensorBoard(log_dir=log_path)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=model_path + 'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode = 'max',
    verbose=1
)

#13 train model

history = model.fit(train_x,
          train_y,
          validation_data=(validation_x, validation_y),
          epochs=100,
          batch_size=10,
          callbacks=[tensorboard, checkpoint],
        )
#14 see the result

model.load_weights(model_path + 'best_model.h5')
loss, accuracy = model.evaluate(test_x, test_y)

print(f'accuracy: {accuracy * 100:6.1f}%')

#15 visualization

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)


plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('images/loss.png')
plt.show()

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)

plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('images/accuracy.png')
plt.show()

