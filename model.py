import os
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
import datetime

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Permisos solo para subir archivos
SCOPES = ['https://www.googleapis.com/auth/drive.file']


def get_drive_service():
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    return service


def upload_to_drive(file_path, folder_id=None):
    service = get_drive_service()
    file_metadata = {
        'name': os.path.basename(file_path)
    }

    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, resumable=True)

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    print(f"✅ Archivo subido a Google Drive. ID: {file.get('id')}")


baseDir  = os.path.dirname(os.path.abspath(__file__))
imagesDir = os.path.join(baseDir, 'DS')
outputDir = os.path.join(baseDir, 'model')
checkpointDir = os.path.join(baseDir, 'Checkpoints')
os.makedirs(outputDir, exist_ok=True)
os.makedirs(checkpointDir, exist_ok=True)

imageSize    = (256, 256)
seed         = 42
batchSize    = 32
valTestSplit = 0.30

subdirs = sorted([
    d for d in os.listdir(imagesDir)
    if os.path.isdir(os.path.join(imagesDir, d))])

try:
  classNames = sorted(subdirs, key=lambda s: float(s))
except ValueError as e:
  raise ValueError(f"Valores numericos, revisar: {subdirs}") from e

bcsValues = [float(c) for c in classNames]
print("Clases detectadas (BCS):", bcsValues)

trainDataset = tf.keras.utils.image_dataset_from_directory(
    imagesDir,
    labels='inferred',
    label_mode='int',
    class_names=classNames,
    color_mode='rgb',
    batch_size=batchSize,
    image_size=imageSize,
    shuffle=True,
    seed=seed,
    validation_split=valTestSplit,
    subset='training'
)

valTestDataset = tf.keras.utils.image_dataset_from_directory(
    imagesDir,
    labels='inferred',
    label_mode='int',
    class_names=classNames,
    color_mode='rgb',
    batch_size=batchSize,
    image_size=imageSize,
    shuffle=True,
    seed=seed,
    validation_split=valTestSplit,
    subset='validation'
)

bcsLookupTable = tf.constant(bcsValues, dtype=tf.float32)

@tf.function
def putRegressionLabels(images, classIndices):
    labels = tf.gather(bcsLookupTable, classIndices)
    labels = tf.expand_dims(labels , axis=-1)
    return images, labels

trainRegDataset = trainDataset.map(
  putRegressionLabels, num_parallel_calls=tf.data.AUTOTUNE)
valTestRegDataset = valTestDataset.map(
  putRegressionLabels,  num_parallel_calls=tf.data.AUTOTUNE)

def splitValidationAndTest(dataset):
    datasetSize = dataset.cardinality().numpy()
    halfSize = datasetSize // 2
    valDataset  = dataset.take(halfSize)
    testDataset = dataset.skip(halfSize)
    return valDataset, testDataset

valRegDataset, testRegDataset = splitValidationAndTest(valTestRegDataset)

dataAugmentationLayer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.1),
], name="dataAugmentation")

rescaleLayer = tf.keras.layers.Rescaling(1./255, name="rescaleLayer")

def augmentThenScale(images, labels):
    images = dataAugmentationLayer(images, training=True)
    images = rescaleLayer(images)
    return images, labels

def onlyScale(images, labels):
    images = rescaleLayer(images)
    return images, labels

autoTune = tf.data.AUTOTUNE

trainDataset = (trainRegDataset
           .shuffle(8 * batchSize, seed=seed, reshuffle_each_iteration=True)
           .map(augmentThenScale, num_parallel_calls=autoTune)
           .prefetch(autoTune))

valDataset = (valRegDataset
         .map(onlyScale, num_parallel_calls=autoTune)
         .prefetch(autoTune))

testDataset = (testRegDataset
          .map(onlyScale, num_parallel_calls=autoTune)
          .prefetch(autoTune))

def buildBcsCnnModel(inputShape):
  model = tf.keras.Sequential([
      layers.Conv2D(32, kernel_size=3, padding='same', activation='relu',
                    kernel_regularizer=regularizers.l2(
                        1e-4), input_shape=inputShape),
      layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(0.25),
      layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
      layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(0.25),
      layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
      layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(0.4),
      layers.Flatten(),
      layers.Dense(256, activation='relu'),
      layers.Dropout(0.4),
      layers.Dense(128, activation='relu'),
      layers.Dense(1, activation='linear', name='bcs_output')
      ])
  return model

inputShape = (256, 256, 3)
model = buildBcsCnnModel(inputShape)
model.summary()

"""## Métricas"""

def mae(yTrue, yPred):
    return tf.reduce_mean(tf.abs(yTrue - yPred))

def bias(yTrue, yPred):
    return tf.reduce_mean(yPred - yTrue)

def variance(yTrue, yPred):
    meanPred = tf.reduce_mean(yPred)
    return tf.reduce_mean(tf.square(yPred - meanPred))

def accuracyWithinHalfBcs(yTrue, yPred):
    absError  = tf.abs(yTrue - yPred)
    return tf.reduce_mean(tf.cast(absError  <= 0.5, tf.float32))

"""## Compilar"""

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='mae',
    metrics=[mae, bias, variance, accuracyWithinHalfBcs]
)

# Rutas de guardado
checkpointPath = os.path.join(checkpointDir, "bcsCnnWeights.weights.h5")
logsDir = os.path.join(
    outputDir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Guardar sólo los pesos
checkpointCallback = ModelCheckpoint(
    filepath=checkpointPath,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_mae',
    mode='min',
    verbose=1
)

# Detener si el modelo deja de mejorar
earlyStoppingCallback = EarlyStopping(
    monitor='val_mae',
    mode='min',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

tensorboardCallback = TensorBoard(log_dir=logsDir)
callbacksList = [checkpointCallback, earlyStoppingCallback, tensorboardCallback]

"""## Entrenamiento"""

epochs = 2

history = model.fit(
    trainDataset,
    validation_data=valDataset,
    epochs=epochs,
    callbacks=callbacksList,
    verbose=1
)

"""Guardado modelo completo Keras"""

finalModelPath = os.path.join(outputDir, "MuuMetricsBcsModel.keras")
model.save(finalModelPath)
print(f"\n✅ Modelo completo guardado en: {finalModelPath}")

drive_folder_id = "1PEc1_SCQBfT-vt7FMDS-VEswDdY5vFTe" #### CHANGE THIS ID


# Subir modelo `.keras`
upload_to_drive(finalModelPath, drive_folder_id)

# Subir los pesos (checkpoints)
upload_to_drive(checkpointPath, drive_folder_id)
