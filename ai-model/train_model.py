import os
import time
import pandas as pd
from tensorflow import keras

from model import ImageClassifier

CLASS_NAMES = [
    "Attractive",
    "Male",
    "Young",
    "Receding_Hairline",
]
PRETRAIN_SAVE_DIRECTORY = '../models/'
ATTRIBUTE_CSV = '../data/archive/list_attr_celeba.csv'
TEST_DIRECTORY = '../data/partitioned-data/test/'
TRAIN_DIRECTORY = '../data/partitioned-data/train/'
IMAGE_SIZE = (178, 218)

# Load and preprocess labels
df = pd.read_csv(ATTRIBUTE_CSV)
df.replace(-1, 0, inplace=True)
df['image_id'] = df['image_id'].apply(lambda file_name: os.path.join(TRAIN_DIRECTORY, file_name))


def train_and_save_model(epochs, batch_size, learning_rate, activation, model_name):
    datagen_kwargs = {
        'rescale': 1. / 255,
        'validation_split': 0.1
    }

    dataflow_kwargs = {
        'target_size': IMAGE_SIZE,
        'batch_size': batch_size,
        'class_mode': 'raw'
    }

    train_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs,
                                                                 rotation_range=15,
                                                                 width_shift_range=0.1,
                                                                 height_shift_range=0.1,
                                                                 shear_range=0.1,
                                                                 zoom_range=0.1,
                                                                 horizontal_flip=True,
                                                                 fill_mode='nearest')

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col='image_id',
        y_col=CLASS_NAMES,
        subset='training',
        **dataflow_kwargs
    )

    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col='image_id',
        y_col=CLASS_NAMES,
        subset='validation',
        **dataflow_kwargs
    )

    model = ImageClassifier(len(CLASS_NAMES), learning_rate, activation)
    print("-------- Model Summary:")
    model.model.summary()
    start_time = time.time()
    print(f"-------- Training Model...")

    # ---------- Train model
    model.train(train_generator, validation_generator, epochs)
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"-------- Training Time: {int(hours)}h:{int(minutes)}m:{int(seconds)}s.")

    # ---------- Get validation metrics
    validation_metrics = model.get_validation_metrics(validation_generator)
    print("\n-------- Validation Metrics:")
    for metric, value in validation_metrics.items():
        print(f"{metric}: {value}")

    # ---------- Save Model
    model_path = os.path.join(PRETRAIN_SAVE_DIRECTORY, model_name)
    model.model.save(model_path)
    print(f"-------- Model saved to {model_path}")
