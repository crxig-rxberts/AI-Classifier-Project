import os
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np


class ImageClassifier:
    def __init__(self, num_classes, learning_rate=0.0001, activation='relu'):
        self.input_shape = (178, 218, 3)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.activation = activation
        self.model = self.build_model()
        self.validation_metrics = {}

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(8, (3, 3), activation=self.activation, input_shape=self.input_shape),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(16, (3, 3), activation=self.activation),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(32, (3, 3), activation=self.activation),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(64, (3, 3), activation=self.activation),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(128, (3, 3), activation=self.activation),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, self.activation, kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.6),
            keras.layers.Dense(self.num_classes, activation='sigmoid')
        ])

        optimizer = keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

        return model

    def train(self, train_generator, val_generator, epochs):
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_steps=val_generator.samples // val_generator.batch_size
        )

        y_true = []
        y_pred_list = []

        # Collect all y_true and y_pred values
        for i in range(val_generator.samples // val_generator.batch_size):
            x_val_batch, y_val_batch = next(val_generator)
            y_pred_batch = (self.model.predict(x_val_batch) > 0.5).astype(int)
            y_true.extend(y_val_batch)
            y_pred_list.extend(y_pred_batch)

        y_pred = np.array(y_pred_list)
        y_true = np.array(y_true)

        probabilities = self.model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size)
        auc_roc_per_class = [
            roc_auc_score(y_true[:, i], probabilities[:, i]) if np.unique(y_true[:, i]).size > 1 else np.nan for i in
            range(self.num_classes)
        ]
        specificity_per_class = [self.specificity_score(y_true[:, i], y_pred[:, i], 1) for i in range(self.num_classes)]

        self.validation_metrics = {
            'validation_loss': history.history['val_loss'],
            'validation_accuracy': history.history['val_binary_accuracy'],
            'validation_precision': precision_score(y_true, y_pred, average='macro', zero_division=1),
            'validation_recall': recall_score(y_true, y_pred, average='macro'),
            'validation_f1_score': f1_score(y_true, y_pred, average='macro'),
            'validation_specificity': specificity_per_class,
            'validation_auc_roc': auc_roc_per_class,
            'validation_confusion_matrix': [confusion_matrix(y_true[:, i], y_pred[:, i]).tolist() for i in
                                            range(self.num_classes)]
        }

    def predict(self, x):
        predictions = self.model.predict(x)
        thresholded_predictions = (predictions > 0.5).astype(int)
        confidence_percentages = predictions * 100
        return predictions, thresholded_predictions, confidence_percentages

    @staticmethod
    def specificity_score(y_true, y_pred, label):
        true_negatives = np.sum((y_true == label) & (y_pred == label))
        false_positives = np.sum((y_true != label) & (y_pred == label))
        if true_negatives + false_positives == 0:
            return np.nan
        return true_negatives / (true_negatives + false_positives)

    def get_validation_metrics(self, val_generator):
        y_true = []
        y_pred_list = []

        for _ in range(len(val_generator)):
            x_val_batch, y_val_batch = next(val_generator)
            y_pred_batch = (self.model.predict(x_val_batch) > 0.5).astype(int)
            y_true.extend(y_val_batch)
            y_pred_list.extend(y_pred_batch)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred_list)

        # Calculate metrics
        validation_metrics = {
            'validation_precision': precision_score(y_true, y_pred, average='macro', zero_division=1),
            'validation_recall': recall_score(y_true, y_pred, average='macro'),
            'validation_f1_score': f1_score(y_true, y_pred, average='macro'),
        }

        # Calculate AUC for each class
        probabilities = self.model.predict(val_generator)
        for i in range(self.num_classes):
            if np.unique(y_true[:, i]).size > 1:
                validation_metrics[f'class_{i}_auc'] = roc_auc_score(y_true[:, i], probabilities[:, i])

        # Calculate specificity for each class
        for i in range(self.num_classes):
            validation_metrics[f'class_{i}_specificity'] = self.specificity_score(y_true[:, i], y_pred[:, i], 1)

        # Add confusion matrices
        validation_metrics['confusion_matrices'] = [confusion_matrix(y_true[:, i], y_pred[:, i]).tolist() for i in
                                                    range(self.num_classes)]

        return validation_metrics
