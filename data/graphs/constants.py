CLASS_NAMES = ['Attractive', 'Male', 'Young', 'Receding_Hairline']
IMAGE_DIR = '../partitioned-data/test/'
CLASS_COLOURS = {
    'Attractive': 'blue',
    'Male': 'green',
    'Young': 'red',
    'Receding_Hairline': 'purple'
}
ACTIVATION_COLOURS = {
    'red',
    'blue',
    'green'
}
METRIC_DESCRIPTIONS = {
    'accuracy': 'Percentage of correct predictions',
    'precision': 'True positive predictions / All positive predictions',
    'recall': 'True positive predictions / All correct positive cases',
    'specificity': 'True negative predictions / All negative predictions',
    'log_loss': 'Confidence in predictions',
    'f1_score': 'Balance of precision and recall'
}
EPOCHS = range(4, 33, 4)
EPOCH_MODELS_DIR = '../../models/epoch/'
EPOCH_MODELS = [
    'e_4_model.h5',
    'e_8_model.h5',
    'e_12_model.h5',
    'e_16_model.h5',
    'e_20_model.h5',
    'e_24_model.h5',
    'e_28_model.h5',
    'e_32_model.h5'
]
BATCH_SIZES = [32, 64, 128, 256, 512, 1048]
BATCH_MODELS_DIR = '../../models/batch-size/'
BATCH_SIZE_MODELS = [
    'bs_32_model.h5',
    'bs_64_model.h5',
    'bs_128_model.h5',
    'bs_256_model.h5',
    'bs_512_model.h5',
    'bs_1048_model.h5']
# LEARNING_RATES = [0.000001, 0.00001, 0.0001, 0.001, 0.01,  0.1]
LEARNING_RATES = [1, 2, 3, 4, 5, 6]
LEARNING_RATE_MODELS_DIR = '../../models/learning-rate/'
LEARNING_RATE_MODELS = [
    'lr_1e-1_model.h5',
    'lr_1e-2_model.h5',
    'lr_1e-3_model.h5',
    'lr_1e-4_model.h5',
    'lr_1e-5_model.h5',
    'lr_1e-6_model.h5'
]
ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'tanh']
ACTIVATION_MODELS_DIR = '../../models/activation/'
ACTIVATION_MODELS = [
    'relu_model.h5',
    'sigmoid_model.h5',
    'tanh_model.h5'
]
