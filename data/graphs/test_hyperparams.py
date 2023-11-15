from constants import CLASS_NAMES, EPOCHS, EPOCH_MODELS, EPOCH_MODELS_DIR, BATCH_SIZES, BATCH_SIZE_MODELS, \
    BATCH_MODELS_DIR, LEARNING_RATE_MODELS_DIR, LEARNING_RATE_MODELS, LEARNING_RATES, ACTIVATION_MODELS, \
    ACTIVATION_MODELS_DIR
from functions import get_metrics_from_models, init_test_data_generator, plot_metrics, plot_activation_function_metrics, \
    plot_activation_metrics, plot_final_vs_b32_metrics

test_datagen, test_generator, df = init_test_data_generator()
y_test = df[CLASS_NAMES].values

epoch_metrics = get_metrics_from_models(EPOCH_MODELS, test_generator, y_test, EPOCH_MODELS_DIR)
plot_metrics(epoch_metrics, EPOCHS, 'Number of Epochs', 'Measuring varying EPOCHS')

batch_metrics = get_metrics_from_models(BATCH_SIZE_MODELS, test_generator, y_test, BATCH_MODELS_DIR)
plot_metrics(batch_metrics, BATCH_SIZES, 'Batch Size', 'Measuring varying BATCH SIZES')

learning_rate_metrics = get_metrics_from_models(LEARNING_RATE_MODELS, test_generator, y_test, LEARNING_RATE_MODELS_DIR)
plot_metrics(batch_metrics, LEARNING_RATES, 'Learning Rate', 'Measuring varying LEARNING RATES')

activation_metrics = get_metrics_from_models(ACTIVATION_MODELS, test_generator, y_test, ACTIVATION_MODELS_DIR)
plot_activation_metrics(activation_metrics)

FINAL_TEST_MODELS = [
    'final_model.h5',
    'bs_32_model.h5'
]
FINAL_TEST_DIR = "../../models/final/"
final_metrics = get_metrics_from_models(FINAL_TEST_MODELS, test_generator, y_test, FINAL_TEST_DIR)
print(final_metrics)
plot_final_vs_b32_metrics(final_metrics)