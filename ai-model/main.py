from train_model import train_and_save_model

EPOCHS = 10  # Number of times the model evaluates the entire data set
BATCH_SIZE = 1048  # Number of images the model processes before updating weights
LEARNING_RATE = 0.0001  # Rate that weights will be adjusted after each batch
ACTIVATION = 'sigmoid'  # Functioned applied to the output of the final layer

# EPOCH TEST
train_and_save_model(4, BATCH_SIZE, LEARNING_RATE, ACTIVATION, 'e_4_model.h5')
train_and_save_model(8, BATCH_SIZE, LEARNING_RATE, ACTIVATION, 'e_8_model.h5')
train_and_save_model(12, BATCH_SIZE, LEARNING_RATE, ACTIVATION, 'e_12_model.h5')
train_and_save_model(16, BATCH_SIZE, LEARNING_RATE, ACTIVATION, 'e_16_model.h5')
train_and_save_model(20, BATCH_SIZE, LEARNING_RATE, ACTIVATION, 'e_20_model.h5')
train_and_save_model(24, BATCH_SIZE, LEARNING_RATE, ACTIVATION, 'e_24_model.h5')
train_and_save_model(28, BATCH_SIZE, LEARNING_RATE, ACTIVATION, 'e_28_model.h5')
train_and_save_model(32, BATCH_SIZE, LEARNING_RATE, ACTIVATION, 'e_32_model.h5')

# BATCH_SIZE TEST
train_and_save_model(EPOCHS, 32, LEARNING_RATE, ACTIVATION, 'bs_32_model.h5')
train_and_save_model(EPOCHS, 64, LEARNING_RATE, ACTIVATION, 'bs_64_model.h5')
train_and_save_model(EPOCHS, 128, LEARNING_RATE, ACTIVATION, 'bs_128_model.h5')
train_and_save_model(EPOCHS, 256, LEARNING_RATE, ACTIVATION, 'bs_256_model.h5')
train_and_save_model(EPOCHS, 512, LEARNING_RATE, ACTIVATION, 'bs_512_model.h5')
train_and_save_model(EPOCHS, 1048, LEARNING_RATE, ACTIVATION, 'bs_1048_model.h5')

# LEARNING_RATE TEST
train_and_save_model(EPOCHS, BATCH_SIZE, 0.000001, ACTIVATION, 'lr_1e-6_model.h5')
train_and_save_model(EPOCHS, BATCH_SIZE, 0.00001, ACTIVATION, 'lr_1e-5_model.h5')
train_and_save_model(EPOCHS, BATCH_SIZE, 0.0001, ACTIVATION, 'lr_1e-4_model.h5')
train_and_save_model(EPOCHS, BATCH_SIZE, 0.001, ACTIVATION, 'lr_1e-3_model.h5')
train_and_save_model(EPOCHS, BATCH_SIZE, 0.01, ACTIVATION, 'lr_1e-2_model.h5')
train_and_save_model(EPOCHS, BATCH_SIZE, 0.1, ACTIVATION, 'lr_1e-1_model.h5')

# ACTIVATION TEST
train_and_save_model(EPOCHS, BATCH_SIZE, LEARNING_RATE, 'relu', 'relu_model.h5')
train_and_save_model(EPOCHS, BATCH_SIZE, LEARNING_RATE, 'sigmoid', 'sigmoid_model.h5')
train_and_save_model(EPOCHS, BATCH_SIZE, LEARNING_RATE, 'tanh', 'tanh_model.h5')

# OPTIMAL MODEL
train_and_save_model(28, 32, 0.000001, 'tanh', 'final_model.h5')
