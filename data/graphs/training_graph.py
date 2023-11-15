import matplotlib.pyplot as plt

# Epochs
epochs = range(1, 29)

# Training and validation accuracy
training_accuracy = [0.5166, 0.5520, 0.5823, 0.6100, 0.6331, 0.6549, 0.6761, 0.6918, 0.7074, 0.7202, 0.7330, 0.7434, 0.7527, 0.7609, 0.7679, 0.7743, 0.7793, 0.7849, 0.7891, 0.7926, 0.7967, 0.8001, 0.8026, 0.8053, 0.8077, 0.8101, 0.8113, 0.8125]
validation_accuracy = [0.5418, 0.5980, 0.6554, 0.6567, 0.6883, 0.7040, 0.7391, 0.7412, 0.7619, 0.7613, 0.7694, 0.7832, 0.7871, 0.7930, 0.7914, 0.8006, 0.8039, 0.8082, 0.8101, 0.8137, 0.8166, 0.8185, 0.8216, 0.8231, 0.8252, 0.8199, 0.8273, 0.8285]

# Training and validation loss
training_loss = [1.6112, 1.5165, 1.4558, 1.4027, 1.3536, 1.3058, 1.2593, 1.2161, 1.1739, 1.1338, 1.0948, 1.0584, 1.0232, 0.9902, 0.9591, 0.9297, 0.9023, 0.8755, 0.8515, 0.8278, 0.8055, 0.7844, 0.7647, 0.7462, 0.7281, 0.7117, 0.6958, 0.6821]
validation_loss = [1.5205, 1.4629, 1.4072, 1.3674, 1.3172, 1.2730, 1.2188, 1.1819, 1.1301, 1.1013, 1.0656, 1.0229, 0.9875, 0.9501, 0.9359, 0.8984, 0.8704, 0.8492, 0.8258, 0.8010, 0.7771, 0.7478, 0.7364, 0.7176, 0.7006, 0.6973, 0.6651, 0.6519]

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plotting the training and validation accuracy
ax1.plot(epochs, training_accuracy, label='Training Accuracy')
ax1.plot(epochs, validation_accuracy, label='Validation Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Plotting the training and validation loss
ax2.plot(epochs, training_loss, label='Training Loss')
ax2.plot(epochs, validation_loss, label='Validation Loss')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
