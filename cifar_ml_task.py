import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the CIFAR-10 Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the images to the range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Step 2: Build the CNN Model
def build_cnn_model():
    model = models.Sequential()
    
    # 1st Convolutional Layer + Max Pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 2nd Convolutional Layer + Max Pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 3rd Convolutional Layer + Max Pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Flatten the output and add Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # Output layer
    model.add(layers.Dense(10, activation='softmax'))  # 10 classes
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Step 3: Train the Model with Early Stopping
def train_model(model):
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    history = model.fit(train_images, train_labels, epochs=50, batch_size=64,
                        validation_data=(test_images, test_labels),
                        callbacks=[early_stopping])

    # Save the model
    model.save('cifar10_cnn_model.h5')
    
    return history

# Step 4: Evaluate the Model
def evaluate_model(model):
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    # Predict the test set
    y_pred = model.predict(test_images)
    y_pred_classes = y_pred.argmax(axis=-1)
    y_true_classes = test_labels.argmax(axis=-1)

    # Calculate Precision, Recall, F1-Score
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Step 5: Implement Transfer Learning using ResNet50 (Optional)
def build_transfer_learning_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False  # Freeze the base model layers

    model = models.Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 6: Learning Rate Scheduler (Optional)
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.1

lr_scheduler = LearningRateScheduler(scheduler)

# Main Execution
if __name__ == "__main__":
    # 1. Build and train CNN model
    cnn_model = build_cnn_model()
    history = train_model(cnn_model)

    # 2. Evaluate CNN model
    evaluate_model(cnn_model)

    # 3. Implement Transfer Learning and train ResNet50 model (Optional)
    resnet_model = build_transfer_learning_model()
    history_resnet = resnet_model.fit(train_images, train_labels, epochs=50, batch_size=64,
                                      validation_data=(test_images, test_labels),
                                      callbacks=[EarlyStopping(monitor='val_loss', patience=5), lr_scheduler])

    # 4. Evaluate ResNet50 model
    evaluate_model(resnet_model)
