CIFAR-10 Image Classification with Deep Learning
File: cifar10_cnn_model.py

python
Copy
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
Explanation of the Code:
Data Loading & Preprocessing:

The CIFAR-10 dataset is loaded, normalized, and converted to one-hot encoded labels.
CNN Model:

The CNN model consists of 3 convolutional layers, max-pooling layers, dropout for regularization, and dense layers for classification. It is compiled using Adam optimizer and categorical cross-entropy loss.
Model Training:

Early stopping is used to prevent overfitting. The model trains for a maximum of 50 epochs but stops early if validation loss does not improve for 5 consecutive epochs.
Model Evaluation:

After training, the model is evaluated using accuracy, precision, recall, and F1-score.
Training and validation accuracy/loss are plotted.
Transfer Learning (ResNet50):

A ResNet50 model is used for transfer learning, with the base layers frozen, and only the top layer is trained for CIFAR-10 classification.
Learning Rate Scheduling (Optional):

A learning rate scheduler is applied to decrease the learning rate after 10 epochs.
How to Upload to GitHub:
Create a new repository on GitHub.
Create a new file named cifar10_cnn_model.py in your local project directory and paste the above code into it.
Run the following commands in your terminal to push your code to GitHub:
bash
Copy
git init
git add cifar10_cnn_model.py
git commit -m "Initial commit for CIFAR-10 CNN and Transfer Learning"
git branch -M main
git remote add origin https://github.com/yourusername/yourrepository.git
git push -u origin main
Replace yourusername/yourrepository with your actual GitHub username and repository name.

Let me know if you need any further adjustments or explanations!

You said:
explain the workings of this for readme file
ChatGPT said:
Certainly! Below is an explanation of how the code works, which you can use in your README file for the GitHub repository.

CIFAR-10 Image Classification with Deep Learning
This project demonstrates the classification of images from the CIFAR-10 dataset using Deep Learning techniques. The primary model is a Convolutional Neural Network (CNN), and we also implement Transfer Learning using a pre-trained ResNet50 model to boost performance.

Project Overview
Dataset: The CIFAR-10 dataset contains 60,000 32x32 color images across 10 classes. The classes are mutually exclusive, meaning each image belongs to exactly one class.
Task: Build and train a deep learning model to classify these images into one of 10 classes using TensorFlow/Keras.
Techniques Used:
Convolutional Neural Networks (CNNs) for image classification.
Transfer Learning with ResNet50 to improve model performance.
Early Stopping to prevent overfitting.
Learning Rate Scheduling for optimizing training.
Model Evaluation using accuracy, precision, recall, and F1-score.
File Overview
The main code is contained in cifar10_cnn_model.py, which includes the following sections:

1. Data Loading & Preprocessing
The CIFAR-10 dataset is loaded directly from TensorFlow’s datasets module. The images are then normalized to a range of [0, 1] by dividing by 255.0. The labels are converted to one-hot encoding using to_categorical.

python
Copy
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
2. Building the CNN Model
The CNN model architecture consists of:

Convolutional Layers (Conv2D): For extracting features from the images.
MaxPooling Layers (MaxPooling2D): For downsampling and reducing the spatial dimensions.
Dropout Layer: To prevent overfitting by randomly setting a fraction of input units to zero during training.
Flatten Layer: To convert the 2D output from convolutional layers into a 1D vector.
Dense Layer: Fully connected layers for classification.
Output Layer: A softmax activation for multi-class classification.
python
Copy
def build_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
3. Training the Model with Early Stopping
The model is trained using the training data, and early stopping is implemented to monitor the validation loss. If the validation loss doesn’t improve for 5 consecutive epochs, the training stops early. This helps in preventing overfitting.

python
Copy
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_images, train_labels, epochs=50, batch_size=64,
                    validation_data=(test_images, test_labels),
                    callbacks=[early_stopping])
4. Evaluating the Model
After training, the model is evaluated using the test set. Various performance metrics, such as accuracy, precision, recall, and F1-score, are computed. Additionally, we plot the training and validation accuracy/loss curves for visual inspection of model performance over epochs.

python
Copy
test_loss, test_acc = model.evaluate(test_images, test_labels)
y_pred = model.predict(test_images)
y_pred_classes = y_pred.argmax(axis=-1)
y_true_classes = test_labels.argmax(axis=-1)

precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
5. Transfer Learning with ResNet50 (Optional)
For improved performance, the code also implements transfer learning using a pre-trained ResNet50 model. We freeze the layers of the ResNet50 model and only train the newly added top layers for CIFAR-10 classification. This leverages the learned features from the ImageNet dataset.

python
Copy
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False
model = models.Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
6. Learning Rate Scheduling (Optional)
The code also demonstrates how to implement learning rate scheduling, where the learning rate is adjusted during training. In this example, the learning rate is reduced by a factor of 0.1 after 10 epochs.

python
Copy
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.1

lr_scheduler = LearningRateScheduler(scheduler)



AI Agent with Retrieval-Augmented Generation (RAG), Arithmetic Function Calling, Summarization, and Web Search
Overview
This project implements an AI agent capable of processing documents, performing basic arithmetic operations, generating summaries and notes, and integrating Retrieval-Augmented Generation (RAG). The agent also supports web search functionality, with a placeholder for real web search integration.

Features
File Ingestion with LlamaIndex:

The agent can ingest files from a specified directory using the LlamaIndex library. It then processes and stores the documents in a vector index for easy retrieval.
Retrieval-Augmented Generation (RAG):

The agent enhances its responses by retrieving relevant information from the ingested documents based on a query. This is achieved using LlamaIndex’s query engine.
Arithmetic Function Calling:

The agent supports basic arithmetic operations such as addition, subtraction, multiplication, and division, which can be called using LangChain's function calling mechanism.
Summarization and Notes Generation:

The agent can generate summaries and notes based on provided content using OpenAI models. This feature is useful for processing large documents or extracting key points.
Web Search:

The agent includes a placeholder for web search functionality, which can be used to augment responses with real-time data from the web (via APIs like SerpAPI). Currently, it's implemented as a mock function and can be replaced with a real API.
How the Code Works
1. File Ingestion with LlamaIndex
The agent first loads documents from a specified directory using LlamaIndex’s SimpleDirectoryReader. This reader converts the documents into Document objects, which are then indexed into a vector store using VectorStoreIndex. The vector store allows the agent to efficiently query and retrieve relevant information later.

python
Copy
def ingest_files(directory: str):
    documents = SimpleDirectoryReader(directory).load_data()
    service_context = ServiceContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index
2. Retrieval-Augmented Generation (RAG)
When the agent receives a query, it uses the query_engine from the LlamaIndex to retrieve documents that are most relevant to the query. The results are then used to generate an enhanced response by combining the information from the query and the retrieved documents.

python
Copy
def retrieve_and_generate(query: str, index):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response
3. Arithmetic Function Calling
The agent supports basic arithmetic operations using the @tool decorator from LangChain. Functions such as add, subtract, multiply, and divide are defined and can be called dynamically by the agent.

python
Copy
@tool
def add(a: float, b: float) -> float:
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    return a - b
These operations can be called within the agent's context, allowing users to perform arithmetic as part of the agent's broader tasks.

4. Summarization and Notes Generation
The agent can summarize and generate notes from provided content. It uses the OpenAI model to process the text and create concise summaries or extract key notes. This is useful for handling large documents or simply summarizing important points from text.

python
Copy
def generate_summary(content: str) -> str:
    prompt = "Summarize the following text:\n" + content
    llm = OpenAI(temperature=0.5)
    chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt, input_variables=[]))
    summary = chain.run("")
    return summary
5. Web Search Integration (Placeholder)
The agent includes a placeholder function, search_web, which simulates web search functionality. It can be connected to a real search API like SerpAPI or Google Custom Search API to fetch live data and augment the agent’s responses.

python
Copy
def search_web(query: str) -> str:
    # Placeholder response for web search
    api_key = "your_serpapi_key"  # Replace with your API key
    params = {
        "q": query,
        "api_key": api_key
    }
    response = requests.get("https://serpapi.com/search", params=params)
    search_results = response.json()
    
    # Process the search results and return relevant data
    return search_results.get("organic_results", [{}])[0].get("snippet", "No relevant information found.")
6. Main Processing Logic
The core of the agent is the process_input function, which integrates all the other functionalities. It takes a query and a directory as input, ingests the documents from the directory, performs RAG, generates a summary, and returns both the RAG-enhanced response and the summary.

python
Copy
def process_input(query: str, directory: str):
    # Ingest files
    index = ingest_files(directory)

    # RAG response
    rag_response = retrieve_and_generate(query, index)

    # Generate a summary of the query (just an example)
    summary = generate_summary(rag_response)

    # Return both RAG response and summary
    return {
        "rag_response": rag_response,
        "summary": summary
    }
Usage Instructions
1. Set Up the Environment
Install the required dependencies:

bash
Copy
pip install -r requirements.txt
Set your OpenAI API key as an environment variable:

bash
Copy
export OPENAI_API_KEY="your_openai_api_key"
If you want to use the web search functionality, obtain an API key from SerpAPI, and replace "your_serpapi_key" in the search_web function with your actual key.

2. Preparing Your Files
Place the documents you want the agent to process in the data/ directory. These documents can be in any format that SimpleDirectoryReader can process (e.g., text, PDFs, etc.).

3. Running the Agent
To query the agent, simply call the process_input function with a query and the directory where your documents are stored:

python
Copy
final_response = process_input("What is Artificial Intelligence?", "data/")
print(f"Final Response: {final_response}")
This will:

Ingest files from the data/ directory.
Use RAG to retrieve relevant information.
Generate a summary based on the retrieved data.
Example Output
Here is what the output might look like when querying for information on Artificial Intelligence:

json
Copy
{
  "rag_response": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.",
  "summary": "AI is the simulation of human intelligence in machines, enabling them to think and learn."
}
Contributing
Feel free to fork this repository and contribute! If you have improvements or features you'd like to add, create a pull request. Some ideas for future improvements:

Replace the web search placeholder with an actual search API.
Add more complex reasoning capabilities using additional tools or models.
