import pandas as pd
from datasets import load_dataset
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from tensorflow.keras import optimizers, losses, metrics

# Load emotion dataset
dataset = load_dataset("dair-ai/emotion")

# Define the labels
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Tokenizing the data
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(dataset['train']['text'], truncation=True, padding=True)
test_encodings = tokenizer(dataset['test']['text'], truncation=True, padding=True)

# Extract the train, validation, and test datasets
train_dataset = pd.DataFrame(dataset['train'])
validation_dataset = pd.DataFrame(dataset['validation'])
test_dataset = pd.DataFrame(dataset['test'])

# Create binary encoded features for each label in the train, validation, and test datasets
for label in labels:
    train_dataset[label] = (train_dataset['label'] == label).astype(int)
    validation_dataset[label] = (validation_dataset['label'] == label).astype(int)
    test_dataset[label] = (test_dataset['label'] == label).astype(int)

# Remove unnecessary columns
train_dataset = train_dataset.drop(columns=['label'])
validation_dataset = validation_dataset.drop(columns=['label'])
test_dataset = test_dataset.drop(columns=['label'])

# Input
train_dataset_tf = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_dataset[labels].values)).shuffle(len(train_encodings)).batch(32)
test_dataset_tf = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_dataset[labels].values)).batch(32)

# The model
num_labels = len(labels)
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Model training
model.compile(
    optimizer=optimizers.Adam(learning_rate=2e-5),
    loss=losses.BinaryCrossentropy(from_logits=False),  # Use binary crossentropy for multi-label classification
    metrics=metrics.BinaryAccuracy()
)

model.fit(train_dataset_tf, epochs=1)

# Model evaluation
loss, accuracy = model.evaluate(test_dataset_tf)
print("Test loss:", loss)
print("Test accuracy:", accuracy)
