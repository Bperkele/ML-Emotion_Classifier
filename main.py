from datasets import load_dataset
import tensorflow as tf
import numpy as np
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

# Convert labels to numpy arrays
train_labels = np.array([example['label'] for example in dataset['train']])
test_labels = np.array([example['label'] for example in dataset['test']])

# Input
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).shuffle(len(train_encodings)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels)).batch(32)

# The model
num_labels = len(labels)
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Model training
model.compile(
    optimizer=optimizers.Adam(learning_rate=5e-5),
    loss=losses.BinaryCrossentropy(from_logits=False),  # Use binary crossentropy for multi-label classification
    metrics=metrics.BinaryAccuracy()
)

model.fit(train_dataset, epochs=1)

# Model evaluation
loss, accuracy = model.evaluate(test_dataset)
print("Test loss:", loss)
print("Test accuracy:", accuracy)
