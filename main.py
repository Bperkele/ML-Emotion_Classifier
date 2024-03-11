from datasets import load_dataset
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import optimizers, losses, metrics

# Load emotion dataset
dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)

# Define the labels
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

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

# Tokenizing the data
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_dataset['text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_dataset['text'].tolist(), truncation=True, padding=True)

# Input
train_datasets = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_dataset[labels].to_dict('list')))
train_datasets = train_datasets.shuffle(len(train_dataset)).batch(32)
test_datasets = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_dataset[labels].to_dict('list')))
test_datasets = test_datasets.batch(32)

# The model
num_labels = len(labels)
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Model training
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
model.compile(
    optimizer=optimizers.Adam(learning_rate=5e-5),
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=metrics.BinaryAccuracy()
)

model.fit(train_datasets, epochs=5)

loss, accuracy = model.evaluate(test_dataset)
print("Test loss:", loss)
print("Test accuracy:", accuracy)