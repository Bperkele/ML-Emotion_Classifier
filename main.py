from datasets import load_dataset
import pandas as pd

dataset = load_dataset("dair-ai/emotion",trust_remote_code=True)


# Define the labels
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Extract the train, validation, and test datasets
train_dataset = pd.DataFrame(dataset['train'])
validation_dataset = pd.DataFrame(dataset['validation'])
test_dataset = pd.DataFrame(dataset['test'])

# Create binary encoded features for each label in the train, validation, and test datasets
for label in labels:
    train_dataset[label] = (train_dataset['label'] == labels.index(label)).astype(int)
    validation_dataset[label] = (validation_dataset['label'] == labels.index(label)).astype(int)
    test_dataset[label] = (test_dataset['label'] == labels.index(label)).astype(int)

# Remove the label column from the train, validation, and test datasets
train_dataset = train_dataset.drop(columns=['label'])
validation_dataset = validation_dataset.drop(columns=['label'])
test_dataset = test_dataset.drop(columns=['label'])

# Print the features of the train, validation, and test datasets
print(train_dataset.head())
print(validation_dataset.head())
print(test_dataset.head())



