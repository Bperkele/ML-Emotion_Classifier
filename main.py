from datasets import load_dataset
import pandas as pd
from sklearn.utils import resample

dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)


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


# Get the minimum number of entries for a single label
min_size = train_dataset[labels].sum().min()


# Create a new dataframe to hold the balanced dataset
balanced_train_dataset = pd.DataFrame()

# Resample each label in the train dataset
for label in labels:
    # Separate the current label
    label_df = train_dataset[train_dataset[label] == 1]
    
    # Resample the current label to match the max size
    label_df_resampled = resample(label_df, replace=True, n_samples=int(min_size), random_state=123)
    
    # Append the resampled label to the new dataframe
    balanced_train_dataset = pd.concat([balanced_train_dataset, label_df_resampled])

#plot the balanced dataset
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(labels, balanced_train_dataset[labels].sum())
plt.show()



