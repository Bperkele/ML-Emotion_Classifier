from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

def balance_dataset(dataset, labels):
    min_size = dataset[labels].sum().min()
    balanced_dataset = pd.DataFrame()

    for label in labels:
        label_df = dataset[dataset[label] == 1]
        label_df_resampled = resample(label_df, replace=True, n_samples=int(min_size), random_state=123)
        balanced_dataset = pd.concat([balanced_dataset, label_df_resampled])

    return balanced_dataset

def plot_dataset(dataset, labels, title):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, dataset[labels].sum())
    plt.title(title)
    plt.show()

dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Extract the train, validation, and test datasets
train_dataset = pd.DataFrame(dataset['train'])
validation_dataset = pd.DataFrame(dataset['validation'])
test_dataset = pd.DataFrame(dataset['test'])

# Create binary encoded features for each label in the train, validation, and test datasets
lb = LabelBinarizer()
lb.fit(train_dataset['label'])

for dataset in [train_dataset, validation_dataset, test_dataset]:
    binary_labels = lb.transform(dataset['label'])
    dataset[labels] = pd.DataFrame(binary_labels, columns=labels)
    dataset.drop(columns=['label'], inplace=True)

# Balance and plot each dataset
for dataset, title in zip([train_dataset, validation_dataset, test_dataset], ['Train', 'Validation', 'Test']):
    balanced_dataset = balance_dataset(dataset, labels)
    #print the description of the balanced dataset
    print(balanced_dataset.describe())

    #plot_dataset(balanced_dataset, labels, f'Balanced {title} Dataset')
