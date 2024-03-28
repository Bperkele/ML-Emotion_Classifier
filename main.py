from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
import seaborn as sns

labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
path = "D:\Documents\ML-project\ML-Emotion_Classifier\emotion.csv"


#if its in pickle format use this
#df = pd.read_pickle("merged_training.pkl")
#convert the emotion to label
#df['label'] = df['emotion'].apply(lambda x: labels.index(x))
#df = df.drop(columns=['emotion'])
#turn labels into integers
#df['label'] = df['label'].astype(int)
#df.to_csv("emotion.csv", index=False)

#print(df.head())
# Convert the dataset to a pandas dataframe ignore first column
df = pd.read_csv(path)

# Assuming df['column_name'] is your column of sentences
df['length'] = df['text'].apply(lambda x: len(x.split()))

def trim_and_downsample_sentences(df, text_col, target_col, min_length=0, max_length=20):
    # 1. Filter through the text lengths
    df['length'] = df[text_col].apply(lambda x: len(x.split()))
    
    # 2. Find the ones that are over 50 words and trim them to 50 exactly
    df.loc[df['length'] > max_length, text_col] = df.loc[df['length'] > max_length, text_col].apply(lambda x: ' '.join(x.split()[:max_length]))
    
    # 3. If the sentence is under 50, check if it's between the range
    df = df[(df['length'] >= min_length) & (df['length'] <= max_length)]
    
    # 4. Downsample the data
    min_size = df[target_col].value_counts().min()
    lst = []
    for class_index, group in df.groupby(target_col):
        lst.append(resample(group, replace=False, n_samples=min_size, random_state=123))
    df_downsampled = pd.concat(lst)
    
    return df_downsampled

df = trim_and_downsample_sentences(df, 'text', 'label')

# Plot a histogram
#plt.hist(df['length'], bins=50)
#plt.show()

#Remove the length column
df = df.drop(columns=['length'])

#plot the distribution of the labels
#plt.hist(df['label'], bins=50)
#plt.show()

print(df.describe())
# Split the dataset into training and testing
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

X_train, Y_train = train['text'], train['label']
X_test, Y_test = test['text'], test['label']

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Convert the labels to one hot encoding
Y_oh_train = convert_to_one_hot(Y_train, C = 6)
Y_oh_test = convert_to_one_hot(Y_test, C = 6)

#helper functions

def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

#read the glove file
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs("D:\Documents\glove.twitter.27B\glove.twitter.27B.25d.txt")

def pad_vector(vec, length):
    # If the vector is shorter than the desired length, pad it with zeros
    if len(vec) < length:
        vec = np.pad(vec, (0, length - len(vec)), 'constant')
    return vec

# Apply the function to your word_to_vec_map
for word, vec in word_to_vec_map.items():
    word_to_vec_map[word] = pad_vector(vec, 25)

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]  # number of training examples
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
                j = j + 1
    return X_indices


class NN(nn.Module):
  def __init__(self, embedding, embedding_dim, hidden_dim, vocab_size, output_dim, batch_size):
      super(NN, self).__init__()

      self.batch_size = batch_size

      self.hidden_dim = hidden_dim

      self.word_embeddings = embedding

      # The LSTM takes word embeddings as inputs, and outputs hidden states
      # with dimensionality hidden_dim.
      self.lstm = nn.LSTM(embedding_dim, 
                          hidden_dim, 
                          num_layers=2,
                          dropout = 0.5,
                          batch_first = True)

      # The linear layer that maps from hidden state space to output space
      self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, sentence):
      
      #sentence = sentence.type(torch.LongTensor)
      #print ('Shape of sentence is:', sentence.shape)

      sentence = sentence.to(device)

      embeds = self.word_embeddings(sentence)
      #print ('Embedding layer output shape', embeds.shape)

      # initializing the hidden state to 0
      #hidden=None
      
      h0 = torch.zeros(2, sentence.size(0), hidden_dim).requires_grad_().to(device)
      c0 = torch.zeros(2, sentence.size(0), hidden_dim).requires_grad_().to(device)
      
      lstm_out, h = self.lstm(embeds, (h0, c0))
      # get info from last timestep only
      lstm_out = lstm_out[:, -1, :]
      #print ('LSTM layer output shape', lstm_out.shape)
      #print ('LSTM layer output ', lstm_out)

      # Dropout
      lstm_out = F.dropout(lstm_out, 0.5)

      fc_out = self.fc(lstm_out)
      #print ('FC layer output shape', fc_out.shape)
      #print ('FC layer output ', fc_out)
      
      out = fc_out
      out = F.softmax(out, dim=1)
      #print ('Output layer output shape', out.shape)
      #print ('Output layer output ', out)
      return out
  
def pretrained_embedding_layer(word_to_vec_map, word_to_index, non_trainable=True):
    num_embeddings = len(word_to_index) +1                  
    embedding_dim = word_to_vec_map["cucumber"].shape[0]  #  dimensionality of GloVe word vectors (= 50)

    # Initialize the embedding matrix as a numpy array of zeros of shape (num_embeddings, embedding_dim)
    weights_matrix = np.zeros((num_embeddings, embedding_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        weights_matrix[index, :] = word_to_vec_map[word]

    embed = nn.Embedding.from_pretrained(torch.from_numpy(weights_matrix).type(torch.FloatTensor), freeze=non_trainable)

    return embed, num_embeddings, embedding_dim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def train(model, trainloader, criterion, optimizer, epochs=10):
    
    model.to(device)
    running_loss = 0
    
    train_losses, test_losses, accuracies, f1_scores, precisions, recalls  = [], [], [], [], [], []
    epoch_precisions, epoch_recalls, epoch_f1_scores = [], [], []
    y_true, y_pred = [], []
    for e in range(epochs):

        running_loss = 0
        
        model.train()
        
        for sentences, labels in trainloader:

            sentences, labels = sentences.to(device), labels.to(device)

            # 1) erase previous gradients (if they exist)
            optimizer.zero_grad()

            # 2) make a prediction
            pred = model.forward(sentences)

            # 3) calculate how much we missed
            loss = criterion(pred, labels)

            # 4) figure out which weights caused us to miss
            loss.backward()

            # 5) change those weights
            optimizer.step()

            # 6) log our progress
            running_loss += loss.item()
        
        
        else:

          model.eval()

          test_loss = 0
          accuracy = 0
          
          # Turn off gradients for validation, saves memory and computations
          with torch.no_grad():
              for sentences, labels in test_loader:
                  sentences, labels = sentences.to(device), labels.to(device)
                  log_ps = model(sentences)
                  test_loss += criterion(log_ps, labels)
                  
                  ps = torch.exp(log_ps)
                  top_p, top_class = ps.topk(1, dim=1)
                  equals = top_class == labels.view(*top_class.shape)
                  accuracy += torch.mean(equals.type(torch.FloatTensor))

                  # Calculate precision, recall, and F1 score
                  precision = precision_score(labels.cpu().numpy(), top_class.cpu().numpy(), average='macro', zero_division=1)
                  recall = recall_score(labels.cpu().numpy(), top_class.cpu().numpy(), average='macro', zero_division=1)
                  f1 = f1_score(labels.cpu().numpy(), top_class.cpu().numpy(), average='macro', zero_division=1)

                  epoch_precisions.append(precision)
                  epoch_recalls.append(recall)
                  epoch_f1_scores.append(f1)

                  y_true.extend(labels.cpu().numpy())
                  y_pred.extend(top_class.cpu().numpy())
                  
          train_losses.append(running_loss/len(train_loader))
          test_losses.append(test_loss/len(test_loader))
          accuracies.append(accuracy / len(test_loader) * 100)

          # calculate average precision, recall, and F1 score for the epoch
          precisions.append(np.mean(epoch_precisions))
          recalls.append(np.mean(epoch_recalls))
          f1_scores.append(np.mean(epoch_f1_scores))
          cm = confusion_matrix(y_true, y_pred)

          print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
                "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
          
    # Plot
    plt.figure(figsize=(20, 5))
    plt.plot(train_losses, c='b', label='Training loss')
    plt.plot(test_losses, c='r', label='Testing loss')
    plt.xticks(np.arange(0, epochs))
    plt.title('Losses')
    plt.legend(loc='upper right')
    plt.savefig('losses.png')  # Save figure

    plt.figure(figsize=(20, 5))
    plt.plot(accuracies)
    plt.xticks(np.arange(0, epochs))
    plt.title('Accuracy')
    plt.savefig('accuracy.png')  # Save figure

    plt.figure(figsize=(20, 5))
    plt.plot(precisions, c='g', label='Precision')
    plt.plot(recalls, c='y', label='Recall')
    plt.plot(f1_scores, c='b', label='F1 Score')
    plt.xticks(np.arange(0, epochs))
    plt.title('Precision, Recall, and F1 Score')
    plt.legend(loc='upper right')
    plt.savefig('precision_recall_f1.png')  # Save figure

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save figure

         

import torch.utils.data


maxLen = len(max(X_train, key=len).split())
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 6)

X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 6)

embedding, vocab_size, embedding_dim = pretrained_embedding_layer(word_to_vec_map, word_to_index, non_trainable=True)

hidden_dim=128
output_size=6
batch_size = 16

#print ('Embedding layer is ', embedding)
#print ('Embedding layer weights ', embedding.weight.shape)

model = NN(embedding, embedding_dim, hidden_dim, vocab_size, output_size, batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_indices).type(torch.LongTensor), torch.tensor(Y_train).type(torch.LongTensor))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_indices).type(torch.LongTensor), torch.tensor(Y_test).type(torch.LongTensor))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

train(model, train_loader, criterion, optimizer, epochs)

test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for sentences, labels in test_loader:
        sentences, labels = sentences.to(device), labels.to(device)
        ps = model(sentences)
        test_loss += criterion(ps, labels).item()

        # Accuracy
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
model.train()
print("Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
      "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
running_loss = 0

def predict(input_text, print_sentence=True):
  labels_dict = {
		0 : 'sadness',
        1 : 'joy',
        2 : 'love',
        3 : 'anger',
        4 : 'fear',
        5 : 'surprise'
	}

  # Convert the input to the model
  x_test = np.array([input_text])
  X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
  sentences = torch.tensor(X_test_indices).type(torch.LongTensor)

  # Get the class label
  ps = model(sentences)
  top_p, top_class = ps.topk(1, dim=1)
  label = int(top_class[0][0])

  if print_sentence:
    print("\nInput Text: \t"+ input_text +'\nEmotion: \t'+  labels_dict[label])

  return label


#model.load_state_dict(torch.load("model.pth"))
#model.eval()

print('-----------------TWEETS----------------')
predict("Just stumbled upon the most amazing little café tucked away in a corner of the city! Who knew such hidden gems existed?")

predict("Woke up to the sun streaming through my window.It's the little things that bring so much joy!")

predict("Sometimes, no matter how hard you try, things just don't work out the way you hoped. Feeling disappointed..")

predict("Watching the sunset with the one I love by my side, feeling like the luckiest person in the world. #love #soulmate")

predict("Just experienced the worst customer service ever. It's infuriating when companies don't value their customers!")

predict("Heart pounding, palms sweaty, stepping out of my comfort zone.Terrified of failing,even more terrified of never trying.")
print("\n------------------------------------")
# Save the model as a new model if there is already a model with that path

torch.save(model.state_dict(), "model.pth") if not os.path.isfile("model.pth") else torch.save(model.state_dict(), "new_model.pth")



