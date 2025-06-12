# Standard Library Imports
import os
import copy
import random
import warnings
from collections import defaultdict
from typing import Optional

# ML
import numpy as np
import scipy as sp
import pandas as pd

# Visual
import matplotlib
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
from IPython.display import display

# DL
import torch as th
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score , roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support

SEED = 42
# Set the random seed for Python
random.seed(SEED)

# Set the random seed for numpy
np.random.seed(SEED)

# Set the random seed for pytorch
th.manual_seed(SEED)

# If using CUDA (for GPU operations)
th.cuda.manual_seed(SEED)

# Set up the device
# TO DO ----------------------------------------------------------------------
DEVICE = "mps"
# TO DO ----------------------------------------------------------------------
#assert DEVICE == "cuda"

DataType = list[tuple[list[str],list[str]]]


def read_data(filepath:str) -> DataType:
    """
    Read data from a single file.
    The function recieves a filepath
    The funtion encodes every sentence using a pair of lists, one list contains the words and one list contains the tags.
    :param filepath: path to the file
    :return: data as a list of tuples
    """
    data = []
    words = []
    tags = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                if words:
                    data.append((words, tags))
                    words = []
                    tags = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    tag = parts[-1]
                    words.append(word)
                    tags.append(tag)

        # Add the final sentence if the file doesn't end with a newline
        if words:
            data.append((words, tags))

    return data


# Initinize ids for special tokens
PAD_TOKEN = -1
UNK_TOKEN = -2

class Vocab:
    def __init__(self, train: DataType):
        """
        Initialize a Vocab instance.
        :param train: train data
        """
        self.word2id = {"__unk__": UNK_TOKEN, "__pad__": PAD_TOKEN}
        self.id2word = {UNK_TOKEN: "__unk__", PAD_TOKEN: "__pad__"}
        self.n_words = 2

        self.tag2id = {}
        self.id2tag = {}
        self.n_tags = 0

        for words, tags in train:
            for word in words:
                if word not in self.word2id:
                    self.word2id[word] = self.n_words
                    self.id2word[self.n_words] = word
                    self.n_words += 1
            for tag in tags:
                if tag not in self.tag2id:
                    self.tag2id[tag] = self.n_tags
                    self.id2tag[self.n_tags] = tag
                    self.n_tags += 1    

    def __len__(self):
        return self.n_words

    def index_tags(self, tags: list[str]) -> list[int]:
        """
        Convert tags to Ids.
        :param tags: list of tags
        :return: list of Ids
        """
        tag_indexes = [self.tag2id[t] for t in tags]
        return tag_indexes

    def index_words(self, words: list[str]) -> list[int]:
        """
        Convert words to Ids.
        :param words: list of words
        :return: list of Ids
        """
        word_indexes = [self.word2id[w] if w in self.word2id else self.word2id["__unk__"] for w in words]
        return word_indexes

def prepare_data(data: DataType, vocab: Vocab):
    data_sequences = []
    PAD_WORD_ID = vocab.word2id["__pad__"]
    PAD_TAG_ID = vocab.tag2id.get("O", 0)  # Default to "O" if no padding tag defined
    
    max_len = max(len(words) for words, _ in data)

    for words, tags in data:
        word_ids = vocab.index_words(words)
        tag_ids = vocab.index_tags(tags)

        # Padding
        padding_len = max_len - len(words)
        word_ids += [PAD_WORD_ID] * padding_len
        tag_ids += [PAD_TAG_ID] * padding_len

        data_sequences.append((word_ids, tag_ids))

    return data_sequences    

def count_oov(sequences) -> int:
    """
    Count the number of OOV words.
    :param sequences: list of sequences
    :return: number of OOV words
    """
    oov = 0

    for word_ids, _ in sequences:
        oov += sum(1 for wid in word_ids if wid == UNK_TOKEN)

    return oov

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        """
        Dataset wrapping a list of (word_ids, tag_ids) pairs.
        :param sequences: list of (word_ids, tag_ids)
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        word_ids, tag_ids = self.sequences[idx]
        word_ids = th.tensor(word_ids, dtype=th.long)
        tag_ids = th.tensor(tag_ids, dtype=th.long)
        return word_ids, tag_ids

def prepare_data_loader(sequences, batch_size: int, train: bool = True):
    """
    Create a DataLoader from a list of sequences.
    :param sequences: list of (word_ids, tag_ids)
    :param batch_size: batch size
    :param train: whether to shuffle the dataloader
    :return: PyTorch DataLoader
    """
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader


class NERNet(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, output_size: int, n_layers: int, directions: int):
        """
        Initialize a NERNet instance.
        :param input_size: the size of the vocabulary
        :param embedding_size: the size of the embeddings
        :param hidden_size: the LSTM hidden size
        :param output_size: the number tags we are predicting for
        :param directions: could be 1 or 2, indicating unidirectional or bidirectional LSTM, respectively
        :param n_layers: the number of layers we want to use in LSTM
        """
        super(NERNet, self).__init__()
        # TO DO ----------------------------------------------------------------------
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers, bidirectional=directions == 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.zero_()
  

        # TO DO ----------------------------------------------------------------------

    def forward(self, input_sentence):
        # TO DO ----------------------------------------------------------------------
        # input_sentence: (batch_size, seq_len)
        
        embed = self.embedding(input_sentence)
        embed = self.dropout(embed)
        lstm_out, _ = self.lstm(embed)
        output = self.fc(lstm_out)
        output = output.view(output.shape[0], -1, output.shape[2])

        # TO DO ----------------------------------------------------------------------
        return output




def train_loop(model: NERNet, n_epochs: int, dataloader_train, dataloader_dev):
    """
    Train a model.
    :param model: model instance
    :param n_epochs: number of epochs to train on
    :param dataloader_train: train dataloader
    :param dataloader_dev: dev dataloader
    :return: loss and accuracy during training
    """
    # Optimizer (ADAM is a fancy version of SGD)
    optimizer = Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1 )
    # Record
    metrics = {'loss': {'train': [], 'dev': []}, 'accuracy': {'train': [], 'dev': []}}

    # Move model to device
    model.to(DEVICE)

    ## TO DO ----------------------------------------------------------------------
    tqdm_epochs = tqdm(range(n_epochs), desc="Epochs",colour="green")
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    for epoch in tqdm_epochs:
        for batch in dataloader_train:
            inputs, labels = batch

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()	
            outputs = model(inputs)
            outputs_flat = outputs.view(-1, outputs.shape[2])
            labels_flat = labels.view(-1) 
            loss = loss_fn(outputs_flat, labels_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=2)  # shape (B, T)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # TO DO ----------------------------------------------------------------------
    metrics['loss']['train'].append(total_loss / len(dataloader_train))
    metrics['accuracy']['train'].append(total_correct / total_samples)
    

    return metrics

def evaluate(model: NERNet, title: str, dataloader: DataLoader, vocab: Vocab):
    """
    Write an evaluation loop for a trained model using the dev and test datasets. This function will print the `Recall`, `Precision`, and `F1` scores and plot a `Confusion Matrix`.
    Perform this evaluation twice:
    1. For all labels (7 labels in total).
    2. For all labels except "O" (6 labels in total).
    :param model: the trained model
    :param title: the title of the model
    :param dataloader: the dataloader
    :param vocab: the vocabulary
    :return: the results
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with th.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            preds = outputs.argmax(dim=2)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics for all labels
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Calculate metrics excluding 'O' label (assuming 'O' is label 0)
    mask = all_labels != 0
    precision_wo_o, recall_wo_o, f1_wo_o, _ = precision_recall_fscore_support(all_labels[mask], all_preds[mask], average='weighted')
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Print results
    print(f"\nResults for {title}:")
    print(f"All labels:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nExcluding 'O' label:")
    print(f"Precision: {precision_wo_o:.4f}")
    print(f"Recall: {recall_wo_o:.4f}")
    print(f"F1 Score: {f1_wo_o:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_wo_o': precision_wo_o,
        'recall_wo_o': recall_wo_o,
        'f1_wo_o': f1_wo_o
    }

def evaluate_model(vocab: Vocab):
    EMB_DIM = 300
    N_EPOCHS = 5
    results = pd.DataFrame(columns=columns)
    file_name = "NER_results.csv"
    # TO DO ----------------------------------------------------------------------
    models = {
        'Model1': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 1, 'directions': 1},
        # 'Model2': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 2, 'directions': 1},
        # 'Model3': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 3, 'directions': 1},
        # 'Model4': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 1, 'directions': 2},
        # 'Model5': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 2, 'directions': 2},
        # 'Model6': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 3, 'directions': 2},
        # 'Model7': {'embedding_size': EMB_DIM, 'hidden_size': 800, 'n_layers': 1, 'directions': 2},
        # 'Model8': {'embedding_size': EMB_DIM, 'hidden_size': 800, 'n_layers': 2, 'directions': 2},
        # 'Model9': {'embedding_size': EMB_DIM, 'hidden_size': 800, 'n_layers': 3, 'directions': 2},
    }

    # TO DO ----------------------------------------------------------------------
    results_test = []

    for model_name, model_params in models.items():
        print(f"\n🚀 Training {model_name}")
        model = NERNet(**model_params, input_size=vocab.n_words, output_size=vocab.n_tags)
        model.to(DEVICE)
        
        train_loop(model, n_epochs=N_EPOCHS, dataloader_train=dl_train,dataloader_dev=dl_dev)
        eval_result = evaluate(model, model_name, dl_test, vocab)

        # Store flat summary
        summary = {
            'N_MODEL': model_name,
            'HIDDEN_SIZE': model_params['hidden_size'],
            'N_LAYERS': model_params['n_layers'],
            'DIRECTIONS': model_params['directions'],
            'PRECISION': eval_result['precision'],
            'RECALL': eval_result['recall'],
            'F1': eval_result['f1'],
            'RECALL_WO_O': eval_result['recall_wo_o'],
            'PRECISION_WO_O': eval_result['precision_wo_o'],
            'F1_WO_O': eval_result['f1_wo_o'],
        }
        results_test.append(summary)
    df = pd.DataFrame(results_test, columns=columns)
    df.to_csv(file_name, index=False)
    print(tabulate(df, headers='keys', tablefmt='psql',floatfmt=".4f"))

train = read_data("data/train.txt")
dev = read_data("data/dev.txt")
test = read_data("data/test.txt")
vocab = Vocab(train)
train_sequences = prepare_data(train, vocab)
dev_sequences = prepare_data(dev, vocab)
test_sequences = prepare_data(test, vocab)

data_loader_train = prepare_data_loader(train_sequences, batch_size=16)


BATCH_SIZE = 16
dl_train = prepare_data_loader(train_sequences, batch_size=BATCH_SIZE)
dl_dev = prepare_data_loader(dev_sequences, batch_size=BATCH_SIZE, train=False)
dl_test = prepare_data_loader(test_sequences, batch_size=BATCH_SIZE, train=False)


columns = ['N_MODEL','HIDDEN_SIZE','N_LAYERS','DIRECTIONS','RECALL','PERCISION','F1','RECALL_WO_O','PERCISION_WO_O','F1_WO_O']

evaluate_model(vocab)
# model = NERNet(vocab.n_words, embedding_size=300, hidden_size=800, output_size=vocab.n_tags, n_layers=2, directions=1)
# model.to(DEVICE)