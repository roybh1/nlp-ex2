#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing
# ![](https://i.imgur.com/qkg2E2D.png)
# 
# ## Assignment 002 - NER Tagger
# 
# > Notebook by:
# > - NLP Course Stuff
# ## Revision History
# 
# | Version | Date       | User        | Content / Changes                                                   |
# |---------|------------|-------------|---------------------------------------------------------------------|
# | 0.1.000 | 29/05/2025 | course staff| First version                                                       |
# | 0.2.000 | 09/06/2025 | course staff| Second version                                                       |

# ## Overview
# In this assignment, you will build a complete training and testing pipeline for a neural sequential tagger for named entities using LSTM.
# 
# ## Dataset
# You will work with the ReCoNLL 2003 dataset, a corrected version of the [CoNLL 2003 dataset](https://www.clips.uantwerpen.be/conll2003/ner/):
# 
# **Click on those links so you have access to the data!**
# - [Train data](https://drive.google.com/file/d/1CqEGoLPVKau3gvVrdG6ORyfOEr1FSZGf/view?usp=sharing)
# 
# - [Dev data](https://drive.google.com/file/d/1rdUida-j3OXcwftITBlgOh8nURhAYUDw/view?usp=sharing)
# 
# - [Test data](https://drive.google.com/file/d/137Ht40OfflcsE6BIYshHbT5b2iIJVaDx/view?usp=sharing)
# 
# As you will see, the annotated texts are labeled according to the `IOB` annotation scheme (more on this below), for 3 entity types: Person, Organization, Location.
# 
# ## Your Implementation
# 
# Please create a local copy of this template Colab's Notebook:
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KGkObwUn5QQm_v0nB0nAUlB4YrwThuzl#scrollTo=Z-fCqGh9ybgm)
# 
# The assignment's instructions are there; follow the notebook.
# 
# ## Submission
# - **Notebook Link**: Add the URL to your assignment's notebook in the `notebook_link.txt` file, following the format provided in the example.
# - **Access**: Ensure the link has edit permissions enabled to allow modifications if needed.
# - **Deadline**: <font color='green'>16/06/2025</font>.
# - **Platform**: Continue using GitHub for submissions. Push your project to the team repository and monitor the test results under the actions section.
# 
# Good Luck 🤗
# 

# <!-- ## NER schemes:  
# 
# > `IO`: is the simplest scheme that can be applied to this task. In this scheme, each token from the dataset is assigned one of two tags: an inside tag (`I`) and an outside tag (`O`). The `I` tag is for named entities, whereas the `O` tag is for normal words. This scheme has a limitation, as it cannot correctly encode consecutive entities of the same type.
# 
# > `IOB`: This scheme is also referred to in the literature as BIO and has been adopted by the Conference on Computational Natural Language Learning (CoNLL) [1]. It assigns a tag to each word in the text, determining whether it is the beginning (`B`) of a known named entity, inside (`I`) it, or outside (`O`) of any known named entities.
# 
# > `IOE`: This scheme works nearly identically to `IOB`, but it indicates the end of the entity (`E` tag) instead of its beginning.
# 
# > `IOBES`: An alternative to the IOB scheme is `IOBES`, which increases the amount of information related to the boundaries of named entities. In addition to tagging words at the beginning (`B`), inside (`I`), end (`E`), and outside (`O`) of a named entity. It also labels single-token entities with the tag `S`.
# 
# > `BI`: This scheme tags entities in a similar method to `IOB`. Additionally, it labels the beginning of non-entity words with the tag B-O and the rest as I-O.
# 
# > `IE`: This scheme works exactly like `IOE` with the distinction that it labels the end of non-entity words with the tag `E-O` and the rest as `I-O`.
# 
# > `BIES`: This scheme encodes the entities similar to `IOBES`. In addition, it also encodes the non-entity words using the same method. It uses `B-O` to tag the beginning of non-entity words, `I-O` to tag the inside of non-entity words, and `S-O` for single non-entity tokens that exist between two entities. -->
# 
# 
# ## NER Schemes
# 
# ### IO
# - **Description**: The simplest scheme for named entity recognition (NER).
# - **Tags**:
# 	- `I`: Inside a named entity.
# 	- `O`: Outside any named entity.
# - **Limitation**: Cannot correctly encode consecutive entities of the same type.
# 
# ### IOB (BIO)
# - **Description**: Adopted by the Conference on Computational Natural Language Learning (CoNLL).
# - **Tags**:
# 	- `B`: Beginning of a named entity.
# 	- `I`: Inside a named entity.
# 	- `O`: Outside any named entity.
# - **Advantage**: Can encode the boundaries of consecutive entities.
# 
# ### IOE
# - **Description**: Similar to IOB, but indicates the end of an entity.
# - **Tags**:
# 	- `I`: Inside a named entity.
# 	- `O`: Outside any named entity.
# 	- `E`: End of a named entity.
# - **Advantage**: Focuses on the end boundary of entities.
# 
# ### IOBES
# - **Description**: An extension of IOB with additional boundary information.
# - **Tags**:
# 	- `B`: Beginning of a named entity.
# 	- `I`: Inside a named entity.
# 	- `O`: Outside any named entity.
# 	- `E`: End of a named entity.
# 	- `S`: Single-token named entity.
# - **Advantage**: Provides more detailed boundary information for named entities.
# 
# ### BI
# - **Description**: Tags entities similarly to IOB and labels the beginning of non-entity words.
# - **Tags**:
# 	- `B`: Beginning of a named entity.
# 	- `I`: Inside a named entity.
# 	- `B-O`: Beginning of a non-entity word.
# 	- `I-O`: Inside a non-entity word.
# - **Advantage**: Distinguishes the beginning of non-entity sequences.
# 
# ### IE
# - **Description**: Similar to IOE but for non-entity words.
# - **Tags**:
# 	- `I`: Inside a named entity.
# 	- `O`: Outside any named entity.
# 	- `E`: End of a named entity.
# 	- `E-O`: End of a non-entity word.
# 	- `I-O`: Inside a non-entity word.
# - **Advantage**: Highlights the end of non-entity sequences.
# 
# ### BIES
# - **Description**: Encodes both entities and non-entity words using the IOBES method.
# - **Tags**:
# 	- `B`: Beginning of a named entity.
# 	- `I`: Inside a named entity.
# 	- `O`: Outside any named entity.
# 	- `E`: End of a named entity.
# 	- `S`: Single-token named entity.
# 	- `B-O`: Beginning of a non-entity word.
# 	- `I-O`: Inside a non-entity word.
# 	- `S-O`: Single non-entity token.
# - **Advantage**: Comprehensive encoding for both entities and non-entities.
# 
# 
# 



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
DEVICE = "mps" if th.backends.mps.is_available() else "cpu"
# TO DO ----------------------------------------------------------------------
#assert DEVICE == "cuda"

DataType = list[tuple[list[str],list[str]]]


# # Part 1 - Dataset Preparation

# ## Step 1: Read Data
# Write a function for reading the data from a single file (of the ones that are provided above).   
# - The function recieves a filepath
# - The funtion encodes every sentence individually using a pair of lists, one list contains the words and one list contains the tags.
# - Each list pair will be added to a general list (data), which will be returned back from the function.
# 
# Example output:
# ```
# [
# 	(['At','Trent','Bridge',':'],['O','B-LOC','I-LOC ','O']),
# 	([...],[...]),
# 	...
# ]
# ```



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


# In[161]:


train = read_data("data/train.txt")
dev = read_data("data/dev.txt")
test = read_data("data/test.txt")


# ## Step 2: Create Vocab
# 
# The `Vocab` class will serve as a dictionary that maps words and tags into IDs. Ensure that you include special tokens to handle out-of-vocabulary words and padding.
# 
# ### Your Task
# 1. **Define Special Tokens**: Define special tokens such as `PAD_TOKEN` and `UNK_TOKEN` and assign them unique IDs.
# 2. **Initialize Dictionaries**: Populate the word and tag dictionaries based on the training set.
# 
# *Note: You may change the `Vocab` class as needed.*

# In[162]:


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


# In[163]:


vocab = Vocab(train)


# ## Step 3: Prepare Data
# Write a function `prepare_data` that takes one of the [train, dev, test] and the `Vocab` instance, for converting each pair of (words, tags) to a pair of indexes. Additionally, the function should pad the sequences to the maximum length sequence **of the given split**.
# 
# Note: Vocabulary is based only on the train set.
# 
# ### Your Task
# 1. Convert each pair of (words, tags) to a pair of indexes using the Vocab instance.
# 2. Pad the sequences to the maximum length of the sequences in the given split.

# In[164]:


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


# In[165]:


train_sequences = prepare_data(train, vocab)
dev_sequences = prepare_data(dev, vocab)
test_sequences = prepare_data(test, vocab)


# ### Your Task
# Print the number of OOV in dev and test sets:

# In[166]:


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


# ## Step 4: Dataloaders
# Create dataloaders for each split in the dataset. They should return the samples as Tensors.
# 
# **Hint** - you can create a Dataset to support this part.
# 
# For the training set, use shuffling, and for the dev and test, not.

# In[167]:


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


data_loader_train = prepare_data_loader(train_sequences, batch_size=16)
for batch in data_loader_train:
	inputs, labels = batch
	print(inputs.shape)
	print(labels.shape)
	break


# In[168]:


BATCH_SIZE = 16
dl_train = prepare_data_loader(train_sequences, batch_size=BATCH_SIZE)
dl_dev = prepare_data_loader(dev_sequences, batch_size=BATCH_SIZE, train=False)
dl_test = prepare_data_loader(test_sequences, batch_size=BATCH_SIZE, train=False)


# # Part 2 - NER Model Training

# ## Step 1: Implement Model
# 
# Write NERNet, a PyTorch Module for labeling words with NER tags.
# 
# > `input_size`: the size of the vocabulary  
# `embedding_size`: the size of the embeddings  
# `hidden_size`: the LSTM hidden size  
# `output_size`: the number tags we are predicting for  
# `n_layers`: the number of layers we want to use in LSTM  
# `directions`: could 1 or 2, indicating unidirectional or bidirectional LSTM, respectively  
# 
# <br>  
# 
# The input for your forward function should be a single sentence tensor.
# 
# *Note: the embeddings in this section are learned embedding. That means that you don't need to use pretrained embedding like the one used in the last excersie. You will use them in part 5.*
# 
# *Note: You may change the NERNet class.*

# In[172]:


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
		self.fc = nn.Linear(hidden_size*directions, output_size)
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


# In[176]:


model = NERNet(vocab.n_words, embedding_size=300, hidden_size=800, output_size=vocab.n_tags, n_layers=2, directions=1)
model.to(DEVICE)


# ## Step 2: Training Loop
# 
# Write a training loop, which takes a model (instance of NERNet), number of epochs to train on, and the train&dev datasets.  
# 
# The function will return the `loss` and `accuracy` durring training.  
# (If you're using a different/additional metrics, return them too)
# 
# The loss is always CrossEntropyLoss and the optimizer is always Adam.
# Make sure to use `tqdm` while iterating on `n_epochs`.
# 

# In[186]:


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


# In[187]:


metrics = train_loop(model, n_epochs=5, dataloader_train=dl_train, dataloader_dev=dl_dev)
metrics


# <br><br><br><br><br><br>

# # Part 3 - Evaluation

# 
# ## Step 1: Evaluation Function
# 
# Write an evaluation loop for a trained model using the dev and test datasets. This function will print the `Recall`, `Precision`, and `F1` scores and plot a `Confusion Matrix`.
# 
# Perform this evaluation twice:
# 1. For all labels (7 labels in total).
# 2. For all labels except "O" (6 labels in total).

# ## Metrics and Display
# 
# ### Metrics
# - **Recall**: True Positive Rate (TPR), also known as Recall.
# - **Precision**: The opposite of False Positive Rate (FPR), also known as Precision.
# - **F1 Score**: The harmonic mean of Precision and Recall.
# 
# *Note*: For all these metrics, use **weighted** averaging:
# Calculate metrics for each label, and find their average weighted by support. Refer to the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support) for more details.
# 
# ### Display
# 1. Print the `Recall`, `Precision`, and `F1` scores in a tabulated format.
# 2. Display a `Confusion Matrix` plot:
# 	- Rows represent the predicted labels.
# 	- Columns represent the true labels.
# 	- Include a title for the plot, axis names, and the names of the tags on the X-axis.

# In[213]:


def evaluate(model: NERNet, title: str, dataloader: DataLoader, vocab: Vocab):
	results = {}
	model.eval()
	all_preds = []
	all_labels = []
	results["title"] = title	
	pad_tag_id = vocab.tag2id.get("<PAD>", vocab.tag2id.get("O", 0))

	with th.no_grad():
		for batch in dataloader:
			inputs, labels = batch
			inputs = inputs.to(DEVICE)
			labels = labels.to(DEVICE)
			outputs = model(inputs)
			preds = outputs.argmax(dim=2)  # shape: (B, T)

			all_preds.extend(preds.cpu().numpy().flatten())
			all_labels.extend(labels.cpu().numpy().flatten())

	# calculate 
	
	return results


# ## Step 2: Train & Evaluate on Dev Set

# Train and evaluate (on the dev set) a few models, all with `embedding_size=300` and `N_EPOCHS=5` (for fairness and computational reasons), and with the following hyper parameters (you may use that as captions for the models as well):
# 
# - Model 1: (hidden_size: 500, n_layers: 1, directions: 1)
# - Model 2: (hidden_size: 500, n_layers: 2, directions: 1)
# - Model 3: (hidden_size: 500, n_layers: 3, directions: 1)
# - Model 4: (hidden_size: 500, n_layers: 1, directions: 2)
# - Model 5: (hidden_size: 500, n_layers: 2, directions: 2)
# - Model 6: (hidden_size: 500, n_layers: 3, directions: 2)
# - Model 7: (hidden_size: 800, n_layers: 1, directions: 2)
# - Model 8: (hidden_size: 800, n_layers: 2, directions: 2)
# - Model 9: (hidden_size: 800, n_layers: 3, directions: 2)
# 
# 
# 

# In[189]:


N_EPOCHS = 5
EMB_DIM = 300


# Here is an example (random numbers) of the display of the results):

# In[190]:


# Example:
results_acc = np.random.rand(9, 10)
columns = ['N_MODEL','HIDDEN_SIZE','N_LAYERS','DIRECTIONS','RECALL','PERCISION','F1','RECALL_WO_O','PERCISION_WO_O','F1_WO_O']
df = pd.DataFrame(results_acc, columns=columns)
df.N_MODEL = [f'model_{n}' for n in range(1,10)]
print(tabulate(df, headers='keys', tablefmt='psql',floatfmt=".4f"))


# In[ ]:


# Define models with their hyperparameters
models = {
	'Model1': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 1, 'directions': 1},
	'Model2': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 2, 'directions': 1},
	'Model3': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 3, 'directions': 1},
	'Model4': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 1, 'directions': 2},
	# 'Model5': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 2, 'directions': 2},
	# 'Model6': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 3, 'directions': 2},
	# 'Model7': {'embedding_size': EMB_DIM, 'hidden_size': 800, 'n_layers': 1, 'directions': 2},
	# 'Model8': {'embedding_size': EMB_DIM, 'hidden_size': 800, 'n_layers': 2, 'directions': 2},
	# 'Model9': {'embedding_size': EMB_DIM, 'hidden_size': 800, 'n_layers': 3, 'directions': 2},
}

# TO DO ----------------------------------------------------------------------
results_dev = []

for model_name, model_params in models.items():
	print(f"\n🚀 Training {model_name}")
	model = NERNet(**model_params, input_size=vocab.n_words, output_size=vocab.n_tags)
	model.to(DEVICE)
	
	train_loop(model, n_epochs=N_EPOCHS, dataloader_train=dl_train, dataloader_dev=dl_dev)
	eval_result = evaluate(model, model_name, dl_dev, vocab)

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
	results_dev.append(summary)

# TO DO ----------------------------------------------------------------------

# Print results in tabulated format
print(tabulate(results_dev, headers='keys', tablefmt='psql', floatfmt=".4f"))


# ## Step 3: Evaluate on Test Set
# Evaluate your models on the test set and save the results as a CSV. Add this file to your repo for submission.

# In[209]:


results = pd.DataFrame(columns=columns)
file_name = "NER_results.csv"
# TO DO ----------------------------------------------------------------------
models = {
	'Model1': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 1, 'directions': 1},
	'Model2': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 2, 'directions': 1},
	'Model3': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 3, 'directions': 1},
	'Model4': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 1, 'directions': 2},
	'Model5': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 2, 'directions': 2},
	'Model6': {'embedding_size': EMB_DIM, 'hidden_size': 500, 'n_layers': 3, 'directions': 2},
	'Model7': {'embedding_size': EMB_DIM, 'hidden_size': 800, 'n_layers': 1, 'directions': 2},
	'Model8': {'embedding_size': EMB_DIM, 'hidden_size': 800, 'n_layers': 2, 'directions': 2},
	'Model9': {'embedding_size': EMB_DIM, 'hidden_size': 800, 'n_layers': 3, 'directions': 2},
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
# TO DO ----------------------------------------------------------------------
df = pd.DataFrame(results_test, columns=columns)
df.to_csv(file_name, index=False)
print(tabulate(df, headers='keys', tablefmt='psql',floatfmt=".4f"))


# ## Step 4 - best model
# Decide which model performs the best, write its configuration, train it for 5 more epochs and evaluate it on the test set.

# In[146]:


best_model_cfg = {'embedding_size':EMB_DIM, 'hidden_size': -1, 'n_layers': -1, 'directions': -1}
# TO DO ----------------------------------------------------------------------

# TO DO ----------------------------------------------------------------------


# <br><br><br><br><br>

# # Part 4 - Pretrained Embeddings

# 
# 
# To prepare for this task, please read [this discussion](https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222).
# 
# **TIP**: Ensure that the vectors are aligned with the IDs in your vocabulary. In other words, make sure that the word with ID 0 corresponds to the first vector in the GloVe matrix used to initialize `nn.Embedding`.
# 
# 

# ## Step 1: Get Data

# 
# 
# Download the GloVe embeddings from [this link](https://nlp.stanford.edu/projects/glove/). Use the 300-dimensional vectors from `glove.6B.zip`.
# 
# 

# In[147]:


# TO DO ----------------------------------------------------------------------

# TO DO ----------------------------------------------------------------------


# ## Step 2: Inject Embeddings

# Then intialize the `nn.Embedding` module in your `NERNet` with these embeddings, so that you can start your training with pre-trained vectors.

# In[148]:


def get_emb_matrix(filepath: str, vocab: Vocab) -> np.ndarray:
	emb_matrix = np.zeros((len(vocab.word2id), 300))
	# TO DO ----------------------------------------------------------------------

	# TO DO ----------------------------------------------------------------------
	return emb_matrix


# In[149]:


def initialize_from_pretrained_emb(model: NERNet, emb_matrix: np.ndarray):
	"""
	Inject the pretrained embeddings into the model.
	:param model: model instance
	:param emb_matrix: pretrained embeddings
	"""
	# TO DO ----------------------------------------------------------------------

	# TO DO ----------------------------------------------------------------------


# In[150]:


# Read embeddings and inject them to a model
emb_file = 'glove.6B.300d.txt'
emb_matrix = get_emb_matrix(emb_file, vocab)
ner_glove = NERNet(input_size=VOCAB_SIZE, embedding_size=EMB_DIM, hidden_size=500, output_size=NUM_TAGS, n_layers=1, directions=1)
initialize_from_pretrained_emb(ner_glove, emb_matrix)


# ## Step 3: Evaluate on Test Set

# Same as the evaluation process before, please display:
# 
# 1. Print a `RECALL-PERCISION-F1` scores in a tabulate format.
# 2. Display a `confusion matrix` plot: where the predicted labels are the rows, and the true labels are the columns.
# 
# Make sure to use the title for the plot, axis names, and the names of the tags on the X-axis.
# 
# Make sure to download and upload this CSV as well.

# In[ ]:


results = pd.DataFrame(columns=columns)
file_name = "NER_results_glove.csv"
# TO DO ----------------------------------------------------------------------

# TO DO ----------------------------------------------------------------------
print(tabulate(results, headers='keys', tablefmt='psql',floatfmt=".4f"))


# ## Step 4 - best model
# Decide which model performs the best, write its configuration, train it for 5 more epochs and evaluate it on the test set.

# In[ ]:


best_model_glove_cfg = {'embedding_size':EMB_DIM, 'hidden_size': -1, 'n_layers': -1, 'directions': -1}
# TO DO ----------------------------------------------------------------------

# TO DO ----------------------------------------------------------------------


# # Part 5 - Error Analysis

# In this part, you'll analyze the errors made by your best model to understand its strengths and weaknesses.

# ## Step 1: Extract Predictions
# 
# First, let's extract predictions from your best model on the test set:

# In[ ]:


def get_predictions(model, dataloader, vocab, PAD_TOKEN, DEVICE):
	"""
	Get predictions from the model on a dataloader.

	Returns:
		- true_tags_list: List of lists of true tag strings
		- pred_tags_list: List of lists of predicted tag strings
		- words_list: List of lists of words
	"""
	import torch

	model.eval()
	true_tags_list = []
	pred_tags_list = []
	words_list = []

	with torch.no_grad():
		# Handle different dataloader output formats
		for batch in dataloader:
			# Unpack based on actual dataloader output
			if len(batch) == 3:  # (input_ids, casing_features, labels)
				input_ids, casing_features, labels = batch
				# Move tensors to device
				input_ids = input_ids.to(DEVICE)
				casing_features = casing_features.to(DEVICE)
				labels = labels.to(DEVICE)

				# Get model predictions
				outputs = model(input_ids, casing_features)
			else:  # (input_ids, labels)
				input_ids, labels = batch
				# Move tensors to device
				input_ids = input_ids.to(DEVICE)
				labels = labels.to(DEVICE)

				# Get model predictions
				outputs = model(input_ids)

			_, predicted = torch.max(outputs, 2)

			# Process each sequence in the batch
			for i in range(input_ids.size(0)):
				# Get sequence length (ignoring padding)
				seq_len = (input_ids[i] != PAD_TOKEN).sum().item()

				# Convert ids to tag strings and words
				true_tags = [vocab.id2tag[tag.item()] for tag in labels[i][:seq_len]]
				pred_tags = [vocab.id2tag[tag.item()] for tag in predicted[i][:seq_len]]
				words = [vocab.id2word[word.item()] for word in input_ids[i][:seq_len]]

				true_tags_list.append(true_tags)
				pred_tags_list.append(pred_tags)
				words_list.append(words)

	return true_tags_list, pred_tags_list, words_list


# ## Step 2: Implement Simple Error Analysis
# 
# Now, implement a function to analyze the errors in predictions:

# In[ ]:


def simple_analyze_errors(true_tags, pred_tags, words):
	"""
	Analyze errors in NER predictions.

	Args:
		true_tags: List of true tag sequences
		pred_tags: List of predicted tag sequences
		words: List of word sequences

	Returns:
		dict: Error statistics and examples
	"""
	# TODO: Implement error analysis
	# 1. Initialize error categories
	# 2. Process each sequence to identify errors
	# 3. Categorize errors and collect examples
	# 4. Return statistics and examples

	# Placeholder
	return {
		'total_entities': 0,
		'correct_entities': 0,
		'accuracy': 0.0,
		'error_counts': {},
		'error_examples': {}
	}


# ## Step 3: Helper Functions
# 
# Implement these helper functions to extract entities and check for overlapping spans:

# In[ ]:


def get_entities_simple(tags):
	"""
	Extract entities from a sequence of tags.
	Returns list of (start_idx, end_idx, entity_type) tuples.
	"""
	# TODO: Implement entity extraction
	return []

def has_overlap(start1, end1, start2, end2):
	"""Check if two spans overlap"""
	# TODO: Implement overlap checking
	return False


# ## Step 4: Visualization and Analysis
# 
# Create a function to display the error analysis results:

# In[ ]:


def print_error_analysis(analysis):
	"""Print a summary of the error analysis results"""
	# TODO: Implement printing function to show:
	# 1. Basic statistics (total entities, correct entities, accuracy)
	# 2. Error counts by category
	# 3. Examples of each error type
	# 4. Suggestions for improvement based on findings
	pass


# ## Step 5: Improvement Suggestions
# 
# Based on your error analysis, suggest at least three specific improvements to your model. Consider:
# 
# 1. What types of errors are most common?
# 2. Are there patterns in the errors (e.g., specific entity types, contexts)?
# 3. What techniques might address these specific error types?
# 
# Write your suggestions in 3-5 sentences for each improvement.

# In[ ]:


# Example usage
if __name__ == "__main__":
	# Sample data for testing
	true_tags = [
		['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O'],
		['B-ORG', 'I-ORG', 'O', 'B-PER', 'O']
	]

	pred_tags = [
		['O', 'B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'O'],
		['B-ORG', 'I-ORG', 'I-ORG', 'B-PER', 'O']
	]

	words = [
		['The', 'John', 'Smith', 'visited', 'New', 'York', 'yesterday'],
		['Google', 'Inc', 'hired', 'Alice', 'recently']
	]

	# Run the error analysis
	analysis = simple_analyze_errors(true_tags_list, pred_tags_list, words_list)
	print_error_analysis(analysis)

	# TODO: Write your improvement suggestions here


# # Testing
# Copy the content of the **tests.py** file from the repo and paste below. This will create the results.json file and download it to your machine.

# In[ ]:


####################
# PLACE TESTS HERE #
train_ds = read_data("data/train.txt")
dev_ds = read_data("data/dev.txt")
test_ds = read_data("data/test.txt")
def test_read_data():
	result = {
		'lengths': (len(train_ds), len(dev_ds), len(test_ds)),
	}
	return result

vocab = Vocab(train_ds)
def test_vocab():
	sent = vocab.index_words(["I", "am", "Spongebob"])
	return {
		'length': vocab.n_words,
		'tag2id_length': len(vocab.tag2id),
		"Spongebob": sent[2]
	}

train_sequences = prepare_data(train_ds, vocab)
dev_sequences = prepare_data(dev_ds, vocab)
test_sequences = prepare_data(test_ds, vocab)

def test_count_oov():
	return {
		'dev_oov': count_oov(dev_sequences),
		'test_oov': count_oov(test_sequences)
	}

BATCH_SIZE = 16
dl_train = prepare_data_loader(train_sequences, batch_size=BATCH_SIZE)
dl_dev = prepare_data_loader(dev_sequences, batch_size=BATCH_SIZE, train=False)
dl_test = prepare_data_loader(test_sequences, batch_size=BATCH_SIZE, train=False)

def test_prepare_data_loader():
	return {
		'lengths': (len(dl_train), len(dl_dev), len(dl_test))
	}


def test_NERNet():
	# Extract best model configuration
	hidden_size = best_model_cfg['hidden_size']
	n_layers = best_model_cfg['n_layers']
	directions = best_model_cfg['directions']


	# Create model
	best_model = NERNet(vocab.n_words, embedding_size=300, hidden_size=hidden_size, output_size=vocab.n_tags, n_layers=n_layers, directions=directions)
	best_model.to(DEVICE)

	# Train model and evaluate
	_ = train_loop(best_model, n_epochs=10, dataloader_train=dl_train, dataloader_dev=dl_dev)
	results = evaluate(best_model, title="", dataloader=dl_test, vocab=vocab)

	return {
		'f1': results['F1'],
		'f1_wo_o': results['F1_WO_O'],
	}

def test_glove():
	# Get embeddings
	emb_file = 'glove.6B.300d.txt'
	emb_matrix = get_emb_matrix(emb_file, vocab)

	# Extract best model configuration
	hidden_size = best_model_glove_cfg['hidden_size']
	n_layers = best_model_glove_cfg['n_layers']
	directions = best_model_glove_cfg['directions']

	# Create model
	best_model = NERNet(vocab.n_words, embedding_size=300, hidden_size=hidden_size, output_size=vocab.n_tags, n_layers=n_layers, directions=directions)
	best_model.to(DEVICE)
	initialize_from_pretrained_emb(ner_glove, emb_matrix)

	# Train model and evaluate
	_ = train_loop(best_model, n_epochs=10, dataloader_train=dl_train, dataloader_dev=dl_dev)
	results = evaluate(best_model, title="", dataloader=dl_test, vocab=vocab)

	return {
		'f1': results['F1'],
		'f1_wo_o': results['F1_WO_O'],
	}

TESTS = [
	test_read_data,
	test_vocab,
	test_count_oov,
	test_prepare_data_loader,
	test_NERNet,
	test_glove
]

# Run tests and save results
res = {}
for test in TESTS:
	try:
		cur_res = test()
		res.update({test.__name__: cur_res})
	except Exception as e:
		res.update({test.__name__: repr(e)})

with open('results.json', 'w') as f:
	json.dump(res, f, indent=2)

# Download the results.json file
files.download('results.json')

####################

