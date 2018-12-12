# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


######################################################################
# Exercise: Augmenting the LSTM part-of-speech tagger with character-level features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# *  two LSTM's: one outputs POS tag scores, and one outputs a character-level representation of each word.
# * The character embeddings are the input to the character LSTM.
#
# I'm not sure whether it is a right solution


torch.manual_seed(1)
# Prepare data:

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

# word-to-index, character-to-index and tag-to-index
word_to_ix = {}
char_to_ix = {}
tag_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
print(word_to_ix)
print(char_to_ix)
print(tag_to_ix)

# index-to-word, index is distinct and unique
ix_to_word = {v : k for k, v in word_to_ix.items()}



######################################################################
# Create the model:

class LSTMTagger(nn.Module):

    def __init__(self,word_embedding_dim,word_hidden_dim, char_embedding_dim, char_hidden_dim, vocab_size, char_size,tagset_size):
        super(LSTMTagger, self).__init__()
        self.word_hidden_dim = word_hidden_dim
        self.char_hidden_dim = char_hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)

        # char level lstm
        # To get the character level representation,
        # do an LSTM over the characters of a word, and let cw be the final hidden state of this LSTM (character representation)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)

        # The LSTM takes word embeddings+ character representation as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.word_lstm = nn.LSTM(word_embedding_dim + char_hidden_dim, word_hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(word_hidden_dim, tagset_size)
        self.word_hidden = self.init_word_hidden()
        self.char_hidden = self.init_char_hidden()

    def init_word_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.word_hidden_dim),
                torch.zeros(1, 1, self.word_hidden_dim))

    def init_char_hidden(self):
        return (torch.zeros(1, 1, self.char_hidden_dim),
                torch.zeros(1, 1, self.char_hidden_dim))

    def forward(self, sentence):
        word_embedss = []
        for ix in sentence:
            word = ix_to_word[ix.item()]
            #print(word)
            word_embed = self.word_embeddings(ix)
            #print("word_embed")
            #print(word_embed)
            self.char_hidden = self.init_char_hidden()
            char_in = prepare_sequence(word, char_to_ix)
            char_embeds = self.char_embeddings(char_in)
            char_lstm_out,self.char_hidden = self.char_lstm(char_embeds.view(len(char_in),1,-1),self.char_hidden)
            #print("char_lstm_out")
            #print(char_lstm_out)
            #print("char_lstm_out_last")
            char_lstm_out_view = char_lstm_out.view(len(char_in),-1)
            #indices = torch.tensor(len(char_in)-1)
            #char_lstm_out_last = torch.index_select(char_lstm_out_view,0,indices)
            #print(char_lstm_out_view)
            #print(char_lstm_out_last)
            #print(char_lstm_out_view[len(char_in)-1])
            #print(torch.index_select(char_lstm_out,0,torch.tensor([len(char_in)-1]))
            # using the last output of the sequence.  why
            char_lstm_out_last = char_lstm_out_view[len(char_in)-1]
            #列连接
            z = torch.cat((word_embed, char_lstm_out_last),-1)
            #print(z)
            word_embedss.append(z)
        word_embedss = torch.cat(word_embedss)
        #print(word_embedss)
        lstm_outs, self.word_hidden = self.word_lstm(word_embedss.view(len(sentence), 1, -1), self.word_hidden)
        tag_space = self.hidden2tag(lstm_outs.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

######################################################################
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
# what is the relation about CHAR_EMBEDDING_DIM and CHAR_HIDDEN_DIM
# CHAR_EMBEDDING_DIM=CHAR_HIDDEN_DIM=25
HIDDEN_DIM = 5
WORD_EMBEDDING_DIM = 6
CHAR_EMBEDDING_DIM = 6
CHAR_HIDDEN_DIM = 3
# Train the model:

#word_embedding_dim,word_hidden_dim, char_embedding_dim, char_hidden_dim, vocab_size, char_size,tagset_size
model = LSTMTagger(WORD_EMBEDDING_DIM, HIDDEN_DIM,CHAR_EMBEDDING_DIM,CHAR_HIDDEN_DIM, len(word_to_ix), len(char_to_ix),len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
print('the scores before training:')
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
print('training----------------------------------------------------:')
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.word_hidden = model.init_word_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        print('Epoch {}: Loss = {}.'.format(epoch + 1, loss.item()))

# See what the scores are after training
print('scores  after training')
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
