import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.autograd as autograd


class ImgLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(ImgLinear, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        # Pass the image features through the linear layer,
        return self.linear(image_features)


class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(WordEmbedding, self).__init__()

        self.embeds = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, word_id):
        # Pass the word id through the embedding layer,
        return self.embeds(word_id)


# class Merge(nn.Module):
#
#     def __init__(self):
#         super(ImgLinear, self).__init__()
#
#     def forward(self, image_features, word_embedding):
#


class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, dropout=0.5):
        super(LSTM, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # self.hidden2Classifier == nn.Linear(hidden_dim, ans_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, embeds):
        embeds = self.drop(embeds)
        lstm_out, self.hidden = self.lstm(embeds.view(len(embeds), 1, -1))
        # ans_space = self.hidden2Classifier(lstm_out.view(len(embeds), -1))
        # ans_scores = F.log_softmax(ans_space)
        return lstm_out


class Classifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, lstm_out):
        ans_space = self.linear(lstm_out.view(len(lstm_out), -1))
        ans_scores = F.log_softmax(ans_space)
        return ans_scores
