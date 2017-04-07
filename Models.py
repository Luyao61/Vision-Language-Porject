import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



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

    def __init__(self, vocab_size, Embedding_dim):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(ImgLinear, self).__init__()

        self.embeds = nn.Embedding(vocab_size, Embedding_dim)

    def forward(self, word_id):
        # Pass the word id through the embedding layer,
        return self.embeds(word_id)


class concatenate(nn.Module):

    def __init__(self):
        super(ImgLinear, self).__init__()

    def forward(image_features, word_embedding):
        
