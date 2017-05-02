import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T

import math


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
        self.init_hidden()

    def init_hidden(self):
        self.hidden = (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def init_hidden_cuda(self):
        self.hidden = (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))

    def forward(self, embeds):
        embeds = self.drop(embeds)
        lstm_out, self.hidden = self.lstm(embeds.view(len(embeds), 1, -1), self.hidden)
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




class FeatureTransfer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FeatureTransfer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.transpose(x, 0, 1)

        # print(x.size())
        return self.drop(F.tanh(self.linear(x)))
        # return F.tanh(self.linear(x))

class WordEmbedding2(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(WordEmbedding2, self).__init__()
        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(0.5)


    def forward(self, word_id):
        # Pass the word id through the embedding layer,
        return self.drop(F.tanh(self.embeds(word_id)))
        # return self.embeds(word_id)


class Attention(nn.Module):
    def __init__(self, input_size, atten_size):
        super(Attention, self).__init__()
        self.ln1 = nn.Linear(input_size, atten_size)

        self.ln2 = nn.Linear(input_size, atten_size, bias=False)
        self.drop = nn.Dropout(0.5)
        self.ln3 = nn.Linear(atten_size, 1)
    def forward(self, V_I, V_Q):
        x1 = self.ln1(V_Q)
        x2 = self.ln2(V_I)

        for x_temp in range(x2.size(0)):
            x2.data[x_temp] = x2.data[x_temp] + x1.data[0]
        h_a = self.drop(F.tanh(x2))   # [196 * 512(k)]
        p = F.softmax(self.ln3(h_a).view(-1, 196))

        return p

































class LSTMAttention(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, context_size):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = 1

        self.input_weights_1 = nn.Parameter(
            torch.Tensor(4 * hidden_size, input_size)
        )
        self.hidden_weights_1 = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size)
        )
        self.input_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.input_weights_2 = nn.Parameter(
            torch.Tensor(4 * hidden_size, context_size)
        )
        self.hidden_weights_2 = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size)
        )
        self.input_bias_2 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_2 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.context2attention = nn.Parameter(
            torch.Tensor(context_size, context_size)
        )
        self.bias_context2attention = nn.Parameter(torch.Tensor(context_size))

        self.hidden2attention = nn.Parameter(
            torch.Tensor(context_size, hidden_size)
        )

        self.input2attention = nn.Parameter(
            torch.Tensor(input_size, context_size)
        )

        self.recurrent2attention = nn.Parameter(torch.Tensor(context_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv_ctx = 1.0 / math.sqrt(self.context_size)

        self.input_weights_1.data.uniform_(-stdv, stdv)
        self.hidden_weights_1.data.uniform_(-stdv, stdv)
        self.input_bias_1.data.fill_(0)
        self.hidden_bias_1.data.fill_(0)

        self.input_weights_2.data.uniform_(-stdv_ctx, stdv_ctx)
        self.hidden_weights_2.data.uniform_(-stdv, stdv)
        self.input_bias_2.data.fill_(0)
        self.hidden_bias_2.data.fill_(0)

        self.context2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.bias_context2attention.data.fill_(0)

        self.hidden2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.input2attention.data.uniform_(-stdv_ctx, stdv_ctx)

        self.recurrent2attention.data.uniform_(-stdv_ctx, stdv_ctx)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden, projected_input, projected_ctx):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            print(hx)
            print(self.hidden_weights_1)
            print(self.hidden_bias_1)

            gates = F.linear(
                input, self.input_weights_1, self.input_bias_1
            ) + F.linear(hx[0], self.hidden_weights_1, self.hidden_bias_1)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            # Attention mechanism

            # Project current hidden state to context size
            hidden_ctx = F.linear(hy, self.hidden2attention)

            # Added projected hidden state to each projected context
            hidden_ctx_sum = projected_ctx + hidden_ctx.unsqueeze(0).expand(
                projected_ctx.size()
            )

            # Add this to projected input at this time step
            hidden_ctx_sum = hidden_ctx_sum + \
                projected_input.unsqueeze(0).expand(hidden_ctx_sum.size())

            # Non-linearity
            hidden_ctx_sum = F.tanh(hidden_ctx_sum)

            # Compute alignments
            alpha = torch.bmm(
                hidden_ctx_sum.transpose(0, 1),
                self.recurrent2attention.unsqueeze(0).expand(
                    hidden_ctx_sum.size(1),
                    self.recurrent2attention.size(0),
                    self.recurrent2attention.size(1)
                )
            ).squeeze()
            alpha = F.softmax(alpha)
            weighted_context = torch.mul(
                ctx, alpha.t().unsqueeze(2).expand(ctx.size())
            ).sum(0).squeeze()

            gates = F.linear(
                weighted_context, self.input_weights_2, self.input_bias_2
            ) + F.linear(hy, self.hidden_weights_2, self.hidden_bias_2)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cy) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        input = input.transpose(0, 1)
        projected_ctx = torch.bmm(
            ctx,
            self.context2attention.unsqueeze(0).expand(
                ctx.size(0),
                self.context2attention.size(0),
                self.context2attention.size(1)
            ),
        )
        projected_ctx += \
            self.bias_context2attention.unsqueeze(0).unsqueeze(0).expand(
                projected_ctx.size()
            )

        projected_input = torch.bmm(
            input,
            self.input2attention.unsqueeze(0).expand(
                input.size(0),
                self.input2attention.size(0),
                self.input2attention.size(1)
            ),
        )

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(
                input[i], hidden, projected_input[i], projected_ctx
            )
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return output, hidden
