import keras
import numpy as np
import my_models
import keras.preprocessing.text
import torch
from torch.autograd import Variable
import time
import copy

use_gpu = torch.cuda.is_available()
Batch_Size = 200
use_gpu = True

cocoqa_train_q = np.genfromtxt("cocoqa/train/questions.txt", dtype = str, delimiter='\n')
cocoqa_test_q = np.genfromtxt("cocoqa/test/questions.txt", dtype = str, delimiter='\n')

# split sentences into words
tokenizer = keras.preprocessing.text.Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
tokenizer.fit_on_texts(cocoqa_train_q)
tokenizer.fit_on_texts(cocoqa_test_q)

# conver sentences into sequences of word ids
sequences_train_q = tokenizer.texts_to_sequences(cocoqa_train_q)
sequences_test_q = tokenizer.texts_to_sequences(cocoqa_test_q)


word2id_q = tokenizer.word_index
id2word_q = {idx: word for (word, idx) in word2id_q.items()}
vocab_size = len(word2id_q)

maxSequenceLength = max([len(seq) for seq in sequences_train_q + sequences_train_q])


cocoqa_train_a = np.genfromtxt("cocoqa/train/answers.txt", dtype = str)
cocoqa_test_a = np.genfromtxt("cocoqa/test/answers.txt", dtype = str)
# split sentences into words
tokenizer = keras.preprocessing.text.Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
tokenizer.fit_on_texts(cocoqa_train_a)
tokenizer.fit_on_texts(cocoqa_test_a)
# conver sentences into sequences of word ids
sequences_train_a = tokenizer.texts_to_sequences(cocoqa_train_a)
sequences_test_a = tokenizer.texts_to_sequences(cocoqa_test_a)
# print(sequences_train_a[0])
# print(cocoqa_train_a[0])

word2id_a = tokenizer.word_index
id2word_a = {idx: word for (word, idx) in word2id_a.items()}
# maxSequenceLength = max([len(seq) for seq in captionSequences])
a_size = len(word2id_a)
# print(a_size)

# print(id2word_a[0])
# print(id2word_a[a_size])

# load image features
train_features = torch.load('train_features.pth')
test_features = torch.load('val_features.pth')


'''
i_embedding = img_embed(Variable(train_features[0].view(1,-1))).view(1,-1)
# print(i_embedding)
# print(torch.Tensor([i_embedding]))
# print(i_embedding.size())

w_embedding = word_embed(Variable(torch.LongTensor(sequences_train_q[0])))

# print(i_embedding)
# print(w_embedding.size())

lstm_input = torch.cat((i_embedding, w_embedding), 0)
lstm_input = lstm_input.view(len(lstm_input), 1, -1)
print(lstm_input.size())
output = lstm(lstm_input)
print(output.size())
class_score = classifier(output[-1])
print(class_score)
'''


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model(model_list, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_models = model_list
    best_acc = 0.0
    img_embed = model_list[0]
    word_embed = model_list[1]
    lstm = model_list[2]
    classifier = model_list[3]


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                for model in model_list:
                    model.train(True)  # Set model to training mode
                features = train_features
                questions = sequences_train_q
                answers = sequences_train_a
            else:
                for model in model_list:
                    model.train(False)  # Set model to evaluate mode
                features = test_features
                questions = sequences_test_q
                answers = sequences_test_a

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            time_stamp = time.time()
            # for i in range(len(features)/Batch_Size):
            for i in range(len(features)):
                # get the inputs
                # inputs, labels = data
                '''
                current_f = features[i*Batch_Size: (i+1)*Batch_Size].view(Batch_Size,-1)
                current_q = torch.LongTensor(questions[i*Batch_Size: (i+1)*Batch_Size])
                # Note: here I -1 because keras tokenizer does not has word with id 0
                current_a = torch.LongTensor([x[0] - 1] for x in answers[i*Batch_Size: (i+1)*Batch_Size])
                '''

                current_f = features[i].view(1,-1)
                current_q = torch.LongTensor(questions[i])
                # Note: here I -1 because keras tokenizer does not has word with id 0
                current_a = torch.LongTensor([answers[i][0] - 1])


                # wrap them in Variable
                if use_gpu:
                    current_f, current_q, current_a = Variable(current_f.cuda()), Variable(current_q.cuda()), Variable(current_a.cuda())
                else:
                    current_f, current_q, current_a = Variable(current_f), Variable(current_q), Variable(current_a)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                i_embedding = img_embed(current_f).view(1,-1)

                w_embedding = word_embed(current_q)

                # lstm_input = torch.cat((i_embedding, w_embedding), 0)

                # lstm_input = lstm_input.view(len(lstm_input), 1, -1)
                if use_gpu:
                    lstm.init_hidden_cuda()
                else:
                    lstm.init_hidden()

                lstm(i_embedding.view(1, 1, -1))
                output = lstm(w_embedding.view(len(w_embedding), 1, -1))

                class_score = classifier(output[-1])
                _, preds = torch.max(class_score.data, 1)
                # print ("{}/{}".format(preds.view(1), current_a.data))

                # print(class_score.size())
                # print(current_a.size())
                # exit()
                loss = criterion(class_score, current_a)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.view(1) == current_a.data)

            epoch_loss = running_loss / len(features)
            epoch_acc = running_corrects / len(features)

            print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.0f}s'.format(
                phase, epoch_loss, epoch_acc, time.time() - time_stamp))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

                # best_models[0] = copy.deepcopy(img_embed)
                # best_models[1] = copy.deepcopy(word_embed)
                # best_models[2] = copy.deepcopy(lstm)
                # best_models[3] = copy.deepcopy(classifier)
                torch.save(img_embed, 'models/04-20/img_embed.pth')
                torch.save(word_embed, 'models/04-20/word_embed.pth')
                torch.save(lstm, 'models/04-20/lstm.pth')
                torch.save(classifier, 'models/04-20/classifier.pth')


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_models


models = []
models.append( my_models.ImgLinear(4096, 512))
models.append( my_models.WordEmbedding(vocab_size + 1 ,512))
models.append( my_models.LSTM(512, 512))
models.append( my_models.Classifier(512, a_size))

if use_gpu:
    for model in models:
        model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([
    {'params': models[0].parameters()},
    {'params': models[1].parameters()},
    {'params': models[2].parameters()},
    {'params': models[3].parameters()}
    ], lr=0.001, momentum=0.9)


models = train_model(models, criterion, optimizer, exp_lr_scheduler, num_epochs=60)
torch.save(models, 'model_04_20.pth')










#
# FI = Variable(train_features_0[0])
# featureTransfer = my_models.FeatureTransfer(512, 1024)
# V_I = featureTransfer(FI)
#
# word_embed = my_models.WordEmbedding2(vocab_size + 1, 500)
# Q_embed = word_embed(Variable(torch.LongTensor(sequences_train_q[0])))
# lstm = nn.LSTM(500, 1024)
# hidden = (autograd.Variable(torch.zeros(1, 1, 1024)),
#         autograd.Variable(torch.zeros(1, 1, 1024)))
# lstm_output, hidden = lstm(Q_embed.view(len(Q_embed), 1, -1), hidden)
#
# V_Q = lstm_output[-1].view(1, 1024)
#
# atten = my_models.Attention(1024, 512)
#
# # stack 1
# p = atten(V_I, V_Q)
# V_I_next = torch.mm(torch.transpose(p, 0, 1), V_I)
# V_Q = V_I_next + V_Q
#
# # stack 2
# p = atten(V_I, V_Q)
# V_I_next = torch.mm(torch.transpose(p, 0, 1), V_I)
# V_Q = V_I_next + V_Q
#
#
# classifier = my_models.Classifier(1024, a_size)
# class_score = classifier(V_Q)
# _, preds = torch.max(class_score.data, 1)
#
# print(preds)
