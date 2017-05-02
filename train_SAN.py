import keras
import numpy as np
import my_models
import keras.preprocessing.text
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import copy
from torchvision import models
import torch.autograd as autograd
import gc



use_gpu = torch.cuda.is_available()
# use_gpu = True

# questions
cocoqa_train_q = np.genfromtxt("cocoqa/train/questions.txt", dtype = str, delimiter='\n')
cocoqa_test_q = np.genfromtxt("cocoqa/test/questions.txt", dtype = str, delimiter='\n')
# print(len(cocoqa_train_q))
# print(len(cocoqa_test_q))

# split questions into words
tokenizer = keras.preprocessing.text.Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
tokenizer.fit_on_texts(cocoqa_train_q)
tokenizer.fit_on_texts(cocoqa_test_q)

# conver questions into sequences of word ids
sequences_train_q = tokenizer.texts_to_sequences(cocoqa_train_q)
sequences_test_q = tokenizer.texts_to_sequences(cocoqa_test_q)

word2id_q = tokenizer.word_index
id2word_q = {idx: word for (word, idx) in word2id_q.items()}
vocab_size = len(word2id_q)

# max question length
maxSequenceLength = max([len(seq) for seq in sequences_train_q + sequences_train_q])

# answers
cocoqa_train_a = np.genfromtxt("cocoqa/train/answers.txt", dtype = str)
cocoqa_test_a = np.genfromtxt("cocoqa/test/answers.txt", dtype = str)

# split answers into words
tokenizer = keras.preprocessing.text.Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
tokenizer.fit_on_texts(cocoqa_train_a)
tokenizer.fit_on_texts(cocoqa_test_a)

# conver answers into sequences of word ids
sequences_train_a = tokenizer.texts_to_sequences(cocoqa_train_a)
sequences_test_a = tokenizer.texts_to_sequences(cocoqa_test_a)
# print(sequences_train_a[0])
# print(cocoqa_train_a[0])

word2id_a = tokenizer.word_index
id2word_a = {idx: word for (word, idx) in word2id_a.items()}
# maxSequenceLength = max([len(seq) for seq in captionSequences])
a_size = len(word2id_a)
# print(a_size)
# print(vocab_size)
# print(id2word_a[0])
# print(id2word_a[a_size])


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model(model_dict, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_acc = 0.0

    # featureTransfer = model_list[0]
    # wordEmbedding2 = model_list[1]
    # attention = model_list[2]
    # classifier = model_list[3]
    # lstm = model_list[4]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                for key in model_dict.keys():
                    model_dict[key].train(True)  # Set model to training mode
            else:
                for key in model_dict.keys():
                    model_dict[key].train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            time_stamp = time.time()
            train_or_val_size = 0
            for part in range(8):
                if phase == 'train':
                    features = torch.load('features/train_features_SAN_%d.pth'%(part))
                    questions = sequences_train_q
                    answers = sequences_train_a
                else:

                    features = torch.load('features/val_features_SAN_%d.pth'%(part))
                    questions = sequences_test_q
                    answers = sequences_test_a


                # Iterate over data.
                for i in range(len(features)):
                    # get the inputs

                    current_f = features[i]
                    current_q = torch.LongTensor(questions[train_or_val_size + i])
                    # Note: here I -1 because keras tokenizer does not has word with id 0
                    current_a = torch.LongTensor([answers[train_or_val_size + i][0] - 1])


                    # wrap them in Variable
                    if use_gpu:
                        current_f, current_q, current_a = Variable(current_f.cuda()), Variable(current_q.cuda()), Variable(current_a.cuda())
                    else:
                        current_f, current_q, current_a = Variable(current_f), Variable(current_q), Variable(current_a)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    FI = current_f
                    V_I = model_dict['feature_trans'](FI)

                    Q_embed = model_dict['word_embed'](current_q)


                    if use_gpu:
                        hidden = (autograd.Variable(torch.zeros(1, 1, 1024)).cuda(),
                                autograd.Variable(torch.zeros(1, 1, 1024)).cuda())
                    else:
                        hidden = (autograd.Variable(torch.zeros(1, 1, 1024)),
                                autograd.Variable(torch.zeros(1, 1, 1024)))
                    lstm_output, hidden = model_dict['lstm'](Q_embed.view(len(Q_embed), 1, -1), hidden)

                    V_Q = lstm_output[-1].view(1, 1024)

                    # stack 1
                    p = model_dict['atten1'](V_I, V_Q)
                    V_I_next = torch.mm(p, V_I)
                    V_Q = V_I_next + V_Q

                    # stack 2
                    p = model_dict['atten2'](V_I, V_Q)
                    V_I_next = torch.mm(p, V_I)
                    V_Q = V_I_next + V_Q

                    class_score = model_dict['classifier'](V_Q)
                    _, preds = torch.max(class_score.data, 1)

                    loss = criterion(class_score, current_a)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds.view(1) == current_a.data)

                train_or_val_size += len(features)

            epoch_loss = running_loss / train_or_val_size
            epoch_acc = running_corrects / train_or_val_size

            print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.0f}s'.format(
                phase, epoch_loss, epoch_acc, time.time() - time_stamp))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

                torch.save(model_dict['feature_trans'], 'models3/05-01/feature_trans.pth')
                torch.save(model_dict['word_embed'], 'models3/05-01/word_embed.pth')
                torch.save(model_dict['lstm'], 'models3/05-01/lstm.pth')
                torch.save(model_dict['atten1'], 'models3/05-01/atten1.pth')
                torch.save(model_dict['atten2'], 'models3/05-01/atten2.pth')
                torch.save(model_dict['classifier'], 'models3/05-01/classifier.pth')

        print (" ")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model_dict


models = {}
models['feature_trans'] = (my_models.FeatureTransfer(512, 1024))
models['word_embed'] = (my_models.WordEmbedding2(vocab_size + 1, 500))
models['lstm'] = (nn.LSTM(500, 1024))
models['atten1'] = (my_models.Attention(1024, 512))
models['atten2'] = (my_models.Attention(1024, 512))
models['classifier'] = (my_models.Classifier(1024, a_size))

# models = {}
# models['feature_trans'] = torch.load('models3/04-24/feature_trans.pth')
# models['word_embed'] = torch.load('models3/04-24/word_embed.pth')
# models['lstm'] = torch.load('models3/04-24/lstm.pth')
# models['atten1'] = torch.load('models3/04-24/atten1.pth')
# models['atten2'] = torch.load('models3/04-24/atten2.pth')
# models['classifier'] = torch.load('models3/04-24/classifier.pth')


if use_gpu:
    for key in models.keys():
        models[key].cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([
    {'params': models['feature_trans'].parameters()},
    {'params': models['word_embed'].parameters()},
    {'params': models['lstm'].parameters()},
    {'params': models['atten1'].parameters()},
    {'params': models['atten2'].parameters()},
    {'params': models['classifier'].parameters()}
    ], lr=0.001, momentum=0.9)

models = train_model(models, criterion, optimizer, exp_lr_scheduler, num_epochs=60)
