import keras
import numpy as np
import my_models
import keras.preprocessing.text
import torch
from torch.autograd import Variable



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
# maxSequenceLength = max([len(seq) for seq in captionSequences])


cocoqa_train_a = np.genfromtxt("cocoqa/train/answers.txt", dtype = str)
cocoqa_test_a = np.genfromtxt("cocoqa/test/answers.txt", dtype = str)
# split sentences into words
tokenizer = keras.preprocessing.text.Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
tokenizer.fit_on_texts(cocoqa_train_a)
tokenizer.fit_on_texts(cocoqa_test_a)
# conver sentences into sequences of word ids
sequences_train_a = tokenizer.texts_to_sequences(cocoqa_train_a)
sequences_test_a = tokenizer.texts_to_sequences(cocoqa_test_a)

word2id_a = tokenizer.word_index
id2word_a = {idx: word for (word, idx) in word2id_a.items()}
# maxSequenceLength = max([len(seq) for seq in captionSequences])
a_size = len(word2id_a)
print(a_size)



# load image features
train_features = torch.load('train_features.pth')
test_features = torch.load('val_features.pth')

img_embed = my_models.ImgLinear(4096, 512)
word_embed = my_models.WordEmbedding(vocab_size ,512)
lstm = my_models.LSTM(512, 512)
classifier = my_models.Classifier(512, a_size)



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
