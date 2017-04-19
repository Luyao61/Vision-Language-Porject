import keras
import numpy as np


cocoqa_train_q = np.genfromtxt("cocoqa/train/questions.txt", dtype = str)
cocoqa_test_q = np.genfromtxt("cocoqa/test/questions.txt", dtype = str)
cocoqa_q = np.concatenate((cocoqa_train_q, cocoqa_test_q), axis=0)
# split sentences into words
tokenizer = keras.preprocessing.text.Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
tokenizer.fit_on_texts(cocoqa_q)
# conver sentences into sequences of word ids
sequences_q = tokenizer.texts_to_sequences(cocoqa_q)

word2id_q = tokenizer.word_index
id2word_q = {idx: word for (word, idx) in word2id.items()}
# maxSequenceLength = max([len(seq) for seq in captionSequences])

cocoqa_train_a = np.genfromtxt("cocoqa/train/answers.txt", dtype = str)
cocoqa_test_a = np.genfromtxt("cocoqa/test/answers.txt", dtype = str)
cocoqa_a = np.concatenate((cocoqa_train_a, cocoqa_test_a), axis=0)
# split sentences into words
tokenizer.fit_on_texts(cocoqa_a)
# conver sentences into sequences of word ids
sequences_a = tokenizer.texts_to_sequences(cocoqa_a)

word2id_a = tokenizer.word_index
id2word_a = {idx: word for (word, idx) in word2id.items()}
# maxSequenceLength = max([len(seq) for seq in captionSequences])
