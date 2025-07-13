# Advanced Applications of Deep Learning in NLP

 In this chapter, we will cover the following advanced recipes:

*   Automated text generation from Shakespeare's writings using LSTM
*   Questions and answers on episodic data using memory networks
*   Language modeling to predict the next best word using recurrent neural networks – LSTM
*   Generative chatbot development using deep learning recurrent networks – LSTM

# Introduction

Deep learning techniques are being utilized well to solve some open-ended problems. This chapter discusses these types of problems, in which a simple *yes* or *no* would be difficult to say. We are hopeful that you will enjoy going through these recipes to obtain the viewpoint of what cutting-edge works are going on in this industry at the moment, and try to learn some basic building blocks of the same with relevant coding snippets.

# Automated text generation from Shakespeare's writings using LSTM

In this recipe, we will use deep **recurrent neural networks** (**RNN**) to predict the next character based on the given length of a sentence. This way of training a model can generate automated text continuously, which imitates the writing style of the original writer with enough training on the number of epochs and so on.

# Getting ready...

The *Project Gutenberg* eBook of the complete works of William Shakespeare's dataset is used to train the network for automated text generation. Data can be downloaded from [http://www.gutenberg.org/](http://www.gutenberg.org/) for the raw file used for training:

```py
>>> from __future__ import print_function
>>> import numpy as np
>>> import random
>>> import sys
```

The following code is used to create a dictionary of characters to indices and vice-versa mapping, which we will be using to convert text into indices at later stages. This is because deep learning models cannot understand English and everything needs to be mapped into indices to train these models:

```py
>>> path = 'C:\\Users\\prata\\Documents\\book_codes\\ NLP_DL\\ shakespeare_final.txt'
>>&gt; text = open(path).read().lower()
>>> characters = sorted(list(set(text)))
>>> print('corpus length:', len(text))
>>> print('total chars:', len(characters))
```

![](img/daf470df-74a1-4e55-b293-16f7fcfba4f5.png)

```py
>>> char2indices = dict((c, i) for i, c in enumerate(characters))
>>> indices2char = dict((i, c) for i, c in enumerate(characters))
```

# How to do it...

Before training the model, various preprocessing steps are involved to make it work. The following are the major steps involved:

1.  **Preprocessing**: Prepare *X* and *Y* data from the given entire story text file and converting them into indices vectorized format.
2.  **Deep learning model training and validation**: Train and validate the deep learning model.
3.  **Text generation**: Generate the text with the trained model.

# How it works...

The following lines of code describe the entire modeling process of generating text from Shakespeare's writings. Here we have chosen character length. This needs to be considered as `40` to determine the next best single character, which seems to be very fair to consider. Also, this extraction process jumps by three steps to avoid any overlapping between two consecutive extractions, to create a dataset more fairly:

```py
# cut the text in semi-redundant sequences of maxlen characters
>>> maxlen = 40
>>> step = 3
>>> sentences = []
>>> next_chars = []
>>> for i in range(0, len(text) - maxlen, step):
... sentences.append(text[i: i + maxlen])
... next_chars.append(text[i + maxlen])
... print('nb sequences:', len(sentences))
```

The following screenshot depicts the total number of sentences considered, `193798`, which is enough data for text generation:

![](img/dcf0bdeb-2033-4410-ab4e-9decd697a6db.png)

The next code block is used to convert the data into a vectorized format for feeding into deep learning models, as the models cannot understand anything about text, words, sentences and so on. Initially, total dimensions are created with all zeros in the NumPy array and filled with relevant places with dictionary mappings:

```py
# Converting indices into vectorized format
>>> X = np.zeros((len(sentences), maxlen, len(characters)), dtype=np.bool)
>>> y = np.zeros((len(sentences), len(characters)), dtype=np.bool)
>>> for i, sentence in enumerate(sentences):
... for t, char in enumerate(sentence):
... X[i, t, char2indices[char]] = 1
... y[i, char2indices[next_chars[i]]] = 1
>>> from keras.models import Sequential
>>> from keras.layers import Dense, LSTM,Activation,Dropout
>>> from keras.optimizers import RMSprop
```

The deep learning model is created with RNN, more specifically Long Short-Term Memory networks with `128` hidden neurons, and the output is in the dimensions of the characters. The number of columns in the array is the number of characters. Finally, the `softmax` function is used with the `RMSprop` optimizer. We encourage readers to try with other various parameters to check out how results vary:

```py
#Model Building
>>> model = Sequential()
>>> model.add(LSTM(128, input_shape=(maxlen, len(characters))))
>>> model.add(Dense(len(characters)))
>>> model.add(Activation('softmax'))
>>> model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
>>> print (model.summary())
```

![](img/34c322b4-4201-4882-94da-07bdfff39e8b.png)

As mentioned earlier, deep learning models train on number indices to map input to output (given a length of 40 characters, the model will predict the next best character). The following code is used to convert the predicted indices back to the relevant character by determining the maximum index of the character:

```py
# Function to convert prediction into index
>>> def pred_indices(preds, metric=1.0):
... preds = np.asarray(preds).astype('float64')
... preds = np.log(preds) / metric
... exp_preds = np.exp(preds)
... preds = exp_preds/np.sum(exp_preds)
... probs = np.random.multinomial(1, preds, 1)
... return np.argmax(probs)
```

The model will be trained over `30` iterations with a batch size of `128`. And also, the diversity has been changed to see the impact on the predictions:

```py
# Train and Evaluate the Model
>>> for iteration in range(1, 30):
... print('-' * 40)
... print('Iteration', iteration)
... model.fit(X, y,batch_size=128,epochs=1)
... start_index = random.randint(0, len(text) - maxlen - 1)
... for diversity in [0.2, 0.7,1.2]:
... print('\n----- diversity:', diversity)
... generated = ''
... sentence = text[start_index: start_index + maxlen]
... generated += sentence
... print('----- Generating with seed: "' + sentence + '"')
... sys.stdout.write(generated)
... for i in range(400):
... x = np.zeros((1, maxlen, len(characters)))
... for t, char in enumerate(sentence):
... x[0, t, char2indices[char]] = 1.
... preds = model.predict(x, verbose=0)[0]
... next_index = pred_indices(preds, diversity)
... pred_char = indices2char[next_index]
... generated += pred_char
... sentence = sentence[1:] + pred_char
... sys.stdout.write(pred_char)
... sys.stdout.flush()
... print("\nOne combination completed \n")
```

The results are shown in the next screenshot to compare the first iteration (`Iteration 1`) and final iteration (`Iteration 29`). It is apparent that with enough training, the text generation seems to be much better than with `Iteration 1`:

![](img/ad139862-1a1c-4394-af42-5ee640ba9f18.png)

Text generation after `Iteration 29` is shown in this image:

![](img/c0bb264f-1530-4235-8eb9-bcf4281467c4.png)

Though the text generation altogether seems to be a bit magical, we have generated text using Shakespeare's writings, proving that with the right training and handling, we can imitate any writer's style of writing.

# Questions and answers on episodic data using memory networks

In this recipe, we will use deep RNN to create a model to work on a question-and-answer system based on episodic memory. It will extract the relevant answers for a given question by reading a story in a sequential manner. For further reading, refer to the paper *Dynamic Memory Networks for Natural Language Processing* by Ankit Kumar et. al. ([https://arxiv.org/pdf/1506.07285.pdf](https://arxiv.org/pdf/1506.07285.pdf)).

# Getting ready...

Facebook's bAbI data has been used for this example, and the same can be downloaded from [http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz](http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz). It consists of about 20 types of tasks, among which we have taken the first one, a single supporting-fact-based question-and-answer system.

After unzipping the file, go to the `en-10k` folder and use the files starting with `qa1_single supporting-fact` for both the train and test files. The following code is used for extraction of stories, questions, and answers in this particular order to create the data required for training:

```py
>>> from __future__ import division, print_function
>>> import collections
>>> import itertools
>>> import nltk
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import os
>>> import random
>>> def get_data(infile):
... stories, questions, answers = [], [], []
... story_text = []
... fin = open(Train_File, "rb")
... for line in fin:
... line = line.decode("utf-8").strip()
... lno, text = line.split(" ", 1)
... if "\t" in text:
... question, answer, _ = text.split("\t")
... stories.append(story_text)
... questions.append(question)
... answers.append(answer)
... story_text = []
... else:
... story_text.append(text)
... fin.close()
... return stories, questions, answers
>>> file_location = "C:/Users/prata/Documents/book_codes/NLP_DL"
>>> Train_File = os.path.join(file_location, "qa1_single-supporting-fact_train.txt")
>>> Test_File = os.path.join(file_location, "qa1_single-supporting-fact_test.txt")
# get the data
>>> data_train = get_data(Train_File)
>>> data_test = get_data(Test_File)
>>> print("\n\nTrain observations:",len(data_train[0]),"Test observations:", len(data_test[0]),"\n\n")
```

After extraction, it seems that about 10k observations were created in the data for both train and test datasets:

![](img/a777c1c7-bf50-4b1a-97c5-6dee55c5a679.png)

# How to do it...

After extraction of basic datasets, we need to follow these steps:

1.  **Preprocessing**: Create a dictionary and map the story, question and answers to vocab to map into vector format.
2.  **Model development and validation**: Train the deep learning models and test on the validation data sample.
3.  **Predicting outcomes based on the trained model**: Trained models are utilized for predicting outcomes on test data.

# How it works...

After train and test data creation, the remaining methodology is described as follows.

First, we will create a dictionary for vocabulary, in which for every word from the story, question and answer data mapping is created. Mappings are used to convert words into integer numbers and subsequently into vector space:

```py
# Building Vocab dictionary from Train and Test data
>>> dictnry = collections.Counter()
>>> for stories,questions,answers in [data_train,data_test]:
... for story in stories:
... for sent in story:
... for word in nltk.word_tokenize(sent):
... dictnry[word.lower()] +=1
... for question in questions:
... for word in nltk.word_tokenize(question):
... dictnry[word.lower()]+=1
... for answer in answers:
... for word in nltk.word_tokenize(answer):
... dictnry[word.lower()]+=1
>>> word2indx = {w:(i+1) for i,(w,_) in enumerate(dictnry.most_common() )}
>>> word2indx["PAD"] = 0
>>> indx2word = {v:k for k,v in word2indx.items()}
>>> vocab_size = len(word2indx)
>>> print("vocabulary size:",len(word2indx))
```

The following screenshot depicts all the words in the vocabulary. It has only `22` words, including the `PAD` word, which has been created to fill blank spaces or zeros:

![](img/6567905e-5112-4afc-9a3e-136b5e303f04.png)

The following code is used to determine the maximum length of words. By knowing this, we can create a vector of maximum size, which can incorporate all lengths of words:

```py
# compute max sequence length for each entity
>>> story_maxlen = 0
>>> question_maxlen = 0
>>> for stories, questions, answers in [data_train,data_test]:
... for story in stories:
... story_len = 0
... for sent in story:
... swords = nltk.word_tokenize(sent)
... story_len += len(swords)
... if story_len > story_maxlen:
... story_maxlen = story_len
... for question in questions:
... question_len = len(nltk.word_tokenize(question))
... if question_len > question_maxlen:
... question_maxlen = question_len>>> print ("Story maximum length:",story_maxlen,"Question maximum length:",question_maxlen)
```

The maximum length of words for story is `14`, and for questions it is `4`. For some of the stories and questions, the length could be less than the maximum length; those words will be replaced with `0` (or `PAD` word). The reason? This padding of extra blanks will make all the observations of equal length. This is for computation efficiency, or else it will be difficult to map different lengths, or creating parallelization in GPU for computation will be impossible:

![](img/84c1ebfd-8183-4b30-b28c-f9bba117674b.png)

Following snippets of code does import various functions from respective classes which we will be using in the following section:

```py
>>> from keras.layers import Input
>>> from keras.layers.core import Activation, Dense, Dropout, Permute
>>> from keras.layers.embeddings import Embedding
>>> from keras.layers.merge import add, concatenate, dot
>>> from keras.layers.recurrent import LSTM
>>> from keras.models import Model
>>> from keras.preprocessing.sequence import pad_sequences
>>> from keras.utils import np_utils
```

Word-to-vectorized mapping is being performed in the following code after considering the maximum lengths for story, question, and so on, while also considering vocab size, all of which we have computed in the preceding segment of code:

```py
# Converting data into Vectorized form
>>> def data_vectorization(data, word2indx, story_maxlen, question_maxlen):
... Xs, Xq, Y = [], [], []
... stories, questions, answers = data
... for story, question, answer in zip(stories, questions, answers):
... xs = [[word2indx[w.lower()] for w in nltk.word_tokenize(s)]
for s in story]
... xs = list(itertools.chain.from_iterable(xs))
... xq = [word2indx[w.lower()] for w in nltk.word_tokenize (question)]
... Xs.append(xs)
... Xq.append(xq)
... Y.append(word2indx[answer.lower()])
... return pad_sequences(Xs, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen),np_utils.to_categorical(Y, num_classes= len(word2indx))
```

The application of `data_vectorization` is shown in this code:

```py
>>> Xstrain, Xqtrain, Ytrain = data_vectorization(data_train, word2indx, story_maxlen, question_maxlen)
>>> Xstest, Xqtest, Ytest = data_vectorization(data_test, word2indx, story_maxlen, question_maxlen)
>>> print("Train story",Xstrain.shape,"Train question", Xqtrain.shape,"Train answer", Ytrain.shape)
>>> print( "Test story",Xstest.shape, "Test question",Xqtest.shape, "Test answer",Ytest.shape)
```

The following image describes the dimensions of train and test data for story, question, and answer segments accordingly:

![](img/8133b229-445b-4f66-a8e7-9a239f8070d2.png)

Parameters are initialized in the following code:

```py
# Model Parameters
>>> EMBEDDING_SIZE = 128
>>> LATENT_SIZE = 64
>>> BATCH_SIZE = 64
>>> NUM_EPOCHS = 40
```

The core building blocks of the model are explained here:

```py
# Inputs
>>> story_input = Input(shape=(story_maxlen,))
>>> question_input = Input(shape=(question_maxlen,)) 
# Story encoder embedding
>>> story_encoder = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_SIZE,input_length= story_maxlen) (story_input)
>>> story_encoder = Dropout(0.2)(story_encoder) 
# Question encoder embedding
>>> question_encoder = Embedding(input_dim=vocab_size,output_dim= EMBEDDING_SIZE, input_length=question_maxlen) (question_input)
>>> question_encoder = Dropout(0.3)(question_encoder) 
# Match between story and question
>>> match = dot([story_encoder, question_encoder], axes=[2, 2]) 
# Encode story into vector space of question
>>> story_encoder_c = Embedding(input_dim=vocab_size, output_dim=question_maxlen,input_length= story_maxlen)(story_input)
>>> story_encoder_c = Dropout(0.3)(story_encoder_c) 
# Combine match and story vectors
>>> response = add([match, story_encoder_c])
>>> response = Permute((2, 1))(response) 
# Combine response and question vectors to answers space
>>> answer = concatenate([response, question_encoder], axis=-1)
>>> answer = LSTM(LATENT_SIZE)(answer)
>>> answer = Dropout(0.2)(answer)
>>> answer = Dense(vocab_size)(answer)
>>> output = Activation("softmax")(answer)
>>> model = Model(inputs=[story_input, question_input], outputs=output)
>>> model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
>>> print (model.summary())
```

By reading the model summary in following image, you can see how blocks are connected and the see total number of parameters required to be trained to tune the model:

![](img/aa0bd335-47c0-4da5-8e42-dbb492fd5af2.png)

 Following code does perform model fitting on train data:

```py
# Model Training
>>> history = model.fit([Xstrain, Xqtrain], [Ytrain], batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_data= ([Xstest, Xqtest], [Ytest]))
```

The model accuracy has significantly improved from the first iteration (*train accuracy = 19.35%* and *validation accuracy = 28.98%*) to the 40^(th) (*train accuracy = 82.22%* and *validation accuracy = 84.51%*), which can be shown in the following image:

![](img/d13ea6d1-b563-4a0d-af7b-0acd4db56e73.png)

Following code does plot both training & validation accuracy change with respective to change in epoch:

```py
# plot accuracy and loss plot
>>> plt.title("Accuracy")
>>> plt.plot(history.history["acc"], color="g", label="train")
>>> plt.plot(history.history["val_acc"], color="r", label="validation")
>>> plt.legend(loc="best")
>>> plt.show()
```

The change in accuracy with the number of iterations is shown in the following image. It seems that the accuracy has improved marginally rather than drastically after `10` iterations:

![](img/b00044e5-cedb-4f5a-8c4a-7028cec8a801.png)

In the following code, results are predicted which is finding probability for each respective class and also applying `argmax` function for finding the class where the probability is maximum:

```py
# get predictions of labels
>>> ytest = np.argmax(Ytest, axis=1)
>>> Ytest_ = model.predict([Xstest, Xqtest])
>>> ytest_ = np.argmax(Ytest_, axis=1)
# Select Random questions and predict answers
>>> NUM_DISPLAY = 10
>>> for i in random.sample(range(Xstest.shape[0]),NUM_DISPLAY):
... story = " ".join([indx2word[x] for x in Xstest[i].tolist() if x != 0])
... question = " ".join([indx2word[x] for x in Xqtest[i].tolist()])
... label = indx2word[ytest[i]]
... prediction = indx2word[ytest_[i]]
... print(story, question, label, prediction)
```

After training the model enough and achieving better accuracies on validation data such as 84.51%, it is time to verify with actual test data to see how much the predicted answers are in line with the actual answers.

Out of ten randomly drawn questions, the model was unable to predict the correct question only once (for the sixth question; the actual answer is `bedroom` and the predicted answer is `office`). This means we have got 90% accuracy  on the sample. Though we may not be able to generalize the accuracy value, this gives some idea to reader about the prediction ability of the model:

![](img/29f8f2b6-ff9e-4e9e-abb5-33338550761e.png)

# Language modeling to predict the next best word using recurrent neural networks LSTM

Predicting the next word based on some typed words has many real-word applications. An example would be to suggest the word while typing it into the Google search bar. This type of feature does improve user satisfaction in using search engines. Technically, this can be called **N-grams** (if two consecutive words are extracted, it will be called **bi-grams**). Though there are so many ways to model this, here we have chosen deep RNNs to predict the next best word based on *N-1* pre-words.

# Getting ready...

Alice in Wonderland data has been used for this purpose and the same data can be downloaded from [http://www.umich.edu/~umfandsf/other/ebooks/alice30.txt](http://www.umich.edu/~umfandsf/other/ebooks/alice30.txt). In the initial data preparation stage, we have extracted N-grams from continuous text file data, which looks like a story file:

```py
>>> from __future__ import print_function
>>> import os
""" First change the following directory link to where all input files do exist """
>>> os.chdir("C:\\Users\\prata\\Documents\\book_codes\\NLP_DL")
>>> from sklearn.model_selection import train_test_split
>>> import nltk
>>> import numpy as np
>>> import string
# File reading
>>> with open('alice_in_wonderland.txt', 'r') as content_file:
... content = content_file.read()
>>> content2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in content]).split())
>>> tokens = nltk.word_tokenize(content2)
>>> tokens = [word.lower() for word in tokens if len(word)>=2]
```

N-grams are selected with the following `N` value. In the following code, we have chosen `N` as `3`, which means each piece of data has three words consecutively. Among them, two pre-words (bi-grams) used to predict the next word in each observation. Readers are encouraged to change the value of `N` and see how the model predicts the words:

Note: With the increase in N-grams to 4, 5, and 6 or so, we need to provide enough amount of incremental data to compensate for the curse of dimensionality.

```py
# Select value of N for N grams among which N-1 are used to predict last Nth word
>>> N = 3
>>> quads = list(nltk.ngrams(tokens,N))
>>> newl_app = []
>>> for ln in quads:
... newl = " ".join(ln)
... newl_app.append(newl)
```

# How to do it...

After extracting basic data observations, we need to perform the following operations:

1.  **Preprocessing**: In the preprocessing step, words are converted to vectorized form, which is needed for working with the model.
2.  **Model development and validation**: Create a convergent-divergent model to map the input to the output, followed by training and validation data.
3.  **Prediction of next best word**: Utilize the trained model to predict the next best word on test data.

# How it works...

Vectorization of the given words (*X* and *Y* words) to vector space using `CountVectorizer` from scikit-learn:

```py
# Vectorizing the words
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> vectorizer = CountVectorizer()
>>> x_trigm = []
>>> y_trigm = []
>>> for l in newl_app:
... x_str = " ".join(l.split()[0:N-1])
... y_str = l.split()[N-1]
... x_trigm.append(x_str)
... y_trigm.append(y_str)
>>> x_trigm_check = vectorizer.fit_transform(x_trigm).todense()
>>> y_trigm_check = vectorizer.fit_transform(y_trigm).todense()
# Dictionaries from word to integer and integer to word
>>> dictnry = vectorizer.vocabulary_
>>> rev_dictnry = {v:k for k,v in dictnry.items()}
>>> X = np.array(x_trigm_check)
>>> Y = np.array(y_trigm_check)
>>> Xtrain, Xtest, Ytrain, Ytest,xtrain_tg,xtest_tg = train_test_split(X, Y,x_trigm, test_size=0.3,random_state=42)
>>> print("X Train shape",Xtrain.shape, "Y Train shape" , Ytrain.shape)
>>> print("X Test shape",Xtest.shape, "Y Test shape" , Ytest.shape)
```

After converting the data into vectorized form, we can see that the column value remains the same, which is the vocabulary length (2559 of all possible words):

![](img/35884aee-14dc-47da-ab5f-70fc2de54929.png)

The following code is the heart of the model, consisting of convergent-divergent architecture that reduces and expands the shape of the neural network:

```py
# Model Building
>>> from keras.layers import Input,Dense,Dropout
>>> from keras.models import Model
>>> np.random.seed(42)
>>> BATCH_SIZE = 128
>>> NUM_EPOCHS = 100
>>> input_layer = Input(shape = (Xtrain.shape[1],),name="input")
>>> first_layer = Dense(1000,activation='relu',name = "first")(input_layer)
>>> first_dropout = Dropout(0.5,name="firstdout")(first_layer)
>>> second_layer = Dense(800,activation='relu',name="second") (first_dropout)
>>> third_layer = Dense(1000,activation='relu',name="third") (second_layer)
>>> third_dropout = Dropout(0.5,name="thirdout")(third_layer)
>>> fourth_layer = Dense(Ytrain.shape[1],activation='softmax',name = "fourth")(third_dropout)
>>> history = Model(input_layer,fourth_layer)
>>> history.compile(optimizer = "adam",loss="categorical_crossentropy", metrics=["accuracy"])
>>> print (history.summary())
```

This screenshot depicts the complete architecture of the model, consisting of a convergent-divergent structure:

![](img/52e02c04-33fd-4dea-9ab5-ddb15528aba3.png)

```py
# Model Training
>>> history.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,epochs=NUM_EPOCHS, verbose=1,validation_split = 0.2)
```

The model is trained on data with 100 epochs. Even after a significant improvement in the train accuracy (from 5.46% to 63.18%), there is little improvement in the validation accuracy (6.63% to 10.53%). However, readers are encouraged to try various settings to improve the validation accuracy further:

![](img/b5db198b-1382-4fe9-914c-0a66d3ef6c1b.png)

```py
# Model Prediction
>>> Y_pred = history.predict(Xtest)
# Sample check on Test data
>>> print ("Prior bigram words", "|Actual", "|Predicted","\n")
>>> for i in range(10):
... print (i,xtest_tg[i],"|",rev_dictnry[np.argmax(Ytest[i])], "|",rev_dictnry[np.argmax(Y_pred[i])])
```

Less validation accuracy provides a hint that the model might not predict the word very well. The reason could be the very-high-dimensional aspect of taking the word rather than the character level (character dimensions are 26, which is much less than the 2559 value of words). In the following screenshot, we have predicted about two times out of `10`. However, it is very subjective to say whether it is a yes or no. Sometimes, the word predicted could be close but not the same:

![](img/78c49c83-741f-4a96-ab85-88af03bcfd8f.png)

# Generative chatbot using recurrent neural networks (LSTM)

Generative chatbots are very difficult to build and operate. Even today, most workable chatbots are retrieving in nature; they retrieve the best response for the given question based on semantic similarity, intent, and so on. For further reading, refer to the paper *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* by Kyunghyun Cho et. al. ([https://arxiv.org/pdf/1406.1078.pdf](https://arxiv.org/pdf/1406.1078.pdf)).

# Getting ready...

The A.L.I.C.E Artificial Intelligence Foundation dataset `bot.aiml`  **Artificial Intelligence Markup Language** (**AIML**), which is customized syntax such as XML file has been used to train the model. In this file, questions and answers are mapped. For each question, there is a particular answer. Complete `.aiml` files are available at *aiml-en-us-foundation-alice.v1-9* from [https://code.google.com/archive/p/aiml-en-us-foundation-alice/downloads](https://code.google.com/archive/p/aiml-en-us-foundation-alice/downloads). Unzip the folder to see the `bot.aiml` file and open it using Notepad. Save as `bot.txt` to read in Python:

```py
>>> import os
""" First change the following directory link to where all input files do exist """
>>> os.chdir("C:\\Users\\prata\\Documents\\book_codes\\NLP_DL")
>>> import numpy as np
>>> import pandas as pd
# File reading
>>> with open('bot.txt', 'r') as content_file:
... botdata = content_file.read()
>>> Questions = []
>>> Answers = []
```

AIML files have unique syntax, similar to XML. The `pattern` word is used to represent the question and the `template` word for the answer. Hence, we are extracting respectively:

```py
>>> for line in botdata.split("</pattern>"):
... if "<pattern>" in line:
... Quesn = line[line.find("<pattern>")+len("<pattern>"):]
... Questions.append(Quesn.lower())
>>> for line in botdata.split("</template>"):
... if "<template>" in line:
... Ans = line[line.find("<template>")+len("<template>"):]
... Ans = Ans.lower()
... Answers.append(Ans.lower())
>>> QnAdata = pd.DataFrame(np.column_stack([Questions,Answers]),columns = ["Questions","Answers"])
>>> QnAdata["QnAcomb"] = QnAdata["Questions"]+" "+QnAdata["Answers"]
>>> print(QnAdata.head())
```

The question and answers are joined to extract the total vocabulary used in the modeling, as we need to convert all words/characters into numeric representation. The reason is the same as mentioned before—deep learning models can't read English and everything is in numbers for the model.

![](img/113ecf8f-6dc5-4f66-86b2-577e574ddafd.png)

# How to do it...

After extracting the question-and-answer pairs, the following steps are needed to process the data and produce the results:

1.  **Preprocessing**: Convert the question-and-answer pairs into vectorized format, which will be utilized in model training.
2.  **Model building and validation**: Develop deep learning models and validate the data.
3.  **Prediction of answers from trained model**: The trained model will be used to predict answers for given questions.

# How it works...

The question and answers are utilized to create the vocabulary of words to index mapping, which will be utilized for converting words into vector mappings:

```py
# Creating Vocabulary
>>> import nltk
>>> import collections
>>> counter = collections.Counter()
>>> for i in range(len(QnAdata)):
... for word in nltk.word_tokenize(QnAdata.iloc[i][2]):
... counter[word]+=1
>>> word2idx = {w:(i+1) for i,(w,_) in enumerate(counter.most_common())}
>>> idx2word = {v:k for k,v in word2idx.items()}
>>> idx2word[0] = "PAD"
>>> vocab_size = len(word2idx)+1
>>> print (vocab_size)
```

![](img/cdec47c6-3e7d-49ca-88a0-b276eb4a0e61.png)

Encoding and decoding functions are used to convert text to indices and indices to text respectively. As we know, Deep learning models work on numeric values rather than text or character data:

```py
>>> def encode(sentence, maxlen,vocab_size):
... indices = np.zeros((maxlen, vocab_size))
... for i, w in enumerate(nltk.word_tokenize(sentence)):
... if i == maxlen: break
... indices[i, word2idx[w]] = 1
... return indices
>>> def decode(indices, calc_argmax=True):
... if calc_argmax:
... indices = np.argmax(indices, axis=-1)
... return ' '.join(idx2word[x] for x in indices)
```

The following code is used to vectorize the question and answers with the given maximum length for both questions and answers. Both might be different lengths. In some pieces of data, the question length is greater than answer length, and in a few cases, it's length is less than answer length. Ideally, the question length is good to catch the right answers. Unfortunately in this case, question length is much less than the answer length, which is a very bad example to develop generative models:

```py
>>> question_maxlen = 10
>>> answer_maxlen = 20
>>> def create_questions(question_maxlen,vocab_size):
... question_idx = np.zeros(shape=(len(Questions),question_maxlen, vocab_size))
... for q in range(len(Questions)):
... question = encode(Questions[q],question_maxlen,vocab_size)
... question_idx[i] = question
... return question_idx
>>> quesns_train = create_questions(question_maxlen=question_maxlen, vocab_size=vocab_size)
>>> def create_answers(answer_maxlen,vocab_size):
... answer_idx = np.zeros(shape=(len(Answers),answer_maxlen, vocab_size))
... for q in range(len(Answers)):
... answer = encode(Answers[q],answer_maxlen,vocab_size)
... answer_idx[i] = answer
... return answer_idx
>>> answs_train = create_answers(answer_maxlen=answer_maxlen,vocab_size= vocab_size)
>>> from keras.layers import Input,Dense,Dropout,Activation
>>> from keras.models import Model
>>> from keras.layers.recurrent import LSTM
>>> from keras.layers.wrappers import Bidirectional
>>> from keras.layers import RepeatVector, TimeDistributed, ActivityRegularization
```

The following code is an important part of the chatbot. Here we have used recurrent networks, repeat vector, and time-distributed networks. The repeat vector used to match dimensions of input to output values. Whereas time-distributed networks are used to change the column vector to the output dimension's vocabulary size:

```py
>>> n_hidden = 128
>>> question_layer = Input(shape=(question_maxlen,vocab_size))
>>> encoder_rnn = LSTM(n_hidden,dropout=0.2,recurrent_dropout=0.2) (question_layer)
>>> repeat_encode = RepeatVector(answer_maxlen)(encoder_rnn)
>>> dense_layer = TimeDistributed(Dense(vocab_size))(repeat_encode)
>>> regularized_layer = ActivityRegularization(l2=1)(dense_layer)
>>> softmax_layer = Activation('softmax')(regularized_layer)
>>> model = Model([question_layer],[softmax_layer])
>>> model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
>>> print (model.summary())
```

The following model summary describes the change in flow of model size across the model. The input layer matches the question's dimension and the output matches the answer's dimension:

![](img/a06e5d6b-d4e4-4ae4-ad8e-c214afa531bc.png)

```py
# Model Training
>>> quesns_train_2 = quesns_train.astype('float32')
>>> answs_train_2 = answs_train.astype('float32')
>>> model.fit(quesns_train_2, answs_train_2,batch_size=32,epochs=30, validation_split=0.05)
```

The results are a bit tricky in the following screenshot even though the accuracy is significantly higher. The chatbot model might produce complete nonsense, as most of the words are padding here. The reason? The number of words in this data is less:

>![](img/fb81b2d5-3b8b-41ce-910c-25ee240b59be.png)

```py
# Model prediction
>>> ans_pred = model.predict(quesns_train_2[0:3])
>>> print (decode(ans_pred[0]))
>>> print (decode(ans_pred[1]))
```

The following screenshot depicts the sample output on test data. The output does not seem to make sense, which is an issue with generative models:

![](img/cc37d6c6-52a3-4209-ab28-5c9323203e98.png)

Our model did not work well in this case, but still some areas of improvement are possible going forward with generative chatbot models. Readers can give it a try:

*   Have a dataset with lengthy questions and answers to catch signals well
*   Create a larger architecture of deep learning models and train over longer iterations
*   Make question-and-answer pairs more generic rather than factoid-based, such as retrieving knowledge and so on, where generative models fail miserably