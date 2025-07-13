# Applications of Deep Learning in NLP

In this chapter, we will cover the following recipes:

*   Classification of emails using deep neural networks after generating TF-IDF
*   IMDB sentiment classification using convolutional networks CNN 1D
*   IMDB sentiment classification using bidirectional LSTM
*   Visualization of high-dimensional words in 2D with neural word vector visualization

# Introduction

In recent times, deep learning has become very prominent in the application of text, voice, and image data to obtain state-of-the-art results, which are primarily used in the creation of applications in the field of artificial intelligence. However, these models turn out to be producing such results in all the fields of application. In this chapter, we will be covering various applications in NLP/text processing.

Convolutional neural networks and recurrent neural networks are central themes in deep learning that you will keep meeting across the domain.

# Convolutional neural networks

CNNs are primarily used in image processing to classify images into a fixed set of categories and so on. CNN's working principle has been described in the following diagram, wherein a filter of size 3 x 3 convolves over the original matrix of size 5 x 5, which produces an output of size 3 x 3\. The filter can stride horizontally by a step size of 1 or any value greater than 1 also. For cell (1, 1) the value obtained is 3, which is a product of the underlying matrix value and filter values. In this way, the filter will hover across the original 5 x 5 matrix to create convolved features of 3 x 3, also known as activation maps:

![](img/1f0de83f-8f28-4815-8546-e0291f2381ed.png)

The advantages of using convolutions:

*   Instead of a fixed size, fully connected layers save the number of neurons and hence the computational power requirement of the machine.
*   Only a small size of filter weights is used to hover across the matrix, rather than each pixel connected to the next layers. So this is a better way of summarization of the input image into the next layers.
*   During backpropagation, only the weights of the filter need to be updated based on the backpropagated errors, hence the higher efficiency.

CNNs perform mappings between spatially/temporally distributed arrays in arbitrary dimensions. They appear to be suitable for application to time series, images, or videos. CNNs are characterized by:

*   Translation invariance (neural weights are fixed with respect to spatial translation)
*   Local connectivity (neural connections exist only between spatially local regions)
*   An optional progressive decrease in spatial resolution (as the number of features is gradually increased)

After convolution, the convolved feature/activation map needs to be reduced based on the most important features, as the same operation reduces the number of points and improves computational efficiency. Pooling is an operation typically performed to reduce unnecessary representations. Brief details about pooling operations are given as follows:

*   **Pooling**: Pooling makes the activation representation (obtained from convolving the filter over the input combination of input and weight values) smaller and more manageable. It operates over each activation map independently. Pooling applies to the width and breadth of the layer, and the depth will remain the same during the pooling stage. In the following diagram, a pooling operation of 2 x 2 is explained. Every original 4 x 4 matrix has been reduced by half. In the first four cell values of 2, 4, 5, and 8, the maximum is extracted, which is 8:

![](img/3170d0b6-f394-4a6e-892a-e4cfa2098e20.png)

Due to the operation of convolution, it is natural that the size of pixels/input data size reduces over the stages. But in some cases, we would really like to maintain the size across operations. A hacky way to achieve this is padding with zeros at the top layer accordingly.

*   **Padding**: The following diagram (its width and breadth) will be shrunk consecutively; this is undesirable in deep networks, and padding keeps the size of the picture constant or controllable in size throughout the network.

![](img/06710dbf-06c0-4d56-83ef-24aa6b2c386b.png)

A simple equation for calculating the activation map size based on given input width, filter size, padding, and stride is shown as follows. This equation gives an idea of how much computational power is needed, and so on.

*   **Calculation of activation map size**: In the following formula, the size of the activation map obtained from the convolutional layer is:

![](img/ed1eab2b-75be-4e81-9f68-bebb519fda72.png)

Where, *W* is the width of original image, *F* is the filter size, *P* is padding size (*1* for a single layer of padding, *2* for a double layer of padding, and so on), *S* is stride length

For example, consider an input image of size 224 x 224 x 3 (3 indicates Red, Green, and Blue channels), with a filter size of 11 x 11 and number of filters as 96\. The stride length is 4 and there is no padding. What is the activation map size generated out from these filters?

![](img/6267bd0f-60a3-4992-914d-1a5380ac7ce1.png)![](img/184c95da-33a8-4650-b17d-def9e96dd982.png)

The activation map dimensions would be 55 x 55 x 96\. Using the preceding formula, only width and depth can be computed, but the depth depends on the number of filters used. In fact, this is what was obtained in step 1 after convolution stage in AlexNet, which we will describe now.

*   **AlexNet used in ImageNet competition during 2012**: The following image describes AlexNet, developed to win the ImageNet competition during 2012\. It produced significantly more accuracy compared with other competitors.

![](img/28456fad-7aea-4783-949f-e3bd0ec3dc14.png)

In AlexNet, all techniques such as convolution, pooling, and padding have been used, and finally get connected with the fully connected layer.

# Applications of CNNs

CNNs are used in various applications, a few of them are as follows:

*   **Image classification**: Compared with other methods, CNNs achieve higher accuracy on large-scale images of data size. In image classification, CNNs are used at the initial stage, and once enough features are extracted using pooling layers, followed by other CNNs and so on, will be finally connected with the fully connected layers to classify them into the number of given classes.
*   **Face recognition**: CNNs are invariant to position, brightness, and so on, which will recognize faces from images and process them despite bad lighting, a face looking sideways, and so on.
*   **Scene labeling**: Each pixel is labeled with the category of the object it belongs to in scene labeling. CNNs are utilized here to combine pixels in a hierarchical manner.
*   **NLP**: In NLP, CNNs are used similarly with bag-of-words, in which the sequence of words does not play a critical role in identifying the final class of email/text and so on. CNNs are used on matrices, which are represented by sentences in vector format. Subsequently, filters are applied but CNNs are one-dimensional, in which width is constant, and filters traverse only across height (the height is 2 for bi-grams, 3 for tri-grams, and so on).

# Recurrent neural networks

A recurrent neural network is used to process a sequence of vectors X by applying a recurrence formula at every time step. In convolutional neural networks, we assume all inputs are independent of each other. But in some tasks, inputs are dependent on each other, for example, time series forecasting data, or predicting the next word in a sentence depending on past words, and so on, which needs to be modeled by considering dependency of past sequences. These types of problems are modeled with RNNs as they provide better accuracy, In theory, RNNs can make use of information in arbitrarily long sequences, but in practice, they are limited to looking back only for a few steps. The next formula explains the RNN functionality:

![](img/6c951b2a-f3cd-4761-895c-174e48adebef.png)

![](img/3d3b30c1-c0b3-4000-91be-d2499713d1c4.png)

![](img/ebc6d56d-0880-4b8f-ba97-517c5726551c.png)

![](img/dbcf9f07-3081-48ae-b330-7d04b13bd121.png)

![](img/db890bd1-2148-4e01-96c4-2cb3ca05aad3.png)

![](img/d39ddee4-9fdf-44bd-8c93-072018b21b35.png)

![](img/5a7099c1-83bd-4c00-a499-c47d03cbc953.png)

*   **Vanishing or exploding the gradient problem in RNNs**: Gradients does vanish quickly with the more number of layers and this issue is severe with RNNs as at each layer there are many time steps which also do occur and recurrent weights are multiplicative in nature, hence gradients either explode or vanish quickly, which makes neural networks untrainable. Exploding gradients can be limited by using a gradient clipping technique, in which an upper limit will be set to explode the gradients, but however vanishing gradient problem still does exists. This issue can be overcome by using **long short-term memory** (**LSTM**) networks.
*   **LSTM**: LSTM is an artificial neural network contains LSTM blocks in addition to regular network units. LSTM blocks contain gates that determine when the input is significant enough to remember, when it should continue to remember or when it should forget the value, and when it should output the value.

![](img/71294f9c-1aff-473a-8966-8cb7616da3d4.png)![](img/b706152e-fc50-47a8-b1df-d695e7c5ab26.png)

Vanishing and exploding gradient problems do not occur in LSTM as the same is an additive model rather than multiplicative model which is the case with RNN.

# Application of RNNs in NLP

RNNs have shown great success in many NLP tasks. The most commonly used variant of RNN is LSTM due to overcoming the issue of vanishing/exploding gradients.

*   **Language modeling**: Given a sequence of words, the task is to predict the next probable word
*   **Text generation**: To generate text from the writings of some authors
*   **Machine translation**: To convert one language into other language (English to Chinese and so on.)
*   **Chat bot**: This application is very much like machine translation; however question and answer pairs are used to train the model
*   **Generating an image description**: By training together with CNNs, RNNs can be used to generate a caption/description of the image

# Classification of emails using deep neural networks after generating TF-IDF

In this recipe, we will use deep neural networks to classify emails into one of the 20 pre-trained categories based on the words present in each email. This is the simple model to start with to understand the subject of deep learning and its applications on NLP.

# Getting ready

The 20 newsgroups dataset from scikit-learn have been utilized to illustrate the concept. Number of observations/emails considered for analysis are 18,846 (train observations - 11,314 and test observations - 7,532) and its corresponding classes/categories are 20, which are shown in the following:

```py
>>> from sklearn.datasets import fetch_20newsgroups
>>> newsgroups_train = fetch_20newsgroups(subset='train')
>>> newsgroups_test = fetch_20newsgroups(subset='test')
>>> x_train = newsgroups_train.data
>>> x_test = newsgroups_test.data
>>> y_train = newsgroups_train.target
>>> y_test = newsgroups_test.target
>>> print ("List of all 20 categories:")
>>> print (newsgroups_train.target_names)
>>> print ("\n")
>>> print ("Sample Email:")
>>> print (x_train[0])
>>> print ("Sample Target Category:")
>>> print (y_train[0])
>>> print (newsgroups_train.target_names[y_train[0]])
```

In the following screenshot, a sample first data observation and target class category has been shown. From the first observation or email we can infer that the email is talking about a two-door sports car, which we can classify manually into autos category which is `8`.

Target value is `7` due to the indexing starts from `0`), which is validating our understanding with actual target class `7`

![](img/1afe5527-7747-4d8e-9cce-291ed16337a9.png)

# How to do it...

Using NLP techniques, we have pre-processed the data for obtaining finalized word vectors to map with final outcomes spam or ham. Major steps involved are:

1.  Pre-processing.
2.  Removal of punctuations.
3.  Word tokenization.
4.  Converting words into lowercase.
5.  Stop word removal.
6.  Keeping words of length of at least 3.
7.  Stemming words.
8.  POS tagging.
9.  Lemmatization of words:
    1.  TF-IDF vector conversion.
    2.  Deep learning model training and testing.
    3.  Model evaluation and results discussion.

# How it works...

The NLTK package has been utilized for all the pre-processing steps, as it consists of all the necessary NLP functionality under one single roof:

```py
# Used for pre-processing data
>>> import nltk
>>> from nltk.corpus import stopwords
>>> from nltk.stem import WordNetLemmatizer
>>> import string
>>> import pandas as pd
>>> from nltk import pos_tag
>>> from nltk.stem import PorterStemmer
```

The function written (pre-processing) consists of all the steps for convenience. However, we will be explaining all the steps in each section:

```py
>>> def preprocessing(text): 
```

The following line of the code splits the word and checks each character to see if it contains any standard punctuations, if so it will be replaced with a blank or else it just don't replace with blank:

```py
...     text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split()) 
```

The following code tokenizes the sentences into words based on whitespaces and puts them together as a list for applying further steps:

```py
...     tokens = [word for sent in nltk.sent_tokenize(text2) for word in nltk.word_tokenize(sent)] 

```

Converting all the cases (upper, lower and proper) into lower case reduces duplicates in corpus:

```py
...     tokens = [word.lower() for word in tokens]
```

As mentioned earlier, Stop words are the words that do not carry much of weight in understanding the sentence; they are used for connecting words and so on. We have removed them with the following line of code:

```py

...     stopwds = stopwords.words('english') 
...     tokens = [token for token in tokens if token not in stopwds]
```

Keeping only the words with length greater than `3` in the following code for removing small words which hardly consists of much of a meaning to carry;

```py
...     tokens = [word for word in tokens if len(word)>=3] 
```

Stemming applied on the words using Porter stemmer which stems the extra suffixes from the words:

```py
...     stemmer = PorterStemmer() 
...     tokens = [stemmer.stem(word) for word in tokens]  
```

POS tagging is a prerequisite for lemmatization, based on whether word is noun or verb or and so on. it will reduce it to the root word

```py
...     tagged_corpus = pos_tag(tokens)     
```

`pos_tag` function returns the part of speed in four formats for Noun and six formats for verb. NN - (noun, common, singular), NNP - (noun, proper, singular), NNPS - (noun, proper, plural), NNS - (noun, common, plural), VB - (verb, base form), VBD - (verb, past tense), VBG - (verb, present participle), VBN - (verb, past participle), VBP - (verb, present tense, not 3rd person singular), VBZ - (verb, present tense, third person singular)

```py
...     Noun_tags = ['NN','NNP','NNPS','NNS'] 
...    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ'] 
...     lemmatizer = WordNetLemmatizer()  
```

The following function, `prat_lemmatize`, has been created only for the reasons of mismatch between the `pos_tag` function and intake values of `lemmatize` function. If the tag for any word falls under the respective noun or verb tags category, `n` or `v` will be applied accordingly in `lemmatize` function:

```py
...     def prat_lemmatize(token,tag): 
...       if tag in Noun_tags: 
...         return lemmatizer.lemmatize(token,'n') 
...       elif tag in Verb_tags: 
...         return lemmatizer.lemmatize(token,'v') 
...       else: 
...         return lemmatizer.lemmatize(token,'n')
```

After performing tokenization and applied all the various operations, we need to join it back to form stings and the following function performs the same:

```py
...     pre_proc_text =  " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])              
...     return pre_proc_text 
```

Applying pre-processing on train and test data:

```py
>>> x_train_preprocessed = []
>>> for i in x_train:
... x_train_preprocessed.append(preprocessing(i))
>>> x_test_preprocessed = []
>>> for i in x_test:
... x_test_preprocessed.append(preprocessing(i))
# building TFIDF vectorizer
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', max_features= 10000,strip_accents='unicode', norm='l2')
>>> x_train_2 = vectorizer.fit_transform(x_train_preprocessed).todense()
>>> x_test_2 = vectorizer.transform(x_test_preprocessed).todense()
```

After the pre-processing step has been completed, processed TF-IDF vectors have to be sent to the following deep learning code:

```py
# Deep Learning modules
>>> import numpy as np
>>> from keras.models import Sequential
>>> from keras.layers.core import Dense, Dropout, Activation
>>> from keras.optimizers import Adadelta,Adam,RMSprop
>>> from keras.utils import np_utils
```

The following image produces the output after firing up the preceding Keras code. Keras has been installed on Theano, which eventually works on Python. A GPU with 6 GB memory has been installed with additional libraries (CuDNN and CNMeM) for four to five times faster execution, with a choking of around 20% memory; hence only 80% memory out of 6 GB is available;

![](img/32236634-34b3-4191-891e-e41530804966.png)

The following code explains the central part of the deep learning model. The code is self-explanatory, with the number of classes considered `20`, batch size `64`, and number of epochs to train, `20`:

```py
# Definition hyper parameters
>>> np.random.seed(1337)
>>> nb_classes = 20
>>> batch_size = 64
>>> nb_epochs = 20
```

The following code converts the `20` categories into one-hot encoding vectors in which `20` columns are created and the values against the respective classes are given as `1`. All other classes are given as `0`:

```py
>>> Y_train = np_utils.to_categorical(y_train, nb_classes)
```

In the following building blocks of Keras code, three hidden layers (`1000`, `500`, and `50` neurons in each layer respectively) are used, with dropout as 50% for each layer with Adam as an optimizer:

```py
#Deep Layer Model building in Keras
#del model
>>> model = Sequential()
>>> model.add(Dense(1000,input_shape= (10000,)))
>>> model.add(Activation('relu'))
>>> model.add(Dropout(0.5))
>>> model.add(Dense(500))
>>> model.add(Activation('relu'))
>>> model.add(Dropout(0.5))
>>> model.add(Dense(50))
>>> model.add(Activation('relu'))
>>> model.add(Dropout(0.5))
>>> model.add(Dense(nb_classes))
>>> model.add(Activation('softmax'))
>>> model.compile(loss='categorical_crossentropy', optimizer='adam')
>>> print (model.summary())
```

The architecture is shown as follows and describes the flow of the data from a start of 10,000 as input. Then there are `1000`, `500`, `50`, and `20` neurons to classify the given email into one of the `20` categories:

![](img/665d79f4-cae5-4110-a960-1eccd8cd3b6f.png)

The model is trained as per the given metrics:

```py
# Model Training
>>> model.fit(x_train_2, Y_train, batch_size=batch_size, epochs=nb_epochs,verbose=1)
```

The model has been fitted with 20 epochs, in which each epoch took about 2 seconds. The loss has been minimized from `1.9281` to `0.0241`. By using CPU hardware, the time required for training each epoch may increase as a GPU massively parallelizes the computation with thousands of threads/cores:

![](img/0566ec9a-9f67-4733-b22d-a199c949fecb.png)

Finally, predictions are made on the train and test datasets to determine the accuracy, precision, and recall values:

```py
#Model Prediction
>>> y_train_predclass = model.predict_classes(x_train_2,batch_size=batch_size)
>>> y_test_predclass = model.predict_classes(x_test_2,batch_size=batch_size)
>>> from sklearn.metrics import accuracy_score,classification_report
>>> print ("\n\nDeep Neural Network - Train accuracy:"),(round(accuracy_score( y_train, y_train_predclass),3))
>>> print ("\nDeep Neural Network - Test accuracy:"),(round(accuracy_score( y_test,y_test_predclass),3))
>>> print ("\nDeep Neural Network - Train Classification Report")
>>> print (classification_report(y_train,y_train_predclass))
>>> print ("\nDeep Neural Network - Test Classification Report")
>>> print (classification_report(y_test,y_test_predclass))
```

![](img/072cd7f7-a8ef-4989-a480-3b3a80a27faa.png)

It appears that the classifier is giving a good 99.9% accuracy on the train dataset and 80.7% on the test dataset.

# IMDB sentiment classification using convolutional networks CNN 1D

In this recipe, we will use the Keras IMDB movie review sentiment data, which has labeled its sentiment (positive/negative). Reviews are pre-processed, and each review is already encoded as a sequence of word indexes (integers). However, we have decoded it to show a you sample in the following code.

# Getting ready

The IMDB dataset from Keras has a set of words and its respective sentiment. The following is the pre-processing of the data:

```py
>>> import pandas as pd
>>> from keras.preprocessing import sequence
>>> from keras.models import Sequential
>>> from keras.layers import Dense, Dropout, Activation
>>> from keras.layers import Embedding
>>> from keras.layers import Conv1D, GlobalMaxPooling1D
>>> from keras.datasets import imdb
>>> from sklearn.metrics import accuracy_score,classification_report
```

In this set of parameters, we did put maximum features or number of words to be extracted are 6,000 with maximum length of an individual sentence as 400 words:

```py
# set parameters:
>>> max_features = 6000
>>> max_length = 400
>>> (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
>>> print(len(x_train), 'train observations')
>>> print(len(x_test), 'test observations')
```

The dataset has an equal number of train and test observations, in which we will build a model on 25,000 observations and test the trained model on the test data with 25,000 data observations. A sample of data can be seen in this screenshot:

![](img/f07abcac-0aa8-4375-8d77-43dc55726312.png)

The following code is used to create the dictionary mapping of a word and its respective integer index value:

```py
# Creating numbers to word mapping
>>> wind = imdb.get_word_index()
>>> revind = dict((v,k) for k,v in wind.iteritems())
>>> print (x_train[0])
>>> print (y_train[0])
```

We see the first observation as a set of numbers rather than any English word, because the computer can only understand and work with numbers rather than characters, words, and so on:

![](img/75264b5f-29a6-4688-a678-f98f27f46e03.png)

Decoding using a created dictionary of inverse mapping is shown here:

```py
>>> def decode(sent_list):
... new_words = []
... for i in sent_list:
... new_words.append(revind[i])
... comb_words = " ".join(new_words)
... return comb_words
>>> print (decode(x_train[0]))
```

The following screenshot describes the stage after converting a number mapping into textual format. Here, dictionaries are utilized to reverse a map from integer format to text format:

![](img/f4e42ef6-d95d-464b-b523-d4c8cba3a207.png)

# How to do it...

The major steps involved are described as follows:

1.  Pre-processing, during this stage, we do pad sequences to bring all observations into one fixed dimension, which enhances speed and enables computation.
2.  CNN 1D model development and validation.
3.  Model evaluation.

# How it works...

The following code does perform padding operation for adding extra sentences which can make up to maximum length of 400 words. By doing this, data will become even and easier to perform computation using neural networks:

```py
#Pad sequences for computational efficiency
>>> x_train = sequence.pad_sequences(x_train, maxlen=max_length)
>>> x_test = sequence.pad_sequences(x_test, maxlen=max_length)
>>> print('x_train shape:', x_train.shape)
>>> print('x_test shape:', x_test.shape)
```

![](img/70c9e3df-a73e-4654-8ead-dab9d254c9eb.png)

The following deep learning code describes the application of Keras code to create a CNN 1D model:

```py
# Deep Learning architecture parameters
>>> batch_size = 32
>>> embedding_dims = 60
>>> num_kernels = 260
>>> kernel_size = 3
>>> hidden_dims = 300
>>> epochs = 3
# Building the model
>>> model = Sequential()
>>> model.add(Embedding(max_features,embedding_dims, input_length= max_length))
>>> model.add(Dropout(0.2))
>>> model.add(Conv1D(num_kernels,kernel_size, padding='valid', activation='relu', strides=1))
>>> model.add(GlobalMaxPooling1D())
>>> model.add(Dense(hidden_dims))
>>> model.add(Dropout(0.5))
>>> model.add(Activation('relu'))
>>> model.add(Dense(1))
>>> model.add(Activation('sigmoid'))
>>> model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
>>> print (model.summary())
```

In the following screenshot, the entire model summary has been displayed, indicating the number of dimensions and its respective number of neurons utilized. These directly impact the number of parameters that will be utilized in computation from input data into the final target variable, whether it is `0` or `1`. Hence a dense layer has been utilized at the last layer of the network:

![](img/abdfbfd6-1fb3-465f-ad18-578f32ee6107.png)

The following code performs model fitting operation on training data in which both `X` and `Y` variables are used to train data by batch wise: 

```py
>>> model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs, validation_split=0.2)
```

The model has been trained for three epochs, in which each epoch consumes 5 seconds on the GPU. But if we observe the following iterations, even though the train accuracy is moving up, validation accuracy is decreasing. This phenomenon can be identified as model overfitting. This indicates that we need to try some other ways to improve the model accuracy rather than just increase the number of epochs. Other ways we probably should look at are increasing the architecture size and so on. Readers are encouraged to experiment with various combinations.

![](img/3c7a82dc-aec0-440f-9207-4bbf0a3b3420.png)

The following code is used for prediction of classes for both train and test data:

```py
#Model Prediction
>>> y_train_predclass = model.predict_classes(x_train,batch_size=batch_size)
>>> y_test_predclass = model.predict_classes(x_test,batch_size=batch_size)
>>> y_train_predclass.shape = y_train.shape
>>> y_test_predclass.shape = y_test.shape

# Model accuracies and metrics calculation
>>> print (("\n\nCNN 1D - Train accuracy:"),(round(accuracy_score(y_train, y_train_predclass),3)))
>>> print ("\nCNN 1D of Training data\n",classification_report(y_train, y_train_predclass))
>>> print ("\nCNN 1D - Train Confusion Matrix\n\n",pd.crosstab(y_train, y_train_predclass,rownames = ["Actuall"],colnames = ["Predicted"]))
>>> print (("\nCNN 1D - Test accuracy:"),(round(accuracy_score(y_test, y_test_predclass),3)))
>>> print ("\nCNN 1D of Test data\n",classification_report(y_test, y_test_predclass))
>>> print ("\nCNN 1D - Test Confusion Matrix\n\n",pd.crosstab(y_test, y_test_predclass,rownames = ["Actuall"],colnames = ["Predicted"]))
```

The following screenshot describes various measurable metrics to judge the model performance. From the result, the train accuracy seems significantly high at 96%; however, the test accuracy is at a somewhat lower value of 88.2 %. This could be due to model overfitting:

![](img/89ab2d00-affb-4a62-8f93-a38cde3ef341.png)

# IMDB sentiment classification using bidirectional LSTM

In this recipe, we are using same IMDB sentiment data to show the difference between CNN and RNN methodology in terms of accuracies and so on. Data pre-processing steps remain the same; only the architecture of the model varies.

# Getting ready

The IMDB dataset from Keras has set of words and its respective sentiment. Here is the pre-processing of the data:

```py
>>> from __future__ import print_function
>>> import numpy as np
>>> import pandas as pd
>>> from keras.preprocessing import sequence
>>> from keras.models import Sequential
>>> from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
>>> from keras.datasets import imdb
>>> from sklearn.metrics import accuracy_score,classification_report

# Max features are limited
>>> max_features = 15000
>>> max_len = 300
>>> batch_size = 64

# Loading data
>>> (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
>>> print(len(x_train), 'train observations')
>>> print(len(x_test), 'test observations')
```

# How to do it...

The major steps involved are described as follows:

1.  Pre-processing, during this stage, we do pad sequences to bring all the observations into one fixed dimension, which enhances speed and enables computation.
2.  LSTM model development and validation.
3.  Model evaluation.

# How it works...

```py
# Pad sequences for computational efficiently
>>> x_train_2 = sequence.pad_sequences(x_train, maxlen=max_len)
>>> x_test_2 = sequence.pad_sequences(x_test, maxlen=max_len)
>>> print('x_train shape:', x_train_2.shape)
>>> print('x_test shape:', x_test_2.shape)
>>> y_train = np.array(y_train)
>>> y_test = np.array(y_test)
```

The following deep learning code describes the application of Keras code to create a bidirectional LSTM model:

Bidirectional LSTMs have a connection from both forward and backward, which enables them to fill in the middle words to get connected well with left and right words:

```py
# Model Building
>>> model = Sequential()
>>> model.add(Embedding(max_features, 128, input_length=max_len))
>>> model.add(Bidirectional(LSTM(64)))
>>> model.add(Dropout(0.5))
>>> model.add(Dense(1, activation='sigmoid'))
>>> model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
# Print model architecture
>>> print (model.summary())
```

Here is the architecture of the model. The embedding layer has been used to reduce the dimensions to `128`, followed by bidirectional LSTM, ending up with a dense layer for modeling sentiment either zero or one:

![](img/72079ae1-a362-4078-8f09-a01ad6a0affa.png)

The following code is used for training the data:

```py
#Train the model
>>> model.fit(x_train_2, y_train,batch_size=batch_size,epochs=4, validation_split=0.2)
```

LSTM models take longer than CNNs because LSTMs are not easily parallelizable with GPU (4x to 5x), whereas CNNs (100x) are massively parallelizable. One important observation: even after an increase in the training accuracy, the validation accuracy was decreasing. This situation indicates overfitting.

![](img/9887f540-5eb9-4a05-95e8-09a1df5f0d30.png)

The following code has been used for predicting the class for both train and test data:

```py
#Model Prediction
>>> y_train_predclass = model.predict_classes(x_train_2,batch_size=1000)
>>> y_test_predclass = model.predict_classes(x_test_2,batch_size=1000)
>>> y_train_predclass.shape = y_train.shape
>>> y_test_predclass.shape = y_test.shape

# Model accuracies and metrics calculation
>>> print (("\n\nLSTM Bidirectional Sentiment Classification - Train accuracy:"),(round(accuracy_score(y_train,y_train_predclass),3)))
>>> print ("\nLSTM Bidirectional Sentiment Classification of Training data\n",classification_report(y_train, y_train_predclass))
>>> print ("\nLSTM Bidirectional Sentiment Classification - Train Confusion Matrix\n\n",pd.crosstab(y_train, y_train_predclass,rownames = ["Actuall"],colnames = ["Predicted"]))
>>> print (("\nLSTM Bidirectional Sentiment Classification - Test accuracy:"),(round(accuracy_score(y_test,y_test_predclass),3)))
>>> print ("\nLSTM Bidirectional Sentiment Classification of Test data\n",classification_report(y_test, y_test_predclass))
>>> print ("\nLSTM Bidirectional Sentiment Classification - Test Confusion Matrix\n\n",pd.crosstab(y_test, y_test_predclass,rownames = ["Actuall"],colnames = ["Predicted"]))
```

![](img/89f3a771-4199-4c30-9edb-5f571e93abb6.png)

It appears that LSTM did provide slightly less test accuracy compared with CNN; however, with careful tuning of the model parameters, we can obtain better accuracies in RNNs compared with CNNs.

# Visualization of high-dimensional words in 2D with neural word vector visualization

In this recipe, we will use deep neural networks to visualize words from a high-dimensional space in a two-dimensional space.

# Getting ready

The *Alice in Wonderland* dataset has been used to extract words and create a visualization using the dense network made to be like the encoder-decoder architecture:

```py
>>> from __future__ import print_function
>>> import os
""" First change the following directory link to where all input files do exist """
>>> os.chdir("C:\\Users\\prata\\Documents\\book_codes\\NLP_DL")
>>> import nltk
>>> from nltk.corpus import stopwords
>>> from nltk.stem import WordNetLemmatizer
>>> from nltk import pos_tag
>>> from nltk.stem import PorterStemmer
>>> import string
>>> import numpy as np
>>> import pandas as pd
>>> import random
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.preprocessing import OneHotEncoder
>>> import matplotlib.pyplot as plt
>>> def preprocessing(text):
... text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
... tokens = [word for sent in nltk.sent_tokenize(text2) for word in
nltk.word_tokenize(sent)]
... tokens = [word.lower() for word in tokens]
... stopwds = stopwords.words('english')
... tokens = [token for token in tokens if token not in stopwds]
... tokens = [word for word in tokens if len(word)>=3]
... stemmer = PorterStemmer()
... tokens = [stemmer.stem(word) for word in tokens]
... tagged_corpus = pos_tag(tokens)
... Noun_tags = ['NN','NNP','NNPS','NNS']
... Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']
... lemmatizer = WordNetLemmatizer()
... def prat_lemmatize(token,tag):
... if tag in Noun_tags:
... return lemmatizer.lemmatize(token,'n')
... elif tag in Verb_tags:
... return lemmatizer.lemmatize(token,'v')
... else:
... return lemmatizer.lemmatize(token,'n')
... pre_proc_text = " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])
... return pre_proc_text
>>> lines = []
>>> fin = open("alice_in_wonderland.txt", "rb")
>>> for line in fin:
... line = line.strip().decode("ascii", "ignore").encode("utf-8")
... if len(line) == 0:
... continue
... lines.append(preprocessing(line))
>>> fin.close()
```

# How to do it...

The major steps involved are described here:

*   Pre-processing, creation of skip-grams and using the middle word to predict either the left or the right word.
*   Application of one-hot encoding for feature engineering.
*   Model building using encoder-decoder architecture.
*   Extraction of the encoder architecture to create two-dimensional features for visualization from test data.

# How it works...

The following code creates dictionary, which is a mapping of word to index and index to word (vice versa). As we knew, models simply do not work on character/word input. Hence, we will be converting words into numeric equivalents (particularly integer mapping), and once the computation has been performed using the neural network model, the reverse of the mapping (index to word) will be applied to visualize them. The counter from the `collections` library has been used for efficient creation of dictionaries:

```py
>>> import collections
>>> counter = collections.Counter()
>>> for line in lines:
... for word in nltk.word_tokenize(line):
... counter[word.lower()]+=1
>>> word2idx = {w:(i+1) for i,(w,_) in enumerate(counter.most_common())}
>>> idx2word = {v:k for k,v in word2idx.items()}
```

The following code applies word-to-integer mapping and extracts the tri-grams from the embedding. Skip-gram is the methodology in which the central word is connected to both left and right adjacent words for training, and if during testing phase if it predicts correctly:

```py
>>> xs = []
>>> ys = []
>>> for line in lines:
... embedding = [word2idx[w.lower()] for w in nltk.word_tokenize(line)]
... triples = list(nltk.trigrams(embedding))
... w_lefts = [x[0] for x in triples]
... w_centers = [x[1] for x in triples]
... w_rights = [x[2] for x in triples]
... xs.extend(w_centers)
... ys.extend(w_lefts)
... xs.extend(w_centers)
... ys.extend(w_rights)
```

The following code describes that the length of the dictionary is the vocabulary size. Nonetheless, based on user specification, any custom vocabulary size can be chosen. Here, we are considering all words though!

```py
>>> print (len(word2idx))
>>> vocab_size = len(word2idx)+1
```

Based on vocabulary size, all independent and dependent variables are transformed into vector representations with the following code, in which the number of rows would be the number of words and the number of columns would be the vocabulary size. The neural network model basically maps the input and output variables over the vector space:

```py
>>> ohe = OneHotEncoder(n_values=vocab_size)
>>> X = ohe.fit_transform(np.array(xs).reshape(-1, 1)).todense()
>>> Y = ohe.fit_transform(np.array(ys).reshape(-1, 1)).todense()
>>> Xtrain, Xtest, Ytrain, Ytest,xstr,xsts = train_test_split(X, Y,xs, test_size=0.3, random_state=42)
>>> print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
```

Out of total 13,868 observations, train and test are split into 70% and 30%, which are created as 9,707 and 4,161 respectively:

![](img/e9c7b361-7274-4cdb-ba36-da1b08448548.png)

The central part of the model is described in the following few lines of deep learning code using Keras software. It is a convergent-divergent code, in which initially the dimensions of all input words are squeezed to achieve the output format.

While doing so, the dimensions are reduced to 2D in the second layer. After training the model, we will extract up to the second layer for predictions on test data. This literally works similar to the conventional encoder-decoder architecture:

```py
>>> from keras.layers import Input,Dense,Dropout
>>> from keras.models import Model
>>> np.random.seed(42)
>>> BATCH_SIZE = 128
>>> NUM_EPOCHS = 20
>>> input_layer = Input(shape = (Xtrain.shape[1],),name="input")
>>> first_layer = Dense(300,activation='relu',name = "first")(input_layer)
>>> first_dropout = Dropout(0.5,name="firstdout")(first_layer)
>>> second_layer = Dense(2,activation='relu',name="second") (first_dropout)
>>> third_layer = Dense(300,activation='relu',name="third") (second_layer)
>>> third_dropout = Dropout(0.5,name="thirdout")(third_layer)
>>> fourth_layer = Dense(Ytrain.shape[1],activation='softmax',name = "fourth")(third_dropout)
>>> history = Model(input_layer,fourth_layer)
>>> history.compile(optimizer = "rmsprop",loss= "categorical_crossentropy", metrics=["accuracy"])
```

The following code is used to train the model:

```py
>>> history.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,epochs=NUM_EPOCHS, verbose=1,validation_split = 0.2)
```

By carefully observing the accuracy on both the training and validation datasets, we can find that the best accuracy values are not even crossing 6%. This happens due to limited data and architecture of deep learning models. In order to make this really work, we need at least gigabytes of data and large architectures. Models too need to be trained for very long. Due to practical constraints and illustration purposes, we have just trained for 20 iterations. However, readers are encouraged to try various combinations to improve the accuracy.

![](img/088aa145-de48-42fd-a671-3b2224d6ec9e.png)

```py
# Extracting Encoder section of the Model for prediction of latent variables
>>> encoder = Model(history.input,history.get_layer("second").output)

# Predicting latent variables with extracted Encoder model
>>> reduced_X = encoder.predict(Xtest)
Converting the outputs into Pandas data frame structure for better representation
>>> final_pdframe = pd.DataFrame(reduced_X)
>>> final_pdframe.columns = ["xaxis","yaxis"]
>>> final_pdframe["word_indx"] = xsts
>>> final_pdframe["word"] = final_pdframe["word_indx"].map(idx2word)
>>> rows = random.sample(final_pdframe.index, 100)
>>> vis_df = final_pdframe.ix[rows]
>>> labels = list(vis_df["word"]);xvals = list(vis_df["xaxis"])
>>> yvals = list(vis_df["yaxis"])

#in inches
>>> plt.figure(figsize=(8, 8))
>>> for i, label in enumerate(labels):
... x = xvals[i]
... y = yvals[i]
... plt.scatter(x, y)
... plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points', ha='right',va='bottom')
>>> plt.xlabel("Dimension 1")
>>> plt.ylabel("Dimension 2")
>>> plt.show()
```

The following image describes the visualization of the words in two-dimensional space. Some words are closer to each other than other words, which indicates closeness and relationships with nearby words. For example, words such as `never`, `ever`, and `ask` are very close to each other.

![](img/7bbb0dc1-8d24-4db3-878c-330052d9f83c.png)

