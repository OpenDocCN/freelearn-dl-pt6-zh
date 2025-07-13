
# Caption Generation for Images

Caption generation is one of the most important applications in the field of deep learning and has gained quite a lot of interest recently. Image captioning models involve a combination of both visual information along with natural language processing. 

In this chapter, we will learn about:

*   Recent advancements in the field of the caption generation
*   How caption generation works
*   Implementation of caption generation models

# What is caption generation?

Caption generation is the task of describing an image with natural language. Previously, caption generation models worked on object detection models combined with templates that were used to generate text for detected objects. With all the advancements in deep learning, these models have been replaced with a combination of convolutional neural networks and recurrent neural networks.

An example is shown as follows:

![](img/41534487-3f30-46d0-94e2-2c346a61bb88.png)

Source: https://arxiv.org/pdf/1609.06647.pdf

There are several datasets that help us create image captioning models. 

# Exploring image captioning datasets

Several datasets are available for captioning image task. The datasets are usually prepared by showing an image to a few persons and asking them to write a sentence each about the image. Through this method, several captions are generated for the same image. Having multiple options of captions helps in better generalization. The difficulty lies in the ranking of model performance. For each generation, preferably, a human has to evaluate the caption. Automatic evaluation is difficult for this task. Let's explore the `Flickr8` dataset.

# Downloading the dataset

`Flickr8` is gathered from Flickr and is not permitted for commercial usage. Download the `Flickr8` dataset from [https://forms.illinois.edu/sec/1713398](https://forms.illinois.edu/sec/1713398). The descriptions can be found at [http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html). Download the text and images separately. Access to it can be obtained by filling in a form shown on the page:

![](img/4cffe58f-56d6-4303-ab4a-5d017ebb7257.png)

An email will be sent with the download link. Once downloaded and extracted, the files should be like this:

```py
Flickr8k_text
CrowdFlowerAnnotations.txt
Flickr_8k.devImages.txt
ExpertAnnotations.txt
Flickr_8k.testImages.txt
Flickr8k.lemma.token.txt
Flickr_8k.trainImages.txt
Flickr8k.token.txt readme.txt
```

The following are a couple of examples given in the dataset:

![](img/d427f636-3f7d-4499-a65c-cb6e45b86f7e.jpg)

The preceding figure shows the following components:

*   A man in street racer armor is examining the tire of another racer's motor bike
*   The two racers drove the white bike down the road
*   Two motorists are riding along on their vehicle that is oddly designed and colored
*   Two people are in a small race car driving by a green hill
*   Two people in racing uniforms in a street car

The following is example two:

![](img/a07da3d8-9578-4836-91b7-a236cc829a20.jpg)

The preceding figure shows the following components:

*   A man in a black hoodie and jeans skateboards down a railing
*   A man skateboards down a steep railing next to some steps
*   A person is sliding down a brick rail on a snowboard
*   A person walks down the brick railing near a set of steps
*   A snowboarder rides down a handrail without snow

As you can see, there are different captions provided for one image. The captions show the difficulty of the image captioning task.

# Converting words into embeddings

English words have to be converted into embeddings for caption generation. An embedding is nothing but a vector or numerical representation of words or images. It is useful if words are converted to a vector form such that arithmetic can be performed using the vectors.

Such an embedding can be learned by two methods, as shown in the following figure:

![](img/6d11a280-a213-4e63-80a6-2ec7971c36b3.png)

The **CBOW** method learns the embedding by predicting a word given the surrounding words. The **Skip-gram** method predicts the surrounding words given a word, which is the reverse of **CBOW**. Based on the history, a target word can be trained, as shown in the following figure:

![](img/918d0423-dbd7-4279-87d3-9574d66af009.png)

Once trained, the embedding can be visualized as follows:

![](img/63a42ddb-81c2-4564-9f58-f9fd2d53ce4b.png)

Visualization of words

This type of embedding can be used to perform vector arithmetic of words. This concept of word embedding will be helpful throughout this chapter.

# Image captioning approaches

There are several approaches to captioning images. Earlier methods used to construct a sentence based on the objects and attributes present in the image. Later, **recurrent neural networks** (**RNN**) were used to generate sentences. The most accurate method uses the attention mechanism. Let's explore these techniques and results in detail in this section. 

# Conditional random field

Initially a method was tried with the **conditional random field** (**CRF**) constructing the sentence with the objects and attributes detected in the image. The steps involved in this process are shown as follows:

![](img/7418c864-3eaf-4ee1-9448-87f293d689ce.png)

System flow for an example images (Source: http://www.tamaraberg.com/papers/generation_cvpr11.pdf)

CRF has limited ability to come up with sentences in a coherent manner. The quality of generated sentences is not great, as shown in the following screenshot:

![](img/c80448c0-73d4-402c-8c3c-5167ac91119a.png)

The sentences shown here are too structured despite getting the objects and attributes correct.

Kulkarni et al., in the paper [http://www.tamaraberg.com/papers/generation_cvpr11.pdf](http://www.tamaraberg.com/papers/generation_cvpr11.pdf), proposed a method of finding the objects and attributes from an image and using it to generate text with a **conditional random field** (**CRF**).

# Recurrent neural network on convolution neural network

A recurrent neural network can be combined with convolutional neural network features to produce new sentences. This enables end-to-end training of the models. The following is the architecture of such a model:

![](img/c66418d9-a2bd-4cf8-8c61-a50790b04846.png)

LSTM model (Source: https://arxiv.org/pdf/1411.4555.pdf)

There are several layers of **LSTM** used to produce the desired results. A few of the results produced by this model are shown in the following screenshot:

![](img/477f57f6-4861-4c3d-912a-e77913dd64b6.png)

Source: https://arxiv.org/pdf/1411.4555.pdf

These results are better than the results produced by CRF. This shows the power of LSTM in generating sentences.

Reference: Vinyals et al., in the paper [https://arxiv.org/pdf/1411.4555.pdf](https://arxiv.org/pdf/1411.4555.pdf), proposed an end to end trainable deep learning for image captioning, which has CNN and RNN stacked back to back. 

# Caption ranking

Caption ranking is an interesting way of selecting a caption from a set of captions. First, the images are ranked according to their features and corresponding captions are picked, as shown in this screenshot:

![](img/1d6c5f49-2c4f-4ad7-8806-66b9de6d2775.png)

Source: http://papers.nips.cc/paper/4470-im2text-describing-images-using-1-million-captioned-photographs.pdf

The top images can be re-ranked using a different set of attributes. By getting more images, the quality can improve a lot as shown in the following screenshot:

![](img/8390766c-a13d-47c9-949c-39f9bd46af62.png)

Source: http://papers.nips.cc/paper/4470-im2text-describing-images-using-1-million-captioned-photographs.pdf

The results are better with an increase in the number of images in the dataset.

To learn more about caption ranking, refer: [http://papers.nips.cc/paper/4470-im2text-describing-images-using-1-million-captioned-photographs.pdf](http://papers.nips.cc/paper/4470-im2text-describing-images-using-1-million-captioned-photographs.pdf)

# Dense captioning

Dense captioning is the problem of multiple captions on a single image. The following is the architecture of the problem:

![](img/96c1aa2c-16be-47f6-8bb5-8d8039b2d61b.png)

Source: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Johnson_DenseCap_Fully_Convolutional_CVPR_2016_paper.pdf

This architecture produces good results.

For more understanding refer: Johnson et al., in the paper [https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Johnson_DenseCap_Fully_Convolutional_CVPR_2016_paper.pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Johnson_DenseCap_Fully_Convolutional_CVPR_2016_paper.pdf), proposed a method for dense captioning.

# RNN captioning

The visual features can be used with sequence learning to form the output.

![](img/5bd7bd01-b1a9-43a4-a024-b30cb6a4984d.png)

This is an architecture for generating captions.  

For details, refer: Donahue et al., in the paper [https://arxiv.org/pdf/1411.4389.pdf](https://arxiv.org/pdf/1411.4389.pdf), proposed **Long-term recurrent convolutional architectures** (**LRCN**) for the task of image captioning. 

# Multimodal captioning

Both the image and text can be mapped to the same embedding space to generate a caption.

![](img/32ad2629-93d0-4a4f-bf63-69e56fc63f33.png)

A decoder is required to generate the caption. 

# Attention-based captioning

 For detailed learning, refer: Xu et al., in the paper, [https://arxiv.org/pdf/1502.03044.pdf](https://arxiv.org/pdf/1502.03044.pdf), proposed a method for image captioning using an **attention mechanism**.

Attention-based captioning has become popular recently as it provides better accuracy:

![](img/c6f17ce4-f2c2-4668-86e6-2424644885af.png)

This method trains an attention model in the sequence of the caption, thereby producing better results:

![](img/79f3a9f7-3f04-4a27-9375-4f82267d442e.png)

Here is a diagram of **LSTM** with attention-generating captions:

![](img/5f0220af-d7d0-4281-b1ba-092ef70358f6.png)

There are several examples shown here, with an excellent visualization of objects unfolding in a time series manner:

![](img/ed973502-76a8-4952-8809-e25aef013f7a.png)

Unfolding objects in time series manner

The results are really excellent!

# Implementing a caption generation model

First, let's read the dataset and transform it the way we need. Import the `os` library and declare the directory in which the dataset is present, as shown in the following code: 

```py
import os
annotation_dir = 'Flickr8k_text'

```

Next, define a function to open a file and return the lines present in the file as a list:

```py
def read_file(file_name):
    with open(os.path.join(annotation_dir, file_name), 'rb') as file_handle:
        file_lines = file_handle.read().splitlines()
    return file_lines
```

Read the image paths of the training and testing datasets followed by the captions file:

```py
train_image_paths = read_file('Flickr_8k.trainImages.txt')
test_image_paths = read_file('Flickr_8k.testImages.txt')
captions = read_file('Flickr8k.token.txt')

print(len(train_image_paths))
print(len(test_image_paths))
print(len(captions))
```

This should print the following:

```py
6000
1000
40460
```

Next, the image-to-caption map has to be generated. This will help in training for easily looking up captions. Also, unique words present in the caption dataset will help to create the vocabulary:

```py
image_caption_map = {}
unique_words = set()
max_words = 0
for caption in captions:
    image_name = caption.split('#')[0]
    image_caption = caption.split('#')[1].split('\t')[1]
    if image_name not in image_caption_map.keys():
        image_caption_map[image_name] = [image_caption]
    else:
        image_caption_map[image_name].append(image_caption)
    caption_words = image_caption.split()
    max_words = max(max_words, len(caption_words))
    [unique_words.add(caption_word) for caption_word in caption_words]
```

Now, two maps have to be formed. One is word to index and the other is index to word map:

```py
unique_words = list(unique_words)
word_to_index_map = {}
index_to_word_map = {}
for index, unique_word in enumerate(unique_words):
    word_to_index_map[unique_word] = index
    index_to_word_map[index] = unique_word
print(max_words)
```

The maximum number of words present in a caption is 38, which will help in defining the architecture. Next, import the libraries:

```py
from data_preparation import train_image_paths, test_image_paths
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
import pickle
import os
```

Now create the `ImageModel` class for loading the VGG model with weights:

```py
class ImageModel:
    def __init__(self):
        vgg_model = VGG16(weights='imagenet', include_top=True)
        self.model = Model(input=vgg_model.input,
                           output=vgg_model.get_layer('fc2').output)
```

The weights are downloaded and stored. It may take some time at the first attempt. Next, a separate model is created so that a second fully connected layer is predicted. The following is a method for reading an image from a path and preprocessing:

```py
    @staticmethod
    def load_preprocess_image(image_path):
        image_array = image.load_img(image_path, target_size=(224, 224))
        image_array = image.img_to_array(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        return image_array
```

Next, define a method to load the image and do prediction. The predicted second fully connected layer can be reshaped to `4096`:

```py
    def extract_feature_from_imagfe_path(self, image_path):
        image_array = self.load_preprocess_image(image_path)
        features = self.model.predict(image_array)
        return features.reshape((4096, 1))
```

Go through a list of image paths and create a list of features:

```py
    def extract_feature_from_image_paths(self, work_dir, image_names):
        features = []
        for image_name in image_names:
            image_path = os.path.join(work_dir, image_name)
            feature = self.extract_feature_from_image_path(image_path)
            features.append(feature)
        return features
```

Next, store the extracted features as a pickle file:

```py
    def extract_features_and_save(self, work_dir, image_names, file_name):
        features = self.extract_feature_from_image_paths(work_dir, image_names)
        with open(file_name, 'wb') as p:
            pickle.dump(features, p)
```

Next, initialize the class and extract both training and testing image features:

```py
I = ImageModel()
I.extract_features_and_save(b'Flicker8k_Dataset',train_image_paths, 'train_image_features.p')
I.extract_features_and_save(b'Flicker8k_Dataset',test_image_paths, 'test_image_features.p')
```

Import the layers required to construct the model:

```py
from data_preparation import get_vocab
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence
```

Get the vocabulary required:

```py
image_caption_map, max_words, unique_words, \
word_to_index_map, index_to_word_map = get_vocab()
vocabulary_size = len(unique_words)
```

For the final caption generation model:

```py
image_model = Sequential()
image_model.add(Dense(128, input_dim=4096, activation='relu'))
image_model.add(RepeatVector(max_words))
```

For the language, a model is created:

```py
lang_model = Sequential()
lang_model.add(Embedding(vocabulary_size, 256, input_length=max_words))
lang_model.add(LSTM(256, return_sequences=True))
lang_model.add(TimeDistributed(Dense(128)))
```

The two different models are merged to form the final model:

```py
model = Sequential()
model.add(Merge([image_model, lang_model], mode='concat'))
model.add(LSTM(1000, return_sequences=False))
model.add(Dense(vocabulary_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
batch_size = 32
epochs = 10
total_samples = 9
model.fit_generator(data_generator(batch_size=batch_size), steps_per_epoch=total_samples / batch_size,
                    epochs=epochs, verbose=2)

```

This model can be trained to generate captions.

# Summary

In this chapter, we learned image captioning techniques. First, we understood the embedding space of word vectors. Then, several approaches for image captioning were learned. Then came the implementation of the image captioning model.

In the next chapter, we will take a look at the concept of **Generative Adversarial Networks** (**GAN**). GANs are intriguing and useful for generating images for various purposes.
