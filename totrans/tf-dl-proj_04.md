
# Building GANs for Conditional Image Creation

Yann LeCun, Director of Facebook AI, has recently stated that "*Generative Adversarial Networks is the most interesting idea in the last ten years in machine learning*", and that is certainly confirmed by the elevated interest in academia about this deep learning solution. If you look at recent papers on deep learning (but also look at the leading trends on LinkedIn or Medium posts on the topic), there has really been an overproduction of variants of GANs.

You can get an idea of what a *zoo* the world of GANs has become just by glancing the continuously updated reference table, created by Hindu Puravinash, which can be found at [https://github.com/hindupuravinash/the-gan-zoo/blob/master/gans.tsv](https://github.com/hindupuravinash/the-gan-zoo/blob/master/gans.tsv) or by studying the GAN timeline prepared by Zheng Liu, which can be found at [https://github.com/dongb5/GAN-Timeline](https://github.com/dongb5/GAN-Timeline) and can help you putting everything into time perspective.

GANs have the power to strike the imagination because they can demonstrate the creative power of AI, not just its computational strength. In this chapter, we are going to:

*   Demystify the topic of GANs by providing you with all the necessary concepts to understand what GANs are, what they can do at the moment, and what they are expected to do
*   Demonstrate how to generate images both based on the initial distribution of example images (the so-called unsupervised GANs)
*   Explain how to condition the GAN to the kind of resulting image you expect them to generate for you
*   Set up a basic yet complete project that can work with different datasets of handwritten characters and icons
*   Provide you with basic instructions how to train your GANs in the Cloud (specifically on Amazon AWS)

The success of GANs much depends, besides the specific neural architecture you use, on the problem they face and the data you feed them with. The datasets we have chosen for this chapter should provide satisfactory results. We hope you will enjoy and be inspired by the creative power of GANs!

# Introducing GANs

We'll start with some quite recent history because GANs are among the newest ideas you'll find around AI and deep learning.

Everything started in 2014, when Ian Goodfellow and his colleagues (there is also Yoshua Bengio closing the list of contributors) at the *Departement d'informatique et de recherche opérationnelle* at Montreal University published a paper on **Generative Adversarial Nets** (**GANs**), a framework capable of generating new data based on a set of initial examples:

*GOODFELLOW*, Ian, et al. Generative Adversarial Nets. In: *Advances in Neural Information Processing Systems*. 2014\. p. 2672-2680: [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661).

The initial images produced by such networks were astonishing, considering the previous attempts using Markov chains which were far from being credible. In the image, you can see some of the examples proposed in the paper, showing examples reproduced from MNIST, **Toronto Face Dataset** (**TFD**) a non-public dataset and CIFAR-10 datasets:

![](img/5550d1dd-463a-469c-a3ef-f4f11b3c5386.png)

Figure 1: Samples from the first paper on GANs using different datasets for learning to generate fresh images: a) MNIST b) TFD c) and d) CIFAR-10
SOURCE: *GOODFELLOW*, Ian, et al. Generative Adversarial Nets. In: *Advances in Neural Information Processing Systems*. 2014\. p. 2672-2680

The paper was deemed quite innovative because it put working together deep neural networks and game theory in a really smart architecture that didn't require much more than the usual back-propagation to train. GANs are generative models, models that can generate data because they have inscribed a model distribution (they learned it, for instance). Consequently when they generate something it is just like if they were sampling from that distribution.

# The key is in the adversarial approach

The key to understanding how GANs can be such successful generative models resides in the term adversarial. Actually, the GANs architecture is made up of two distinct networks that are optimized based on the pooling of respective errors, and that's called an **adversarial process**.

You start with a real dataset, let's call it R, containing your images or your data of a different kind (GANs are not limited to images only, though they constitute the major application). You then set up a generator network, G, which tries to make fake data that looks like the genuine data, and you set up a discriminator, D, whose role is to compare the data produced by G mixed against the real data, R, and figures out which is genuine and which is not.

Goodfellow used the art forgers metaphor to describe the process, being, the generator the forgers, and the discriminator the detective (or the art critic) that has to disclose their misdeed. There is a challenge between the forgers and the detective because while the forgers have to become more skillful in order not to be detected, the detective has to become better at detecting fakes. Everything turns into an endless fight between the forgers and the detective until the forged artifacts are completely similar to the originals. When GANs overfit, in fact, they just reproduce the originals. It really seems an explanation of a competitive market, and it really is, because the idea comes from competitive game theory.

In GANs, the generator is incentivized to produce images that the discriminator cannot figure out if they are a fake or not. An obvious solution for the generator is simply to copy some training image or to just settle down for some produced image that seems successful with the discriminator. One solution is *one-sided label smoothing* a technique which we will be applying in our project. It is described in SALIMANS, Tim, et al. Improved techniques for training gans. In: <q>Advances in Neural Information Processing Systems</q><q>. 2016\. p. 2234-2242</q>: [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498).

Let's discuss how things actually work a little bit more. At first, the generator, *G*, is clueless and produces completely random data (it has actually never seen a piece of original data), it is therefore punished by the discriminator, *D--*an easy job figuring out the real versus the fake data. *G* takes full blame and starts trying something different to get better feedback from *D*. This is done completely randomly because the only data the generator sees is a random input called *Z*, it never touches the real data. After many trials and fails, hinted by the discriminator, the generator at last figures out what to do and starts to produce credible outputs. In the end, given enough time, the generator will exactly replicate all the original data without ever having seen a single example of it:

![](img/6c2256ef-2d1e-4ded-a5ca-0ac8cd9f16b8.png)

Figure 2: Illustrative example of how a vanilla GAN architecture works

# A cambrian explosion

As mentioned, there are new papers on GANs coming out every month (as you can check on the reference table made by Hindu Puravinash that we mentioned at the beginning of the chapter).

Anyway, apart from the vanilla implementation described in the initial paper from Goodfellow and his colleagues, the most notable implementations to take notice of are **deep convolutional generative adversarial networks **(**DCGANs**) and **conditional GANs (**CGANs**)**.

*   DCGANs are GANs based on CNN architecture (<q>RADFORD, Alec; METZ, Luke; CHINTALA, Soumith. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434</q><q>, 2015</q>: [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)).
*   CGANs are DCGANs which are conditioned on some input label so that you can obtain as a result an image with certain desired characteristics (<q>MIRZA, Mehdi; OSINDERO, Simon. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784</q><q>, 2014</q>: [https://arxiv.org/abs/1411.1784](https://arxiv.org/abs/1411.1784)). Our project will be programming a `CGAN` class and training it on different datasets in order to prove its functioning.

But there are also other interesting examples around (which are not covered by our project) offering practical solutions to problems related to image creation or improvement:

*   A CycleGAN translates an image into another (the classic example is the horse that becomes a zebra: <q>ZHU, Jun-Yan, et al. Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint arXiv:1703.10593</q><q>, 2017</q>: [https://arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593)) 
*   A StackGAN creates a realistic image from a text describing the image (<q>ZHANG, Han, et al. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. arXiv preprint arXiv:1612.03242</q><q>, 2016</q>: [https://arxiv.org/abs/1612.03242](https://arxiv.org/abs/1612.03242))
*   A Discovery GAN (DiscoGAN) transfers stylistic elements from one image to another, thus transferring texture and decoration from a fashion item such as a bag to another fashion item such as a pair of shoes (<q>KIM, Taeksoo, et al. Learning to discover cross-domain relations with generative adversarial networks. arXiv preprint arXiv:1703.05192</q><q>, 2017</q>: [https://arxiv.org/abs/1703.05192](https://arxiv.org/abs/1703.05192))
*   A SRGAN can convert low-quality images into high-resolution ones (<q>LEDIG, Christian, et al. Photo-realistic single image super-resolution using a generative adversarial network. arXiv preprint arXiv:1609.04802</q><q>, 2016</q>: [https://arxiv.org/abs/1609.04802](https://arxiv.org/abs/1609.04802))

# DCGANs

DCGANs are the first relevant improvement on the GAN architecture. DCGANs always successfully complete their training phase and, given enough epochs and examples, they tend to generate satisfactory quality outputs. That soon made them the baseline for GANs and helped to produce some amazing achievements, such as generating new Pokemon from known ones: [https://www.youtube.com/watch?v=rs3aI7bACGc](https://www.youtube.com/watch?v=rs3aI7bACGc) or creating faces of celebrities that actually never existed but are incredibly realistic (nothing uncanny), just as NVIDIA did: [https://youtu.be/XOxxPcy5Gr4](https://youtu.be/XOxxPcy5Gr4) using a new training approach called **progressing growing**: [http://research.nvidia.com/sites/default/files/publications/karras2017gan-paper.pdf](http://research.nvidia.com/sites/default/files/publications/karras2017gan-paper.pdf). They have their root in using the same convolutions used in image classification by deep learning supervised networks, and they use some smart tricks:

*   Batch normalization in both networks
*   No fully hidden connected layers
*   No pooling, just stride-in convolutions
*   ReLU activation functions

# Conditional GANs

In **conditional GANs** (**CGANs**), adding a vector of features controls the output and provides a better guide to the generator in figuring out what to do. Such a vector of features could encode the class the image should be derived be from (that is an image of a woman or a man if we are trying to create faces of imaginary actors) or even a set of specific characteristics we expect from the image (for imaginary actors, it could be the type of hair, eyes or complexion). The trick is done by incorporating the information into the images to be learned and into the *Z* input, which is not completely random anymore. The evaluation by the discriminator is done not only on the resemblance of fake data to the original data but also on the correspondence of the fake data image to its input label (or features):

![](img/bf357ac9-ebea-4b71-b3d1-868b0738625d.png)

Figure 3: Combining Z input with Y input (a labeling feature vector) allows generating controlled images

# The project

Importing the right libraries is where we start. Apart from `tensorflow`, we will be using `numpy` and math for computations, `scipy`, `matplolib` for images and graphics, and `warnings`, `random`, and `distutils` for support in specific operations:

```py
import numpy as np
import tensorflow as tf
import math
import warnings
import matplotlib.pyplot as plt
from scipy.misc import imresize
from random import shuffle
from distutils.version import LooseVersion
```

# Dataset class

Our first step is to provide the data. We will rely on datasets that have already been preprocessed, but our readers could use different kinds of images for their own GAN implementation. The idea is to keep separate a `Dataset` class that will have the task of providing batches of normalized and reshaped images to the GANs class we will build later.

In the initialization, we will deal with both images and their labels (if available). Images are first reshaped (if their shape differs from the one defined when instantiating the class), then shuffled. Shuffling helps GANs learning better if any order, for instance by class, is initially inscribed into the dataset - and this is actually true for any machine learning algorithm based on stochastic gradient descent: <q>BOTTOU, Léon. Stochastic gradient descent tricks. In: Neural networks: Tricks of the trade</q><q>. Springer, Berlin, Heidelberg, 2012\. p. 421-436</q>: [https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf). Labels instead are encoded using one-hot encoding, that is, a binary variable is created for each one of the classes, which is set to one (whereas others are set to zero) to represent the label as a vector.

For instance, if our classes are `{dog:0, cat:1}`, we will have these two one-hot encoded vectors to represent them: `{dog:[1, 0], cat:[0, 1]}`.

In such a way, we can easily add the vector to our image, as a further channel, and inscribe into it some kind of visual characteristic to be replicated by our GAN. Moreover, we could arrange the vectors in order to inscribe even more complex classes with special characteristics. For instance, we could specify the code for a class we prefer to be generated, and we can also specify some of its characteristics:

```py
class Dataset(object):
    def __init__(self, data, labels=None, width=28, height=28, 
                                    max_value=255, channels=3):
        # Record image specs
        self.IMAGE_WIDTH = width
        self.IMAGE_HEIGHT = height
        self.IMAGE_MAX_VALUE = float(max_value)
        self.CHANNELS = channels
        self.shape = len(data), self.IMAGE_WIDTH, 
                                self.IMAGE_HEIGHT, self.CHANNELS
        if self.CHANNELS == 3:
            self.image_mode = 'RGB'
            self.cmap = None
        elif self.CHANNELS == 1:
            self.image_mode = 'L'
            self.cmap = 'gray'

        # Resize if images are of different size
        if data.shape[1] != self.IMAGE_HEIGHT or \
                            data.shape[2] != self.IMAGE_WIDTH:
            data = self.image_resize(data, 
                   self.IMAGE_HEIGHT, self.IMAGE_WIDTH)

        # Store away shuffled data
        index = list(range(len(data)))
        shuffle(index)
        self.data = data[index]

        if len(labels) > 0:
            # Store away shuffled labels
            self.labels = labels[index]
            # Enumerate unique classes
            self.classes = np.unique(labels)
            # Create a one hot encoding for each class
            # based on position in self.classes
            one_hot = dict()
            no_classes = len(self.classes)
            for j, i in enumerate(self.classes):
                one_hot[i] = np.zeros(no_classes)
                one_hot[i][j] = 1.0
            self.one_hot = one_hot
        else:
            # Just keep label variables as placeholders
            self.labels = None
            self.classes = None
            self.one_hot = None

    def image_resize(self, dataset, newHeight, newWidth):
        """Resizing an image if necessary"""
        channels = dataset.shape[3]
        images_resized = np.zeros([0, newHeight, 
                         newWidth, channels], dtype=np.uint8)
        for image in range(dataset.shape[0]):
            if channels == 1:
                temp = imresize(dataset[image][:, :, 0],
                               [newHeight, newWidth], 'nearest')
                temp = np.expand_dims(temp, axis=2)
            else:
                temp = imresize(dataset[image], 
                               [newHeight, newWidth], 'nearest')
            images_resized = np.append(images_resized, 
                            np.expand_dims(temp, axis=0), axis=0)
        return images_resized
```

The `get_batches` method will just release a batch subset of the dataset and normalize the data by dividing the pixel values by the maximum (256) and subtracting -0.5\. The resulting images will have float values in the interval [-0.5, +0.5]:

```py
def get_batches(self, batch_size):
    """Pulling batches of images and their labels"""
    current_index = 0
    # Checking there are still batches to deliver
    while current_index < self.shape[0]:
        if current_index + batch_size > self.shape[0]:
            batch_size = self.shape[0] - current_index
        data_batch = self.data[current_index:current_index \
                               + batch_size]
        if len(self.labels) > 0:
            y_batch = np.array([self.one_hot[k] for k in \
            self.labels[current_index:current_index +\
            batch_size]])
        else:
            y_batch = np.array([])
        current_index += batch_size
        yield (data_batch / self.IMAGE_MAX_VALUE) - 0.5, y_batch
```

# CGAN class

The `CGAN` class contains all the functions necessary for running a conditional GAN based on the `CGAN` model. The deep convolutional generative adversarial networks proved to have the performance in generating photo-like quality outputs. We have previously introduced CGANs, so just to remind you, their reference paper is:

RADFORD, Alec; METZ, Luke; CHINTALA, Soumith. Unsupervised representation learning with deep convolutional Generative Adversarial Networks. arXiv preprint arXiv:1511.06434, 2015 at [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434).

In our project, we will then add the conditional form of the `CGAN` that uses label information as in a supervised learning task. Using labels and integrating them with images (this is the trick) will result in much better images and in the possibility of deciding the characteristics of the generated image.

The reference paper for conditional GANs is:

MIRZA, Mehdi; OSINDERO, Simon. *Conditional Generative Adversarial Nets*. arXiv preprint arXiv:1411.1784, 2014, [https://arxiv.org/abs/1411.1784](https://arxiv.org/abs/1411.1784).

Our `CGAN` class expects as input a dataset class object, the number of epochs, the image `batch_size`, the dimension of the random input used for the generator (`z_dim`), and a name for the GAN (for saving purposes). It also can be initialized with different values for alpha and smooth. We will discuss later what these two parameters can do for the GAN network.

The instantiation sets all the internal variables and performs a performance check on the system, raising a warning if a GPU is not detected:

```py
class CGan(object):
    def __init__(self, dataset, epochs=1, batch_size=32, 
                 z_dim=96, generator_name='generator',
                 alpha=0.2, smooth=0.1, 
                 learning_rate=0.001, beta1=0.35):

        # As a first step, checking if the 
        # system is performing for GANs
        self.check_system()

        # Setting up key parameters
        self.generator_name = generator_name
        self.dataset = dataset
        self.cmap = self.dataset.cmap
        self.image_mode = self.dataset.image_mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.alpha = alpha
        self.smooth = smooth
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.g_vars = list()
        self.trained = False

    def check_system(self):
        """
        Checking system suitability for the project
        """
        # Checking TensorFlow version >=1.2
        version = tf.__version__
        print('TensorFlow Version: %s' % version)

        assert LooseVersion(version) >= LooseVersion('1.2'),\
        ('You are using %s, please use TensorFlow version 1.2 \
                                         or newer.' % version)

        # Checking for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found installed on the system.\
                           It is advised to train your GAN using\
                           a GPU or on AWS')
        else:
            print('Default GPU Device: %s' % tf.test.gpu_device_name())
```

The `instantiate_inputs` function creates the TensorFlow placeholders for the inputs, both real and random. It also provides the labels (treated as images of the same shape of the original but for a channel depth equivalent to the number of classes), and for the learning rate of the training procedure:

```py
    def instantiate_inputs(self, image_width, image_height,
                           image_channels, z_dim, classes):
        """
        Instantiating inputs and parameters placeholders:
        real input, z input for generation, 
        real input labels, learning rate
        """
        inputs_real = tf.placeholder(tf.float32, 
                       (None, image_width, image_height,
                        image_channels), name='input_real')
        inputs_z = tf.placeholder(tf.float32, 
                       (None, z_dim + classes), name='input_z')
        labels = tf.placeholder(tf.float32, 
                        (None, image_width, image_height,
                         classes), name='labels')
        learning_rate = tf.placeholder(tf.float32, None)
        return inputs_real, inputs_z, labels, learning_rate
```

Next, we pass to work on the architecture of the network, defining some basic functions such as the `leaky_ReLU_activation` function (that we will be using for both the generator and the discriminator, contrary to what is prescribed in the original paper on deep convolutional GANs):

```py
 def leaky_ReLU_activation(self, x, alpha=0.2):
     return tf.maximum(alpha * x, x)

 def dropout(self, x, keep_prob=0.9):
     return tf.nn.dropout(x, keep_prob)
```

Our next function represents a discriminator layer. It creates a convolution using Xavier initialization, operates batch normalization on the result, sets a `leaky_ReLU_activation`, and finally applies `dropout` for regularization:

```py
    def d_conv(self, x, filters, kernel_size, strides,
               padding='same', alpha=0.2, keep_prob=0.5,
               train=True):
        """
        Discriminant layer architecture
        Creating a convolution, applying batch normalization,     
        leaky rely activation and dropout
        """
        x = tf.layers.conv2d(x, filters, kernel_size, 
                          strides, padding, kernel_initializer=\
                          tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)
        return x
```

Xavier initialization assures that the initial weights of the convolution are not too small, nor too large, in order to allow a better transmission of the signals through the network since the initial epochs. 

Xavier initialization provides a Gaussian distribution with a zero mean whose variance is given by 1.0 divided by the number of neurons feeding into a layer. It is because of this kind of initialization that deep learning moved away from pre-training techniques, previously used to set initial weights that could transmit back propagation even in the presence of many layers. You can read more about it and about the Glorot and Bengio's variant of the initialization in this post: [http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization.](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)

Batch normalization is described by this paper:

IOFFE, Sergey; SZEGEDY, Christian. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: *International Conference on Machine Learning*. 2015\. p. 448-456.

As noted by the authors, the batch normalization algorithm for normalization deals with covariate shift ([http://sifaka.cs.uiuc.edu/jiang4/domain_adaptation/survey/node8.html](http://sifaka.cs.uiuc.edu/jiang4/domain_adaptation/survey/node8.html)), that is, changing distribution in the inputs which could cause the previously learned weights not to work properly anymore. In fact, as distributions are initially learned in the first input layers, they are transmitted to all the following layers, and shifting later because suddenly the input distribution has changed (for instance, initially you had more input photos of cats than dogs, now it's the contrary) could prove quite daunting unless you have set the learning rate very low.

Batch normalization solves the problem of changing distribution in the inputs because it normalizes each batch by both mean and variance (using batch statistics), as illustrated by the paper <q>IOFFE, Sergey; SZEGEDY, Christian</q>. Batch normalization: Accelerating deep network training by reducing internal covariate shift. <q>In: International Conference on Machine Learning. 2015\. p. 448-456</q> (it can be found on the Internet at [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)).

``g_reshaping `and` g_conv_transpose`` are two functions that are part of the generator. They operate by reshaping the input, no matter if it is a flat layer or a convolution. Practically, they just reverse the work done by convolutions, restoring back the convolution-derived features into the original ones:

```py
    def g_reshaping(self, x, shape, alpha=0.2, 
                    keep_prob=0.5, train=True):
        """
        Generator layer architecture
        Reshaping layer, applying batch normalization, 
        leaky rely activation and dropout
        """
        x = tf.reshape(x, shape)
        x = tf.layers.batch_normalization(x, training=train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)
        return x

    def g_conv_transpose(self, x, filters, kernel_size, 
                         strides, padding='same', alpha=0.2, 
                         keep_prob=0.5, train=True):
        """
        Generator layer architecture
        Transposing convolution to a new size, 
        applying batch normalization,
        leaky rely activation and dropout
        """
        x = tf.layers.conv2d_transpose(x, filters, kernel_size,
                                       strides, padding)
        x = tf.layers.batch_normalization(x, training=train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)
        return x
```

The discriminator architecture operates by taking images as input and, by various convolutions, transforming them until the result is flattened and turned into logits and probabilities (using the sigmoid function). Practically, everything is the same as in an ordinal convolution:

```py
 def discriminator(self, images, labels, reuse=False):
     with tf.variable_scope('discriminator', reuse=reuse):
         # Input layer is 28x28x3 --> concatenating input
         x = tf.concat([images, labels], 3)

         # d_conv --> expected size is 14x14x32
         x = self.d_conv(x, filters=32, kernel_size=5,
                         strides=2, padding='same',
                         alpha=0.2, keep_prob=0.5)

         # d_conv --> expected size is 7x7x64
         x = self.d_conv(x, filters=64, kernel_size=5,
                         strides=2, padding='same',
                         alpha=0.2, keep_prob=0.5)

         # d_conv --> expected size is 7x7x128
         x = self.d_conv(x, filters=128, kernel_size=5,
                         strides=1, padding='same',
                         alpha=0.2, keep_prob=0.5)

         # Flattening to a layer --> expected size is 4096
         x = tf.reshape(x, (-1, 7 * 7 * 128))

         # Calculating logits and sigmoids
         logits = tf.layers.dense(x, 1)
         sigmoids = tf.sigmoid(logits)

         return sigmoids, logits
```

As for the generator, the architecture is exactly the opposite of the discriminator. Starting from an input vector, `z`, a dense layer is first created, then a series of transpositions aims to rebuild the inverse process of convolutions in the discriminator, ending in a tensor of the same shape of the input images, which undergoes a further transformation by a `tanh` activation function: 

```py
    def generator(self, z, out_channel_dim, is_train=True):

        with tf.variable_scope('generator', 
                                reuse=(not is_train)):
            # First fully connected layer
            x = tf.layers.dense(z, 7 * 7 * 512)

            # Reshape it to start the convolutional stack
            x = self.g_reshaping(x, shape=(-1, 7, 7, 512),
                                 alpha=0.2, keep_prob=0.5,
                                 train=is_train)

            # g_conv_transpose --> 7x7x128 now
            x = self.g_conv_transpose(x, filters=256,
                                      kernel_size=5,
                                      strides=2, padding='same',
                                      alpha=0.2, keep_prob=0.5,  
                                      train=is_train)

            # g_conv_transpose --> 14x14x64 now
            x = self.g_conv_transpose(x, filters=128,
                                      kernel_size=5, strides=2,
                                      padding='same', alpha=0.2,
                                      keep_prob=0.5,
                                      train=is_train)

            # Calculating logits and Output layer --> 28x28x5 now
            logits = tf.layers.conv2d_transpose(x,  
                                         filters=out_channel_dim, 
                                         kernel_size=5, 
                                         strides=1, 
                                         padding='same')
            output = tf.tanh(logits)

            return output
```

The architecture is very similar to the one depicted in the paper introducing CGANs, depicting how to reconstruct a 64 x 64 x 3 image from an initial input of a vector of size 100:

![](img/a4918bff-eb00-41c2-aea4-ff729f556f37.png)

Figure 4: The DCGAN architecture of the generator.
SOURCE: arXiv, 1511.06434,2015

After defining the architecture, the loss function is the next important element to define. It uses two outputs, the output from the generator, which is pipelined into the discriminator outputting logits, and the output from the real images pipelined themselves into the discriminator. For both, a loss measure is then calculated. Here, the smooth parameter comes in handy because it helps to smooth the probabilities of the real images into something that is not 1.0, allowing a better, more probabilistic learning by the GAN network (with full penalization it could become more difficult for the fake images to have a chance against the real ones).

The final discriminator loss is simply the sum of the loss calculated on the fake and on the real images. The loss is calculated on the fake comparing the estimated logits against the probability of zero. The loss on the real images is calculated comparing the estimated logit against the smoothed probability (in our case it is 0.9), in order to prevent overfitting and having the discriminator learn simply to spot the real images because it memorized them. The generator loss is instead calculated from the logits estimated by the discriminator for the fake images against a probability of 1.0\. In this way, the generator should strive to produce fake images that are estimated by the discriminator as most likely true (thus using a high probability). Therefore, the loss simply transmits from the discriminator evaluation on fake images to the generator in a feedback loop:

```py
    def loss(self, input_real, input_z, labels, out_channel_dim):

        # Generating output
        g_output = self.generator(input_z, out_channel_dim)
        # Classifying real input
        d_output_real, d_logits_real = self.discriminator(input_real, labels, reuse=False)
        # Classifying generated output
        d_output_fake, d_logits_fake = self.discriminator(g_output, labels, reuse=True)
        # Calculating loss of real input classification
        real_input_labels = tf.ones_like(d_output_real) * (1 - self.smooth) # smoothed ones
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                    labels=real_input_labels))
        # Calculating loss of generated output classification
        fake_input_labels = tf.zeros_like(d_output_fake) # just zeros
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                    labels=fake_input_labels))
        # Summing the real input and generated output classification losses
        d_loss = d_loss_real + d_loss_fake # Total loss for discriminator
        # Calculating loss for generator: all generated images should have been
        # classified as true by the discriminator
        target_fake_input_labels = tf.ones_like(d_output_fake) # all ones
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                    labels=target_fake_input_labels))

        return d_loss, g_loss
```

Since the work of the GAN is visual, there are a few functions for visualizing a sample of the current production from the generator, as well as a specific set of images:

```py
    def rescale_images(self, image_array):
        """
        Scaling images in the range 0-255
        """
        new_array = image_array.copy().astype(float)
        min_value = new_array.min()
        range_value = new_array.max() - min_value
        new_array = ((new_array - min_value) / range_value) * 255
        return new_array.astype(np.uint8)

    def images_grid(self, images, n_cols):
        """
        Arranging images in a grid suitable for plotting
        """
        # Getting sizes of images and defining the grid shape
        n_images, height, width, depth = images.shape
        n_rows = n_images // n_cols
        projected_images = n_rows * n_cols
        # Scaling images to range 0-255
        images = self.rescale_images(images)
        # Fixing if projected images are less
        if projected_images < n_images:
            images = images[:projected_images]
        # Placing images in a square arrangement
        square_grid = images.reshape(n_rows, n_cols, 
                                     height, width, depth)
        square_grid = square_grid.swapaxes(1, 2)
        # Returning a image of the grid
        if depth >= 3:
            return square_grid.reshape(height * n_rows, 
                                       width * n_cols, depth)
        else:
            return square_grid.reshape(height * n_rows, 
                                       width * n_cols)

    def plotting_images_grid(self, n_images, samples):
        """
        Representing the images in a grid
        """
        n_cols = math.floor(math.sqrt(n_images))
        images_grid = self.images_grid(samples, n_cols)
        plt.imshow(images_grid, cmap=self.cmap)
        plt.show()

    def show_generator_output(self, sess, n_images, input_z, 
                              labels, out_channel_dim,
                              image_mode):
        """
        Representing a sample of the 
        actual generator capabilities
        """
        # Generating z input for examples
        z_dim = input_z.get_shape().as_list()[-1]
        example_z = np.random.uniform(-1, 1, size=[n_images, \
                                       z_dim - labels.shape[1]])
        example_z = np.concatenate((example_z, labels), axis=1)
        # Running the generator
        sample = sess.run(
            self.generator(input_z, out_channel_dim, False),
            feed_dict={input_z: example_z})
        # Plotting the sample
        self.plotting_images_grid(n_images, sample)

    def show_original_images(self, n_images):
        """
        Representing a sample of original images
        """
        # Sampling from available images
        index = np.random.randint(self.dataset.shape[0], 
                                  size=(n_images))
        sample = self.dataset.data[index]
        # Plotting the sample
        self.plotting_images_grid(n_images, sample)
```

Using the Adam optimizer, both the discriminator loss and the generator one are reduced, starting first from the discriminator (establishing how good is the generator's production against true images) and then propagating the feedback to the generator, based on the evaluation of the effect the fake images produced by the generator had on the discriminator:

```py
    def optimization(self):
        """
        GAN optimization procedure
        """
        # Initialize the input and parameters placeholders
        cases, image_width, image_height,\
        out_channel_dim = self.dataset.shape
        input_real, input_z, labels, learn_rate = \    
                        self.instantiate_inputs(image_width,
                                               image_height,
                                            out_channel_dim, 
                                                 self.z_dim, 
                                  len(self.dataset.classes))

        # Define the network and compute the loss
        d_loss, g_loss = self.loss(input_real, input_z, 
                                    labels, out_channel_dim)

        # Enumerate the trainable_variables, split into G and D parts
        d_vars = [v for v in tf.trainable_variables() \
                    if v.name.startswith('discriminator')]
        g_vars = [v for v in tf.trainable_variables() \
                    if v.name.startswith('generator')]
        self.g_vars = g_vars

        # Optimize firt the discriminator, then the generatvor
        with tf.control_dependencies(\
                     tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(               
                                             self.learning_rate,
                   self.beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(
                                             self.learning_rate,
                   self.beta1).minimize(g_loss, var_list=g_vars)

        return input_real, input_z, labels, learn_rate, 
               d_loss, g_loss, d_train_opt, g_train_opt
```

At last, we have the complete training phase. In the training, there are two parts that require attention:

*   How the optimization is done in two steps:
    1.  Running the discriminator optimization
    2.  Working on the generator's one
*   How the random input and the real images are preprocessed by mixing them with labels in a way that creates further image layers containing the one-hot encoded information of the class relative to the image's label

In this way, the class is incorporated into the image, both in input and in output, conditioning the generator to take this information into account also, since it is penalized if it doesn't produce realistic images, that is, images with the right label attached. Let's say that our generator produces the image of a cat, but gives it the label of a dog. In this case, it will be penalized by the discriminator because the discriminator will notice how the generator cat is different from the real cats because of the different labels: 

```py
def train(self, save_every_n=1000):
    losses = []
    step = 0
    epoch_count = self.epochs
    batch_size = self.batch_size
    z_dim = self.z_dim
    learning_rate = self.learning_rate
    get_batches = self.dataset.get_batches
    classes = len(self.dataset.classes)
    data_image_mode = self.dataset.image_mode

    cases, image_width, image_height,\
    out_channel_dim = self.dataset.shape
    input_real, input_z, labels, learn_rate, d_loss,\ 
    g_loss, d_train_opt, g_train_opt = self.optimization()

    # Allowing saving the trained GAN
    saver = tf.train.Saver(var_list=self.g_vars)

    # Preparing mask for plotting progression
    rows, cols = min(5, classes), 5
    target = np.array([self.dataset.one_hot[i] \
             for j in range(cols) for i in range(rows)])

    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       for epoch_i in range(epoch_count):
           for batch_images, batch_labels \
                     in get_batches(batch_size):
                # Counting the steps
                step += 1
                # Defining Z
                batch_z = np.random.uniform(-1, 1, size=\
                                      (len(batch_images), z_dim))
                batch_z = np.concatenate((batch_z,\
                                           batch_labels), axis=1)
                # Reshaping labels for generator
                batch_labels = batch_labels.reshape(batch_size, 1, 1, classes)
                batch_labels = batch_labels * np.ones((batch_size, image_width, image_height, classes))
                # Sampling random noise for G
                batch_images = batch_images * 2
                # Running optimizers
                _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z,
                                                         labels: batch_labels, learn_rate: learning_rate})
                _ = sess.run(g_train_opt, feed_dict={input_z: batch_z, input_real: batch_images,
                                                         labels: batch_labels, learn_rate: learning_rate})

                # Cyclic reporting on fitting and generator output
                if step % (save_every_n//10) == 0:
                    train_loss_d = sess.run(d_loss,
                                                {input_z: batch_z, input_real: batch_images, labels: batch_labels})
                    train_loss_g = g_loss.eval({input_z: batch_z, labels: batch_labels})
                    print("Epoch %i/%i step %i..." % (epoch_i + 1, epoch_count, step),
                              "Discriminator Loss: %0.3f..." % train_loss_d,
                              "Generator Loss: %0.3f" % train_loss_g)
                if step % save_every_n == 0:
                    rows = min(5, classes)
                    cols = 5
                    target = np.array([self.dataset.one_hot[i] for j in range(cols) for i in range(rows)])
                    self.show_generator_output(sess, rows * cols, input_z, target, out_channel_dim, data_image_mode)
                    saver.save(sess, './'+self.generator_name+'/generator.ckpt')

            # At the end of each epoch, get the losses and print them out
            try:
                train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images, labels: batch_labels})
                train_loss_g = g_loss.eval({input_z: batch_z, labels: batch_labels})
                print("Epoch %i/%i step %i..." % (epoch_i + 1, epoch_count, step),
                         "Discriminator Loss: %0.3f..." % train_loss_d,
                          "Generator Loss: %0.3f" % train_loss_g)
            except:
                train_loss_d, train_loss_g = -1, -1

            # Saving losses to be reported after training
            losses.append([train_loss_d, train_loss_g])

        # Final generator output
        self.show_generator_output(sess, rows * cols, input_z, target, out_channel_dim, data_image_mode)
        saver.save(sess, './' + self.generator_name + '/generator.ckpt')

    return np.array(losses)
```

During the training, the network is constantly saved on disk. When it is necessary to generate new images, you don't need to retrain, but just upload the network and specify the label you want the GAN to produce images for:

```py
def generate_new(self, target_class=-1, rows=5, cols=5, plot=True):
        """
        Generating a new sample
        """
        # Fixing minimum rows and cols values
        rows, cols = max(1, rows), max(1, cols)
        n_images = rows * cols

        # Checking if we already have a TensorFlow graph
        if not self.trained:
            # Operate a complete restore of the TensorFlow graph
            tf.reset_default_graph()
            self._session = tf.Session()
            self._classes = len(self.dataset.classes)
            self._input_z = tf.placeholder(tf.float32, (None, self.z_dim + self._classes), name='input_z')
            out_channel_dim = self.dataset.shape[3]
            # Restoring the generator graph
            self._generator = self.generator(self._input_z, out_channel_dim)
            g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
            saver = tf.train.Saver(var_list=g_vars)
            print('Restoring generator graph')
            saver.restore(self._session, tf.train.latest_checkpoint(self.generator_name))
            # Setting trained flag as True
            self.trained = True

        # Continuing the session
        sess = self._session
        # Building an array of examples examples
        target = np.zeros((n_images, self._classes))
        for j in range(cols):
            for i in range(rows):
                if target_class == -1:
                    target[j * cols + i, j] = 1.0
                else:
                    target[j * cols + i] = self.dataset.one_hot[target_class].tolist()
        # Generating the random input
        z_dim = self._input_z.get_shape().as_list()[-1]
        example_z = np.random.uniform(-1, 1, 
                    size=[n_images, z_dim - target.shape[1]])
        example_z = np.concatenate((example_z, target), axis=1)
        # Generating the images
        sample = sess.run(
            self._generator,
            feed_dict={self._input_z: example_z})
        # Plotting
        if plot:
            if rows * cols==1:
                if sample.shape[3] <= 1:
                    images_grid = sample[0,:,:,0]
                else:
                    images_grid = sample[0]
            else:
                images_grid = self.images_grid(sample, cols)
            plt.imshow(images_grid, cmap=self.cmap)
            plt.show()
        # Returning the sample for later usage 
        # (and not closing the session)
        return sample
```

The class is completed by the `fit` method, which accepts both the learning rate parameter and the beta1 (an Adam optimizer parameter, adapting the parameter learning rates based on the average first moment, that is, the mean), and plots the resulting losses from the discriminator and the generator after the training is completed:

```py
    def fit(self, learning_rate=0.0002, beta1=0.35):
        """
        Fit procedure, starting training and result storage
        """
        # Setting training parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        # Training generator and discriminator
        with tf.Graph().as_default():
            train_loss = self.train()
        # Plotting training fitting
        plt.plot(train_loss[:, 0], label='Discriminator')
        plt.plot(train_loss[:, 1], label='Generator')
        plt.title("Training fitting")
        plt.legend()
```

# Putting CGAN to work on some examples

Now that the `CGAN` class is completed, let's go through some examples in order to provide you with fresh ideas on how to use this project. First of all, we will have to get everything ready for both downloading the necessary data and training our GAN. We start by importing the routine libraries:

```py
import numpy as np
import urllib.request
import tarfile
import os
import zipfile
import gzip
import os
from glob import glob
from tqdm import tqdm
```

We then proceed by loading in the dataset and `CGAN` classes that we previously prepared:

```py
from cGAN import Dataset, CGAN
```

The class `TqdmUpTo` is just a `tqdm` wrapper that enables the use of the progress display also for downloads. The class has been taken directly from the project's page at [https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm):

```py
class TqdmUpTo(tqdm):
    """
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by https://github.com/pypa/twine/pull/242
    https://github.com/pypa/twine/commit/42e55e06
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Total size (in tqdm units). 
        If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        # will also set self.n = b * bsize
        self.update(b * bsize - self.n)
```

Finally, if we are using a Jupyter notebook (warmly suggested for this roadshow), you have to enable the inline plotting of images:

```py
%matplotlib inline
```

We are now ready to proceed with the first example.

# MNIST

The `MNIST` database of handwritten digits was provided by Yann LeCun when he was at Courant Institute, NYU, and by Corinna Cortes (Google Labs) and Christopher J.C. Burges (Microsoft Research). It is considered the standard for learning from real-world image data with minimal effort in preprocessing and formatting. The database consists of handwritten digits, offering a training set of 60,000 examples and a test set of 10,000\. It is actually a subset of a larger set available from NIST. All the digits have been size-normalized and centered in a fixed-size image:

[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

![](img/ce2ca276-49f1-411a-be98-f281615230de.png)

Figure 5: A sample of the original MNIST helps to understand the quality of the images to be reproduced by the CGAN.

As a first step, we upload the dataset from the Internet and store it locally:

```py
labels_filename = 'train-labels-idx1-ubyte.gz'
images_filename = 'train-images-idx3-ubyte.gz'

url = "http://yann.lecun.com/exdb/mnist/"
with TqdmUpTo() as t: # all optional kwargs
    urllib.request.urlretrieve(url+images_filename,  
                               'MNIST_'+images_filename, 
                               reporthook=t.update_to, data=None)
with TqdmUpTo() as t: # all optional kwargs
    urllib.request.urlretrieve(url+labels_filename, 
                               'MNIST_'+labels_filename, 
                               reporthook=t.update_to, data=None)
```

In order to learn this set of handwritten numbers, we apply a batch of 32 images, a learning rate of `0.0002`, a `beta1` of `0.35`, a `z_dim` of `96`, and `15` epochs for training:

```py
labels_path = './MNIST_train-labels-idx1-ubyte.gz'
images_path = './MNIST_train-images-idx3-ubyte.gz'

with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), 
                               dtype=np.uint8, offset=8)

with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
        offset=16).reshape(len(labels), 28, 28, 1)

batch_size = 32
z_dim = 96
epochs = 16

dataset = Dataset(images, labels, channels=1)
gan = CGAN(dataset, epochs, batch_size, z_dim, generator_name='mnist')

gan.show_original_images(25)
gan.fit(learning_rate = 0.0002, beta1 = 0.35)
```

The following image represents a sample of the numbers generated by the GAN at the second epoch and at the last one:

![](img/3fe5beef-20c5-45ab-8e30-18b7ab0c260f.png)

Figure 6: The GAN's results as they appear epoch after epoch

After 16 epochs, the numbers appear to be well shaped and ready to be used. We then extract a sample of all the classes arranged by row.

Evaluating the performances of a GAN is still most often the matter of visual inspecting some of its results by a human judge, trying to figure out if the image could be a fake (like a discriminator) from its overall aspect or by precisely revealing details. GANs lack an objective function to help to evaluate and compare them, though there are some computational techniques that could be used as a metric such as the *log-likelihood*, as described by <q>THEIS, Lucas; OORD, Aäron van den; BETHGE, Matthias. A note on the evaluation of generative models. arXiv preprint arXiv:1511.01844</q><q>, 2015</q>: [https://arxiv.org/abs/1511.01844](https://arxiv.org/abs/1511.01844).

We will keep our evaluation simple and empirical and thus we will use a sample of images generated by the trained GAN in order to evaluate the performances of the network and we also try to inspect the training loss for both the generator and the discriminator in order to spot any particular trend:

![](img/57ca84ee-b495-4231-a862-a24a0c50afcc.png)

Figure 7: A sample of the final results after training on MNIST reveals it is an accessible task for a GAN network

Observing the training fit chart, represented in the figure the following, we notice how the generator reached the lowest error when the training was complete. The discriminator, after a previous peak, is struggling to get back to its previous performance values, pointing out a possible generator's breakthrough. We can expect that even more training epochs could improve the performance of this GAN network, but as you progress in the quality the output, it may take exponentially more time. In general, a good indicator of convergence of a GAN is having a downward trend of both the discriminator and generator, which is something that could be inferred by fitting a linear regression line to both loss vectors:

![](img/246d5831-c0aa-4504-947f-1236b006eea9.png)

Figure 8: The training fit along the 16 epochs

Training an amazing GAN network may take a very long time and a lot of computational resources. By reading this recent article appeared in the New York Times, [https://www.nytimes.com/interactive/2018/01/02/technology/ai-generated-photos.html](https://www.nytimes.com/interactive/2018/01/02/technology/ai-generated-photos.html), you can find a chart from NVIDIA showing the progress in time for the training of a progressive GAN learning from photos of celebrities. Whereas it can take a few days to get a decent result, for an astonishing one you need at least a fortnight. In the same way, even with our examples, the more training epochs you put in, the better the results.

# Zalando MNIST

Fashion `MNIST` is a dataset of Zalando's article images, composed of a training set of 60,000 examples and a test set of 10,000 examples. As with `MNIST`, each example is a 28x28 grayscale image, associated with a label from 10 classes. It was intended by authors from Zalando Research ([https://github.com/zalandoresearch/fashion-mnist/graphs/contributors](https://github.com/zalandoresearch/fashion-mnist/graphs/contributors)) as a replacement for the original MNIST dataset in order to better benchmark machine learning algorithms since it is more challenging to learn and much more representative of deep learning in real-world tasks ([https://twitter.com/fchollet/status/852594987527045120](https://twitter.com/fchollet/status/852594987527045120)).

[https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

![](img/72d9410c-dc59-41ba-b547-13ba2893f256.png)

Figure 9: A sample of the original Zalando dataset

We download the images and their labels separately:

```py
url = "http://fashion-mnist.s3-website.eu-central-\
       1.amazonaws.com/train-images-idx3-ubyte.gz"
filename = "train-images-idx3-ubyte.gz"
with TqdmUpTo() as t: # all optional kwargs
    urllib.request.urlretrieve(url, filename, 
                               reporthook=t.update_to, data=None)
url = "http://fashion-mnist.s3-website.eu-central-\
       1.amazonaws.com/train-labels-idx1-ubyte.gz"
filename = "train-labels-idx1-ubyte.gz"
_ = urllib.request.urlretrieve(url, filename)
```

In order to learn this set of images, we apply a batch of 32 images, a learning rate of `0.0002`, a `beta1` of `0.35`, a `z_dim` of `96`, and `10` epochs for training:

```py
labels_path = './train-labels-idx1-ubyte.gz'
images_path = './train-images-idx3-ubyte.gz'
label_names = ['t_shirt_top', 'trouser', 'pullover', 
               'dress', 'coat', 'sandal', 'shirt', 
               'sneaker', 'bag', 'ankle_boots']

with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), 
                               dtype=np.uint8,
                               offset=8)

with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
        offset=16).reshape(len(labels), 28, 28, 1)

batch_size = 32
z_dim = 96
epochs = 64

dataset = Dataset(images, labels, channels=1)
gan = CGAN(dataset, epochs, batch_size, z_dim, generator_name='zalando')

gan.show_original_images(25)
gan.fit(learning_rate = 0.0002, beta1 = 0.35)
```

The training takes a long time to go through all the epochs, but the quality appears to soon stabilize, though some problems take more epochs to disappear (for instance holes in shirts):

![](img/9ef7ad88-3460-48b3-a62d-11ed0db43d9a.png)

Figure 10: The evolution of the CGAN's training through epochs

Here is the result after 64 epochs:

![](img/7dc822bf-59d2-48af-8183-da1d5bd141b7.png)

Figure 11: An overview of the results achieved after 64 epochs on Zalando dataset

The result is fully satisfactory, especially for clothes and men's shoes. Women's shoes, however, seem more difficult to be learned because smaller and more detailed than the other images.

# EMNIST

The `EMNIST` dataset is a set of handwritten character digits derived from the `NIST` Special Database and converted to a 28 x 28 pixel image format and dataset structure that directly matches the `MNIST` dataset. We will be using `EMNIST` Balanced, a set of characters with an equal number of samples per class, which consists of 131,600 characters spread over 47 balanced classes. You can find all the references to the dataset in:

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from [http://arxiv.org/abs/1702.05373](http://arxiv.org/abs/1702.05373).

You can also explore complete information about `EMNIST` by browsing the official page of the dataset: [https://www.nist.gov/itl/iad/image-group/emnist-dataset](https://www.nist.gov/itl/iad/image-group/emnist-dataset). Here is an extraction of the kind of characters that can be found in the EMNIST Balanced:

![](img/aff40bc1-76d6-4e47-b6ea-bb6e1b027db8.png)

Figure 11: A sample of the original EMNIST dataset

```py
url = "http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
filename = "gzip.zip"
with TqdmUpTo() as t: # all optional kwargs
    urllib.request.urlretrieve(url, filename,  
                               reporthook=t.update_to, 
                               data=None)
```

After downloading from the NIST website, we unzip the downloaded package:

```py
zip_ref = zipfile.ZipFile(filename, 'r')
zip_ref.extractall('.')
zip_ref.close()
```

We remove the unused ZIP file after checking that the unzipping was successful:

```py
if os.path.isfile(filename):
    os.remove(filename)
```

In order to learn this set of handwritten numbers, we apply a batch of 32 images, a learning rate of `0.0002`, a `beta1` of `0.35`, a `z_dim` of `96`, and 10 epochs for training:

```py
labels_path = './gzip/emnist-balanced-train-labels-idx1-ubyte.gz'
images_path = './gzip/emnist-balanced-train-images-idx3-ubyte.gz'
label_names = []

with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
         offset=8)

with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                  offset=16).reshape(len(labels), 28, 28, 1)

batch_size = 32
z_dim = 96
epochs = 32

dataset = Dataset(images, labels, channels=1)
gan = CGAN(dataset, epochs, batch_size, z_dim,  
           generator_name='emnist')

gan.show_original_images(25)
gan.fit(learning_rate = 0.0002, beta1 = 0.35)
```

Here is a sample of some handwritten letters when completing the training after 32 epochs:

![](img/b9e19ccb-f85c-451d-837e-4c8b081285cc.png)

Figure 12: An overview of the results obtained training a CGAN on the EMNIST dataset

As for MNIST, a GAN can learn in a reasonable time to replicate handwritten letters in an accurate and credible way.

# Reusing the trained CGANs

After training a CGAN, you may find useful to use the produced images in other applications. The method `generate_new` can be used to extract single images as well as a set of images (in order to check the quality of results for a specific image class). It operates on a previously trained `CGan` class, so all you have to do is just to pickle it in order first to save it, then to restore it again when needed.

When the training is complete, you can save your `CGan` class using `pickle`, as shown by these commands:

```py
import pickle
pickle.dump(gan, open('mnist.pkl', 'wb'))
```

In this case, we have saved the `CGAN` trained on the MNIST dataset.

After you have restarted the Python session and memory is clean of any variable, you can just `import` again all the classes and restore the pickled `CGan`:

```py
from CGan import Dataset, CGan
import pickle
gan = pickle.load(open('mnist.pkl', 'rb'))
```

When done, you set the target class you would like to be generated by the `CGan` (in the example we ask for the number `8` to be printed) and you can ask for a single example, a grid 5 x 5 of examples or a larger 10 x 10 grid:

```py
nclass = 8
_ = gan.generate_new(target_class=nclass, 
                     rows=1, cols=1, plot=True)
_ = gan.generate_new(target_class=nclass, 
                     rows=5, cols=5, plot=True)
images = gan.generate_new(target_class=nclass,
                     rows=10, cols=10, plot=True)
print(images.shape)
```

If you just want to obtain an overview of all the classes, just set the parameter `target_class` to -1.

After having set out target class to be represented, the `generate_new` is called three times and the last one the returned values are stored into the `images` variable, which is sized (100, 28, 28, 1) and contains a Numpy array of the produced images that can be reused for our purposes. Each time you call the method, a grid of results is plotted as shown in the following figure:

![](img/f9f18203-dfc6-4fdc-8306-d108b47f9c45.png)

Figure 13: The plotted grid is a composition of the produced images, that is an image itself. From left to right, the plot of a
request for a 1 x 1, 5 x 5, 10 x 10 grid of results. The real images are returned by the method and can be reused.

If you don't need `generate_new` to plot the results, you simply set the `plot` parameter to False: `images = gan.generate_new(target_class=nclass, rows=10, cols=10, plot=False)`.

# Resorting to Amazon Web Service

As previously noticed, it is warmly suggested you use a GPU in order to train the examples proposed in this chapter. Managing to obtain results in a reasonable time using just a CPU is indeed impossible, and also using a GPU may turn into quite long hours waiting for the computer to complete the training. A solution, requiring the payment of a fee, could be to resort to Amazon Elastic Compute Cloud, also known as Amazon EC2 ([https://aws.amazon.com/it/ec2/](https://aws.amazon.com/it/ec2/)), part of the **Amazon Web Services** (**AWS**). On EC2 you can launch virtual servers that you can control from your computer using the Internet connection. You can require servers with powerful GPUs on EC2 and make your life with TensorFlow projects much easier.

Amazon EC2 is not the only cloud service around. We have suggested you this service because it is the one we used in order to test the code in this book. Actually, there are alternatives, such as Google Cloud Compute ([cloud.google.com](http://cloud.google.com)), Microsoft Azure (azure.microsoft.com) and many others.

Running the chapter’s code on EC2 requires having an account in AWS. If you don’t have one, the first step is to register at [aws.amazon.com](https://aws.amazon.com/), complete all the necessary forms and start with a free Basic Support Plan.

After you are registered on AWS, you just sign in and visit the EC2 page ([https://aws.amazon.com/ec2](https://aws.amazon.com/ec2)). There you will:

1.  Select a region which is both cheap and near to you which allows the kind of GPU instances we need, from EU (Ireland), Asia Pacific (Tokyo), US East (N. Virginia) and US West (Oregon).
2.  Upgrade your EC2 Service Limit report at: [https://console.aws.amazon.com/ec2/v2/home?#Limits](https://console.aws.amazon.com/ec2/v2/home?#Limits). You will need to access a **p3.2xlarge** instance. Therefore if your actual limit is zero, that should be taken at least to one, using the *Request Limit Increase* form (this may take up to 24 hours, but before it's complete, you won’t be able to access this kind of instance).
3.  Get some AWS credits (providing your credit card, for instance).

After setting your region and having enough credit and request limit increase, you can start a **p3.2xlarge** server (a GPU compute server for deep learning applications) set up with an OS already containing all the software you need (thanks to an AMI, an image prepared by Amazon):

1.  Get to the EC2 Management Console, and click on the **Launch Instance** button.
2.  Click on AWS Marketplace, and search for **Deep Learning AMI with Source Code v2.0 (ami-bcce6ac4)** AMI. This AMI has everything pre-installed: CUDA, cuDNN ([https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)), Tensorflow.
3.  Select the *GPU* compute **p3.2xlarge** instance. This instance has a powerful NVIDIA Tesla V100 GPU.
4.  Configure a security group (which you may call **Jupyter**) by adding **Custom TCP Rule**, with TCP protocol, on `port 8888`, accessible from anywhere. This will allow you to run a Jupyter server on the machine and see the interface from any computer connected to the Internet.
5.  Create an **Authentication Key Pair**. You can call it `deeplearning_jupyter.pem` for instance. Save it on your computer in a directory you can easily access.
6.  Launch the instance. Remember that you will be paying since this moment unless you **stop** it from the AWS menu—you still will incur in some costs, but minor ones and you will have the instance available for you, with all your data—or simply **terminate** it and don’t pay any more for it.

After everything is launched, you can access the server from your computer using ssh.

*   Take notice of the IP of the machine. Let’s say it is `xx.xx.xxx.xxx`, as an example.
*   From a shell pointing to the directory where you `.pem` file is, type:
    `ssh -i deeplearning_jupyter.pem ubuntu@ xx.xx.xxx.xxx`
*   When you have accessed the server machine, configure its Jupyter server by typing these commands: 
    `jupyter notebook --generate-config`
    `sed -ie "s/#c.NotebookApp.ip = 'localhost'/#c.NotebookApp.ip = '*'/g" ~/.jupyter/jupyter_notebook_config.py`

*   Operate on the server by copying the code (for instance by git cloning the code repository) and installing any library you may require. For instance, you could install these packages for this specific project:
    `sudo pip3 install tqdm`
    `sudo pip3 install conda`
*   Launch the Jupyter server by running the command:
    `jupyter notebook --ip=0.0.0.0 --no-browser`

*   At this point, the server will run and your ssh shell will prompt you the logs from Jupyter. Among the logs, take note of the token (it is something like a sequence of numbers and letters).
*   Open your browser and write in the address bar:
     `http:// xx.xx.xxx.xxx:8888/`

When required type the token and you are ready to use the Jupiter notebook as you were on your local machine, but it is actually operating on the server. At this point, you will have a powerful server with GPU for running all your experiments with GANs.

# Acknowledgements

In concluding this chapter, we would like to thank Udacity and Mat Leonard for their DCGAN tutorial, licensed under MIT ([https://github.com/udacity/deep-learning/blob/master/LICENSE](https://github.com/udacity/deep-learning/blob/master/LICENSE)) which provided a good starting point and a benchmark for this project.

# Summary

In this chapter, we have discussed at length the topic of Generative Adversarial Networks, how they work, and how they can be trained and used for different purposes. As a project, we have created a conditional GAN, one that can generate different types of images, based on your input and we learned how to process some example datasets and train them in order to have a pickable class capable of creating new images on demand.

