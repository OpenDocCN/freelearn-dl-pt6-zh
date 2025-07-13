
# Annotating Images with Object Detection API

Computer vision has made great leaps forward in recent years because of deep learning, thus granting computers a higher grade in understanding visual scenes. The potentialities of deep learning in vision tasks are great: allowing a computer to visually perceive and understand its surroundings is a capability that opens the door to new artificial intelligence applications in both mobility (for instance, self-driving cars can detect if an appearing obstacle is a pedestrian, an animal or another vehicle from the camera mounted on the car and decide the correct course of action) and human-machine interaction in everyday-life contexts (for instance, allowing a robot to perceive surrounding objects and successfully interact with them).

After presenting ConvNets and how they operate in the first chapter, we now intend to create a quick, easy project that will help you to use a computer to understand images taken from cameras and mobile phones, using images collected from the Internet or directly from your computer's webcam. The goal of the project is to find the exact location and the type of the objects in an image.

In order to achieve such classification and localization, we will leverage the new TensorFlow object detection API, a Google project that is part of the larger TensorFlow models project which makes a series of pre-trained neural networks available off-the-shelf for you to wrap up in your own custom applications.

In this chapter, we are going to illustrate the following:

*   The advantages of using the right data for your project
*   A brief presentation of the TensorFlow object detection API
*   How to annotate stored images for further use
*   How to visually annotate a video using `moviepy`
*   How to go real-time by annotating images from a webcam

# The Microsoft common objects in context

Advances in application of deep learning  in computer vision are often highly focalized on the kind of classification problems that can be summarized by challenges such as ImageNet (but also, for instance, PASCAL VOC - [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)) and the ConvNets suitable to crack it (Xception, VGG16, VGG19, ResNet50, InceptionV3, and MobileNet, just to quote the ones available in the well-known package `Keras`: [https://keras.io/applications/](https://keras.io/applications/)).

Though deep learning networks based on ImageNet data are the actual state of the art,  such networks can experience difficulties when faced with real-world applications. In fact, in practical applications, we have to process images that are quite different from the examples provided by ImageNet. In ImageNet the elements to be classified are clearly the only clear element present in the image, ideally set in an unobstructed way near the center of a neatly composed photo. In the reality of images taken from the field, objects are randomly scattered around, in often large number.  All these objects are also quite different from each other, creating sometimes confusing settings. In addition, often objects of interest cannot be clearly and directly perceived because they are visually obstructed by other potentially interesting objects.

Please refer to the figure from the following mentioned reference:

Figure 1: A sample of images from ImageNet: they are arranged in a hierarchical structure, allowing working with both general or more specific classes.

SOURCE: DENG, Jia, et al. Imagenet: A large-scale hierarchical image database. In: Computer Vision and Pattern Recognition, 2009\. CVPR 2009\. IEEE Conference on. IEEE, 2009\. p. 248-255.

Realistic images contain multiple objects that sometimes can hardly be distinguished from a noisy background. Often you really cannot create interesting projects just by labeling an image with a tag simply telling you the object was recognized with the highest confidence.

In a real-world application, you really need to be able to do the following:

*   Object classification of single and multiple instances when recognizing various objects, often of the same class
*   Image localization, that is understanding where the objects are in the image
*   Image segmentation,  by marking each pixel in the images with a label: the type of object or background in order to be able to cut off interesting parts from the background.

The necessity to train a ConvNet to be able to achieve some or all of the preceding mentioned objectives led to the creation of the **Microsoft common objects in context** (**MS COCO**) dataset, as described in the paper: LIN, Tsung-Yi, et al. Microsoft coco: common objects in context. In: *European conference on computer vision*. Springer, Cham, 2014\. p. 740-755. (You can read the original paper at the following link: [https://arxiv.org/abs/1405.0312](https://arxiv.org/abs/1405.0312).) This dataset is made up of 91 common object categories, hierarchically ordered, with 82 of them having more than 5,000 labeled instances. The dataset totals 2,500,000 labeled objects distributed in 328,000 images.

Here are the classes that can be recognized in the MS COCO dataset:

```py
{1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
```

Though the `ImageNet` dataset can present 1,000 object classes (as described at [https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)) distributed in 14,197,122 images,  MS COCO offers the peculiar feature of multiple objects distributed in a minor number of images (the dataset has been gathered using Amazon Mechanical Turk, a somehow more costly approach but shared by ImageNet, too). Given such premises, the MS COCO images can be considered very good examples of *contextual relationships and non-iconic object views*, since objects are arranged in realistic positions and settings. This can be verified from this comparative example taken from the MS COCO paper previously mentioned:

![](img/e385e147-49ee-4177-a540-cd7715d1a81d.png)

Figure 2: Examples of iconic and non-iconic images. SOURCE: <q>LIN, Tsung-Yi, et al. Microsoft coco: common objects in context. In: European conference on computer vision. Springer, Cham, 2014\. p. 740-755.</q>

In addition, the image annotation of MS COCO is particularly rich, offering the coordinates of the contours of the objects present in the images. The contours can be easily translated into bounding boxes, boxes that delimit the part of the image where the object is located. This is a rougher way to locate objects than the original one used for training MS COCO itself, based on pixel segmentation.

In the following figure, a crowded row has been carefully segmented by defining notable areas in an image and creating a textual description of those areas. In machine learning terms, this translates to assigning a label to every pixel in the image and trying to predict the segmentation class (corresponding to the textual description). Historically this has been done with image processing until ImageNet 2012 when deep learning proved a much more efficient solution.

2012 marked a milestone in computer vision because for the first time a deep learning solution provided many superior results than any technique used before: <q>KRIZHEVSKY, Alex; SUTSKEVER, Ilya; HINTON, Geoffrey E. Imagenet classification with deep convolutional neural networks. In: Advances in neural information processing systems. 2012\. p. 1097-1105</q> ( [https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)).

Image segmentation is particularly useful for various tasks, such as:

*   Highlighting the important objects in an image, for instance in medical applications detecting areas with illness
*   Locating objects in an image so that a robot can pick them up or manipulate them
*   Helping with road scene understanding for self-driving cars or drones to navigate
*   Editing images by automatically extracting portions of an image or removing a background

This kind of annotation is very expensive (hence the reduced number of examples in MS COCO) because it has to be done completely by hand and it requires attention and precision. There are some tools to help with annotating by segmenting an image. You can find a comprehensive list at [https://stackoverflow.com/questions/8317787/image-labelling-and-annotation-tool](https://stackoverflow.com/questions/8317787/image-labelling-and-annotation-tool). However, we can suggest the following two tools, if you want to annotate by segmentation images by yourself:

*   LabelImg [https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)
*   FastAnnotationTool [https://github.com/christopher5106/FastAnnotationTool](https://github.com/christopher5106/FastAnnotationTool)

All these tools can also be used for the much simpler annotation by bounding boxes, and they really can come in handy if you want to retrain a model from MS COCO using a class of your own. (We will mention this again at the end of the chapter):

![](img/c1fdcbaf-6a54-4ee1-8097-2ceb57ac2a70.png)

A pixel segmentation of an image used in MS COCO training phase

# The TensorFlow object detection API

As a way of boosting the capabilities of the research community, Google research scientists and software engineers often develop state-of-the-art models and make them available to the public instead of keeping them proprietary. As described in the Google research blog post, [https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html](https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html) , on October 2016, Google's in-house object detection system placed first in the COCO detection challenge, which is focused on finding objects in images (estimating the chance that an object is in this position) and their bounding boxes (you can read the technical details of their solution at [https://arxiv.org/abs/1611.10012](https://arxiv.org/abs/1611.10012)).

The Google solution has not only contributed to quite a few papers and been put to work in some Google products (Nest Cam - [https://nest.com/cameras/nest-aware/](https://nest.com/cameras/nest-aware/), Image Search - [https://www.blog.google/products/search/now-image-search-can-jump-start-your-search-style/](https://www.blog.google/products/search/now-image-search-can-jump-start-your-search-style/), and Street View - [https://research.googleblog.com/2017/05/updating-google-maps-with-deep-learning.html](https://research.googleblog.com/2017/05/updating-google-maps-with-deep-learning.html)), but has also been released to the larger public as an open source framework built on top of TensorFlow.

The framework offers some useful functions and  these five pre-trained different models (constituting the so-called pre-trained Model Zoo):

*   Single Shot Multibox Detector (SSD) with MobileNets
*   SSD with Inception V2
*   Region-Based Fully Convolutional Networks (R-FCN) with Resnet 101
*   Faster R-CNN with Resnet 101
*   Faster R-CNN with Inception Resnet v2

The models are in growing order of precision in detection and slower speed of execution of the detection process. MobileNets, Inception and Resnet refer to different types of CNN network architectures (MobileNets, as the name suggests, it is the architecture optimized for mobile phones, smaller in size and faster in execution). We have discussed CNN architecture in the previous chapter, so you can refer there for more insight on such architectures. If you need a refresher, this blog post by Joice Xu can help you revise the topic in an easy way: [https://towardsdatascience.com/an-intuitive-guide-to-deep-network-architectures-65fdc477db41](https://towardsdatascience.com/an-intuitive-guide-to-deep-network-architectures-65fdc477db41).

**Single Shot Multibox Detector** (**SSD**),  **Region-Based Fully convolutional networks** (**R-FCN**)  and **Faster Region-based convolutional neural networks** (**Faster R-CNN**) are instead the different models to detect multiple objects in images. In the next paragraph, we are going to explain something about how they effectively work.

Depending on your application, you can decide on the most suitable model for you (you have to experiment a bit), or aggregate results from multiple models in order to get better results (as done by the researchers at Google in order to win the COCO competition).

# Grasping the basics of R-CNN, R-FCN and  SSD models

Even if you have clear in mind how a  CNN can manage to classify an image, it could be less obvious for you how a neural network can localize multiple objects into an image by defining its bounding box (a rectangular perimeter bounding the object itself). The first and easiest solution that you may imagine could be to have a sliding window and apply the CNN on each window, but that could be really computationally expensive for most real-world applications (if you are powering the vision of a self-driving car, you do want it to recognize the obstacle and stop before hitting it).

You can find more about the sliding windows approach for object detection in this blog post by Adrian Rosebrock: [https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/](https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/) that makes an effective example by pairing it with image pyramid.

Though reasonably intuitive, because of its complexity and being computationally cumbersome (exhaustive and working at different image scales), the sliding window has quite a few limits, and an alternative preferred solution has immediately been found in the *region proposal* algorithms. Such algorithms use image segmentation (segmenting, that is dividing the image into areas based on the main color differences between areas themselves) in order to create a tentative enumeration of possible bounding boxes in an image. You can find a detailed explanation of how the algorithm works in this post by Satya Mallik: [https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/](https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/). The point is that region proposal algorithms suggest a limited number of boxes to be evaluated, a much smaller one than the one proposed by an exhaustive sliding windows algorithm. That allowed them to be applied in the first R-CNN, Region-based convolutional neural networks, which worked by:

1.  finding a few hundreds or thousands of regions of interest in the image, thanks to a region proposal algorithm
2.  Process by a CNN each region of interest, in order to create features of each area
3.  Use the features to classify the region by a support vector machine and a linear regression to compute bounding boxes that are more precise.

The immediate evolution of R-CNN was Fast R-CNN which made things even speedier because:

1.  it processed all the image at once with CNN, transformed it and applied the region proposal on the transformation. This cut down the CNN processing from a few thousand calls to a single one.
2.  Instead of using an SVM for classification, it used a soft-max layer and a linear classifier, thus simply extending the CNN instead of passing the data to a different model.

In essence, by using a Fast R-CNN we had again a single classification network characterized by a special filtering and selecting layer, the region proposal layer, based on a non-neural network algorithm. Faster R-CNN even changed that layer, by replacing it with a region proposal neural network. That made the model even more complicated but most effective and faster than any previous method.

R-FCN, anyway, are even faster than Faster R-CNN, because they are fully convolutional networks, that don’t use any fully connected layer after their convolutional layers. They are end-to-end networks: from input by convolutions to output. That simply makes them even faster (they have a much lesser number of weights than CNN with a fully connect layer at their end). But their speed comes at a price, they have not been characterized anymore by image invariance (CNN can figure out the class of an object, no matter how the object is rotated). Faster R-CNN supplements this weakness by a position-sensitive score map, that is a way to check if parts of the original image processed by the FCN correspond to parts of the class to be classified. In easy words, they don’t compare to classes, but to part of classes. For instance, they don’t classify a dog, but a dog-upper-left part, a dog-lower-right-part and so on. This approach allows to figure out if there is a dog in a part of the image, no matter how it is orientated. Clearly, this speedier approach comes at the cost of less precision, because position-sensitive score maps cannot supplement all the original CNN characteristics.

Finally, we have SSD (Single Shot Detector). Here the speed is even greater because the network simultaneously predicts the bounding box location and its class as it processes the image. SSD computes a large number of bounding boxes, by simply skipping the region proposal phase. It just reduces highly-overlapping boxes, but still, it processes the largest number of bounding boxes compared to all the model we mentioned up-so-far. Its speed is because as it delimits each bounding box it also classifies it: by doing everything in one shot, it has the fastest speed, though performs in a quite comparable way.

Another short article by Joice Xu can provide you with more details on the detection models we discussed up so far: [https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)

Summing up all the discussion, in order to choose the network you have to consider that you are combining different CNN architectures in classification power and network complexity and different detection models. It is their combined effect to determinate the capability of the network to spot objects, to correctly classify them, and to do all that in a timely fashion.

If you desire to have more reference in regard to the speed and precision of the models we have briefly explained, you can consult: *Speed/accuracy trade-offs for modern convolutional object detectors*. Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z, Song Y, Guadarrama S, Murphy K, CVPR 2017: [http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_SpeedAccuracy_Trade-Offs_for_CVPR_2017_paper.pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_SpeedAccuracy_Trade-Offs_for_CVPR_2017_paper.pdf) Yet, we cannot but advise to just test them in practice for your application, evaluating is they are good enough for the task and if they execute in a reasonable time. Then it is just a matter of a trade-off you have to best decide for your application.

# Presenting our project plan

Given such a powerful tool made available by TensorFlow, our plan is to leverage its API by creating a class you can use for annotating images both visually and in an external file. By annotating, we mean the following:

*   Pointing out the objects in an image (as recognized by a model trained on MS COCO)
*   Reporting the level of confidence in the object recognition (we will consider only objects above a minimum probability threshold, which is set to 0.25, based on the *speed/accuracy trade-offs for modern convolutional object detector*s discussed in the paper previously mentioned)
*   Outputting the coordinates of two opposite vertices of the bounding box for each image
*   Saving all such information in a text file in JSON format
*   Visually representing the bounding box on the original image, if required

In order to achieve such objectives, we need to:

1.  Download one of the pre-trained models (available in `.pb` format - [protobuf](https://developers.google.com/protocol-buffers/)) and make it available in-memory as a TensorFlow session.
2.  Reformulate the helper code provided by TensorFlow in order to make it easier to load labels, categories, and visualization tools by a class that can be easily imported into your scripts.
3.  Prepare a simple script to demonstrate its usage with single images, videos, and videos captured from a webcam.

We start by setting up an environment suitable for the project.

# Setting up an environment suitable for the project

You don't need any specialized environment in order to run the project, though we warmly suggest installing Anaconda `conda` and creating a separated environment for the project. The instructions to run if `conda` is available on your system are as follows:

```py
conda create -n TensorFlow_api python=3.5 numpy pillow
activate TensorFlow_api
```

After activating the environment, you can install some other packages that require a `pip install` command or a `conda install` command pointing to another repository (`menpo`, `conda-forge`):

```py
pip install TensorFlow-gpu
conda install -c menpo opencv
conda install -c conda-forge imageio
pip install tqdm, moviepy
```

In case you prefer another way of running this project, just consider that you need `numpy`, `pillow`, `TensorFlow`, `opencv`, `imageio`, `tqdm`, and `moviepy` in order to run it successfully.

For everything to run smoothly, you also need to create a directory for your project and to save in it the `object_detection` directory of the TensorFlow object detection API project ([https://github.com/tensorflow/models/tree/master/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)).

You can simply obtain that by using the `git` command on the entire TensorFlow models' project and selectively pulling only that directory. This is possible if your Git version is 1.7.0 (February 2012) or above:

```py
mkdir api_project
cd api_project
git init
git remote add -f origin https://github.com/tensorflow/models.git
```

These commands will fetch all the objects in the TensorFlow models project, but it won't check them out. By following those previous commands by:

```py
git config core.sparseCheckout true
echo "research/object_detection/*" >> .git/info/sparse-checkout
git pull origin master
```

You will now have only the `object_detection` directory and its contents as *checked out* on your filesystem and no other directories or files present.

Just keep in mind that the project will need to access the `object_detection` directory, thus you will have to keep the project script in the very same directory of `object_detection` directory. In order to use the script outside of its directory, you will need to access it using a full path.

# Protobuf compilation

The TensorFlow object detection API uses *protobufs*, protocol buffers -- Google's data interchange format ([https://github.com/google/protobuf](https://github.com/google/protobuf)), to configure the models and their training parameters. Before the framework can be used, the protobuf libraries must be compiled, and that requires different steps if you are in a Unix (Linux or Mac) or Windows OS environment.

# Windows installation

First, unpack the [protoc-3.2.0-win32.zip](https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-win32.zip) that can be found at [https://github.com/google/protobuf/releases](https://github.com/google/protobuf/releases) into the project folder. Now you should have a new `protoc-3.4.0-win32` directory, containing a `readme.txt` and two directories, `bin`, and `include`. The folders contain a precompiled binary version of the protocol buffer compiler (*protoc*). All you have to do is add the `protoc-3.4.0-win32` directory to the system path.

After adding it to the system path, you can execute the following command:

```py
protoc-3.4.0-win32/bin/protoc.exe object_detection/protos/*.proto --python_out=.
```

That should be enough to allow the TensorFlow object detection API to work on your computer.

# Unix installation

For Unix environments, the installation procedure can be done using shell commands, just follow the instructions available at [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

# Provisioning of the project code

We start scripting our project in the file `tensorflow_detection.py` by loading the necessary packages:

```py
import os
import numpy as np
import tensorflow as tf
import six.moves.urllib as urllib
import tarfile
from PIL import Image
from tqdm import tqdm
from time import gmtime, strftime
import json
import cv2
```

In order to be able to process videos, apart from OpenCV 3, we also need the `moviepy` package. The package `moviepy` is a project that can be found at [http://zulko.github.io/moviepy/](http://zulko.github.io/moviepy/) and freely used since it is distributed with an MIT license. As described on its home page, `moviepy` is a tool for video editing (that is cuts, concatenations, title insertions), video compositing (non-linear editing), video processing, or to create advanced effects.

The package operates with the most common video formats, including the GIF format. It needs the `FFmpeg` converter ([https://www.ffmpeg.org/](https://www.ffmpeg.org/)) in order to properly operate, therefore at its first usage it will fail to start and will download `FFmpeg` as a plugin using `imageio`:

```py
try:
    from moviepy.editor import VideoFileClip
except:
    # If FFmpeg (https://www.ffmpeg.org/) is not found 
    # on the computer, it will be downloaded from Internet 
    # (an Internet connect is needed)
    import imageio
    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip
```

Finally, we require two useful functions available in the `object_detection` directory from the TensorFlow API project:

```py
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
```

We define the `DetectionObj` class and its `init` procedure. The initialization expects only a parameter and the model name (which is initially set to the less well performing, but faster and more lightweight model, the SSD MobileNet), but a few internal parameters can be changed to suit your use of the class:

*   `self.TARGET_PATH` pointing out the directory where you want the processed annotations to be saved.
*   `self.THRESHOLD` fixing the probability threshold to be noticed by the annotation process. In fact, any model of the suit will output many low probability detections in every image. Objects with too low probabilities are usually false alarms, for such reasons you fix a threshold and ignore such highly unlikely detection. As a rule of thumb, 0.25 is a good threshold in order to spot uncertain objects due to almost total occlusion or visual clutter.

```py
class DetectionObj(object):
    """
    DetectionObj is a class suitable to leverage 
    Google Tensorflow detection API for image annotation from   
    different sources: files, images acquired by own's webcam,
    videos.
    """

    def __init__(self, model='ssd_mobilenet_v1_coco_11_06_2017'):
        """
        The instructions to be run when the class is instantiated
        """

        # Path where the Python script is being run
        self.CURRENT_PATH = os.getcwd()

        # Path where to save the annotations (it can be modified)
        self.TARGET_PATH = self.CURRENT_PATH

        # Selection of pre-trained detection models
        # from the Tensorflow Model Zoo
        self.MODELS = ["ssd_mobilenet_v1_coco_11_06_2017",
                       "ssd_inception_v2_coco_11_06_2017",
                       "rfcn_resnet101_coco_11_06_2017",
                       "faster_rcnn_resnet101_coco_11_06_2017",
                       "faster_rcnn_inception_resnet_v2_atrous_\
                        coco_11_06_2017"]

        # Setting a threshold for detecting an object by the models
        self.THRESHOLD = 0.25 # Most used threshold in practice

        # Checking if the desired pre-trained detection model is available
        if model in self.MODELS:
            self.MODEL_NAME = model
        else:
            # Otherwise revert to a default model
            print("Model not available, reverted to default", self.MODELS[0])
            self.MODEL_NAME = self.MODELS[0]

        # The file name of the Tensorflow frozen model
        self.CKPT_FILE = os.path.join(self.CURRENT_PATH, 'object_detection',
                                      self.MODEL_NAME,  
                                      'frozen_inference_graph.pb')

        # Attempting loading the detection model, 
        # if not available on disk, it will be 
        # downloaded from Internet
        # (an Internet connection is required)
        try:
            self.DETECTION_GRAPH = self.load_frozen_model()
        except:
            print ('Couldn\'t find', self.MODEL_NAME)
            self.download_frozen_model()
            self.DETECTION_GRAPH = self.load_frozen_model()

        # Loading the labels of the classes recognized by the detection model
        self.NUM_CLASSES = 90
        path_to_labels = os.path.join(self.CURRENT_PATH,
                                    'object_detection', 'data',
                                       'mscoco_label_map.pbtxt')
        label_mapping = \ 
                    label_map_util.load_labelmap(path_to_labels)
        extracted_categories = \
                 label_map_util.convert_label_map_to_categories(
                 label_mapping, max_num_classes=self.NUM_CLASSES,                                              
                                           use_display_name=True)
        self.LABELS = {item['id']: item['name'] \
                       for item in extracted_categories}
        self.CATEGORY_INDEX = label_map_util.create_category_index\
(extracted_categories)

        # Starting the tensorflow session
        self.TF_SESSION = tf.Session(graph=self.DETECTION_GRAPH)
```

As a convenient variable to have access to, you have the `self.LABELS` containing a dictionary relating a class numerical code to its textual representation. Moreover, the `init` procedure will have the `TensorFlow` session loaded, open, and ready to be used at `self.TF_SESSION`.

The functions `load_frozen_model` and `download_frozen_model` will help the `init` procedure to load the chosen frozen model from disk and, if not available, will help to download it as a TAR file from the internet and unzip it in the proper directory (which is `object_detection`):

```py
def load_frozen_model(self):
    """
    Loading frozen detection model in ckpt 
    file from disk to memory
    """

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(self.CKPT_FILE, 'rb') as fid:
             serialized_graph = fid.read()
             od_graph_def.ParseFromString(serialized_graph)
             tf.import_graph_def(od_graph_def, name='')

    return detection_graph
```

The function `download_frozen_model` leverages the `tqdm` package in order to visualize its progress as it downloads the new models from the internet. Some models are quite large (over 600 MB) and it may take a long time. Providing visual feedback on the progress and estimated time of completion will allow the user to be more confident about the progression of the operations:

```py
def download_frozen_model(self):
    """
    Downloading frozen detection model from Internet
    when not available on disk
    """
    def my_hook(t):
        """
        Wrapping tqdm instance in order to monitor URLopener 
        """
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner

    # Opening the url where to find the model
    model_filename = self.MODEL_NAME + '.tar.gz'
    download_url = \
        'http://download.tensorflow.org/models/object_detection/'
    opener = urllib.request.URLopener()

    # Downloading the model with tqdm estimations of completion
    print('Downloading ...')
    with tqdm() as t:
        opener.retrieve(download_url + model_filename,
                        model_filename, reporthook=my_hook(t))

    # Extracting the model from the downloaded tar file
    print ('Extracting ...')
    tar_file = tarfile.open(model_filename)
    for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file,
                     os.path.join(self.CURRENT_PATH,  
                                  'object_detection'))
```

The following two functions, `load_image_from_disk` and `load_image_into_numpy_array`, are necessary in order to pick an image from disk and transform it into a Numpy array suitable for being processed by any of the TensorFlow models available in this project:

```py
    def load_image_from_disk(self, image_path):

        return Image.open(image_path)

    def load_image_into_numpy_array(self, image):

        try:
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)
        except:
            # If the previous procedure fails, we expect the
            # image is already a Numpy ndarray
            return image
```

The `detect` function, instead, is the core of the classification functionality of the class. The function just expects lists of images to be processed. A Boolean flag, `annotate_on_image`, just tells the script to visualize the bounding box and the annotation directly on the provided images.

Such a function is able to process images of different sizes, one after the other, but it necessitates processing each one singularly. Therefore, it takes each image and expands the dimension of the array, adding a further dimension. This is necessary because the model expects an array of size: number of images * height * width * depth.

Note, we could pack all the batch images to be predicted into a single matrix. That would work fine, and it would be faster if all the images were of the same height and width, which is an assumption that our project does not make, hence the single image processing.

We then take a few tensors in the model by name (`detection_boxes`, `detection_scores`, `detection_classes`, `num_detections`), which are exactly the outputs we expect from the model, and we feed everything to the input tensor, `image_tensor`, which will normalize the image in a suitable form for the layers of the model to process.

The results are gathered into a list and the images are processed with the detection boxes and represented if required:

```py
def detect(self, images, annotate_on_image=True):
        """
        Processing a list of images, feeding it 
        into the detection model and getting from it scores,  
        bounding boxes and predicted classes present 
        in the images
        """
        if type(images) is not list:
            images = [images]
        results = list()
        for image in images:
            # the array based representation of the image will 
            # be used later in order to prepare the resulting
            # image with boxes and labels on it.
            image_np = self.load_image_into_numpy_array(image)

            # Expand dimensions since the model expects images
            # to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = \ 
                  self.DETECTION_GRAPH.get_tensor_by_name(  
                                               'image_tensor:0')

            # Each box represents a part of the image where a 
            # particular object was detected.
            boxes = self.DETECTION_GRAPH.get_tensor_by_name(
                                            'detection_boxes:0')

            # Each score represent how level of confidence 
            # for each of the objects. Score could be shown 
            # on the result image, together with the class label.
            scores = self.DETECTION_GRAPH.get_tensor_by_name(
                                           'detection_scores:0')
            classes = self.DETECTION_GRAPH.get_tensor_by_name(
                                          'detection_classes:0')
            num_detections = \
                     self.DETECTION_GRAPH.get_tensor_by_name(
                                             'num_detections:0')

         # Actual detection happens here
         (boxes, scores, classes, num_detections) = \
                     self.TF_SESSION.run(
                     [boxes, scores, classes, num_detections],
                     feed_dict={image_tensor: image_np_expanded})

        if annotate_on_image:
            new_image = self.detection_on_image(
                            image_np, boxes, scores, classes)
            results.append((new_image, boxes, 
                            scores, classes, num_detections))
        else:
            results.append((image_np, boxes, 
                            scores, classes, num_detections))
        return results
```

The function `detection_on_image` just processes the results from the `detect` function and returns a new image enriched by bounding boxes which will be represented on screen by the function `visualize_image` (You can adjust the latency parameter, which corresponds to the seconds the image will stay on screen before the script passes to process another image).

```py
    def detection_on_image(self, image_np, boxes, scores,  
                           classes):
        """
        Put detection boxes on the images over 
        the detected classes
        """
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.CATEGORY_INDEX,
            use_normalized_coordinates=True,
            line_thickness=8)
        return image_np
```

The function `visualize_image` offers a few parameters that could be modified in order to suit your needs in this project. First of all, `image_size` provides the desired size of the image to be represented on screen. Larger or shorter images are therefore modified in order to partially resemble this prescribed size. The `latency` parameter, instead, will define the time in seconds that each image will be represented on the screen, thus locking the object detection procedure, before moving to the next one. Finally, the `bluish_correction` is just a correction to be applied when images are offered in the **BGR** format (in this format the color channels are arranged in the order: **blue-green-red** and it is the standard for the OpenCV library: [https://stackoverflow.com/questions/14556545/why-opencv-using-bgr-colour-space-instead-of-rgb](https://stackoverflow.com/questions/14556545/why-opencv-using-bgr-colour-space-instead-of-rgb)) , instead of the **RGB** (**red-green-blue**), which is the image format the model is expecting:

```py
    def visualize_image(self, image_np, image_size=(400, 300), 
                        latency=3, bluish_correction=True):

        height, width, depth = image_np.shape
        reshaper = height / float(image_size[0])
        width = int(width / reshaper)
        height = int(height / reshaper)
        id_img = 'preview_' + str(np.sum(image_np))
        cv2.startWindowThread()
        cv2.namedWindow(id_img, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(id_img, width, height)
        if bluish_correction:
            RGB_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            cv2.imshow(id_img, RGB_img)
        else:
            cv2.imshow(id_img, image_np)
        cv2.waitKey(latency*1000)
```

Annotations are prepared and written to disk by the `serialize_annotations` function, which will create single JSON files containing, for each image, the data regarding the detected classes, the vertices of the bounding boxes, and the detection confidence. For instance, this is the result from a detection on a dog's photo:

```py
"{"scores": [0.9092628359794617], "classes": ["dog"], "boxes": [[0.025611668825149536, 0.22220897674560547, 0.9930437803268433, 0.7734537720680237]]}"
```

The JSON points out the detected class, a single dog, the level of confidence (about 0.91 confidence), and the vertices of the bounding box, and expresses as percentages the height and width of the image (they are therefore relative, not absolute pixel points):

```py
    def serialize_annotations(self, boxes, scores, classes,                                       filename='data.json'):
        """
        Saving annotations to disk, to a JSON file
        """

        threshold = self.THRESHOLD
        valid = [position for position, score in enumerate(
                                              scores[0]) if score > threshold]
        if len(valid) > 0:
            valid_scores = scores[0][valid].tolist()
            valid_boxes  = boxes[0][valid].tolist()
            valid_class = [self.LABELS[int(
                                   a_class)] for a_class in classes[0][valid]]
            with open(filename, 'w') as outfile:
                json_data = {'classes': valid_class,
                    'boxes':valid_boxes, 'scores': valid_scores})
                json.dump(json_data, outfile)
```

The function `get_time` conveniently transforms the actual time into a string that can be used in a filename:

```py
    def get_time(self):
        """
        Returning a string reporting the actual date and time
        """
        return strftime("%Y-%m-%d_%Hh%Mm%Ss", gmtime())
```

Finally, we prepare three detection pipelines, for images, videos, and webcam. The pipeline for images loads each image into a list. The pipeline for videos lets the `VideoFileClip` module from `moviepy` do all the heavy lifting after simply passing the `detect` function appropriately wrapped in the `annotate_photogram` function. Finally, the pipeline for webcam capture relies on a simple `capture_webcam` function that, based on OpenCV's VideoCapture, records a number of snapshots from the webcam returning just the last (the operation takes into account the time necessary for the webcam before adjusting to the light levels of the environment):

```py
    def annotate_photogram(self, photogram):
        """
        Annotating a video's photogram with bounding boxes
        over detected classes
        """
        new_photogram, boxes, scores, classes, num_detections =  
                                                      self.detect(photogram)[0]
        return new_photogram
```

The `capture_webcam` function will acquire an image from your webcam using the `cv2.VideoCapture` functionality ([http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html](http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html)) . As webcams have first to adjusts to the light conditions present in the environment where the picture is taken, the procedure discards a number of initial shots, before taking the shot that will be used in the object detection procedure. In this way, the webcam has all the time to adjust its light settings, :

```py
    def capture_webcam(self):
        """
        Capturing an image from the integrated webcam
        """

        def get_image(device):
            """
            Internal function to capture a single image  
            from the camera and return it in PIL format
            """

            retval, im = device.read()
            return im

        # Setting the integrated webcam
        camera_port = 0

        # Number of frames to discard as the camera 
        # adjusts to the surrounding lights
        ramp_frames = 30

        # Initializing the webcam by cv2.VideoCapture
        camera = cv2.VideoCapture(camera_port)

        # Ramping the camera - all these frames will be 
        # discarded as the camera adjust to the right light levels
        print("Setting the webcam")
        for i in range(ramp_frames):
            _ = get_image(camera)

        # Taking the snapshot
        print("Now taking a snapshot ... ", end='')
        camera_capture = get_image(camera)
        print('Done')

        # releasing the camera and making it reusable
        del (camera)
        return camera_capture
```

The `file_pipeline` comprises all the steps necessary to load images from storage and visualize/annotate them:

1.  Loading images from disk.
2.  Applying object detection on the loaded images.
3.  Writing the annotations for each image in a JSON file.
4.  If required by the Boolean parameter `visualize`, represent the images with its bounding boxes on the computer's screen:

```py
    def file_pipeline(self, images, visualize=True):
        """
        A pipeline for processing and annotating lists of
        images to load from disk
        """
        if type(images) is not list:
            images = [images]
        for filename in images:
            single_image = self.load_image_from_disk(filename)
            for new_image, boxes, scores, classes, num_detections in  
                                                     self.detect(single_image):
                self.serialize_annotations(boxes, scores, classes,
                                           filename=filename + ".json")
                if visualize:
                    self.visualize_image(new_image)
```

The `video_pipeline` simply arranges all the steps necessary to annotate a video with bounding boxes and, after completing the operation, saves it to disk:

```py
    def video_pipeline(self, video, audio=False):
        """
        A pipeline to process a video on disk and annotating it
        by bounding box. The output is a new annotated video.
        """
        clip = VideoFileClip(video)
        new_video = video.split('/')
        new_video[-1] = "annotated_" + new_video[-1]
        new_video = '/'.join(new_video)
        print("Saving annotated video to", new_video)
        video_annotation = clip.fl_image(self.annotate_photogram)
        video_annotation.write_videofile(new_video, audio=audio)
```

The `webcam_pipeline` is the function that arranges all the steps when you want to annotate an image acquired from your webcam:

1.  Captures an image from the webcam.
2.  Saves the captured image to disk (using `cv2.imwrite` which has the advantage of writing different image formats based on the target filename, see at: [http://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html](http://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html)
3.  Applies object detection on the image.
4.  Saves the annotation JSON file.
5.  Represents visually the image with bounding boxes:

```py
    def webcam_pipeline(self):
        """
        A pipeline to process an image acquired by the internal webcam
        and annotate it, saving a JSON file to disk
        """
        webcam_image = self.capture_webcam()
        filename = "webcam_" + self.get_time()
        saving_path = os.path.join(self.CURRENT_PATH, filename + ".jpg")
        cv2.imwrite(saving_path, webcam_image)
        new_image, boxes, scores, classes, num_detections =  
                                            self.detect(webcam_image)[0]
        json_obj = {'classes': classes, 'boxes':boxes, 'scores':scores}
        self.serialize_annotations(boxes, scores, classes,                                   filename=filename+".json")
        self.visualize_image(new_image, bluish_correction=False)
```

# Some simple applications

As a concluding paragraph of the code provisioning, we demonstrate just three simple scripts leveraging the three different sources used by our project: files, videos, webcam.

Our first testing script aims at annotating and visualizing three images after importing the class `DetectionObj` from the local directory (In cases where you operate from another directory, the import won't work unless you add the project directory to the Python path).

In order to add a directory to the Python path in your script, you just have to put `sys.path.insert` command before the part of the script that needs access to that directory:

`import sys`
`sys.path.insert(0,'/path/to/directory')`

Then we activate the class, declaring it using the SSD MobileNet v1 model. After that, we have to put the path to every single image into a list and feed it to the method `file_pipeline`:

```py
from TensorFlow_detection import DetectionObj
if __name__ == "__main__":
    detection = DetectionObj(model='ssd_mobilenet_v1_coco_11_06_2017')
    images = ["./sample_images/intersection.jpg",
              "./sample_images/busy_street.jpg", "./sample_images/doge.jpg"]
    detection.file_pipeline(images)
```

The output that we receive after our detection class has been placed on the intersection image and will return us another image enriched with bounding boxes around objects recognized with enough confidence:

![](img/1ec56e41-57e4-43c2-b6ac-b96c25af2b86.png)

Object detection by SSD MobileNet v1 on a photo of an intersection

After running the script, all three images will be represented with their annotations on the screen (each one for three seconds) and a new JSON file will be written on disk (in the target directory, which corresponds to the local directory if you have not otherwise stated it by modifying the class variable `TARGET_CLASS`).

In the visualization, you will see all the bounding boxes relative to objects whose prediction confidence is above 0.5\. Anyway, you will notice that, in this case of an annotated image of an intersection (depicted in the preceding figure), not all cars and pedestrians have been spotted by the model.

By looking at the JSON file, you will discover that many other cars and pedestrians have been located by the model, though with lesser confidence. In the file, you will find all the objects detected with at least 0.25 confidence, a threshold which represents a common standard in many studies on object detection (but you can change it by modifying the class variable `THRESHOLD`).

Here you can see the scores generated in the JSON file. Only eight detected objects are above the visualization threshold of 0.5, whereas 16 other objects have lesser scores:

```py
"scores": [0.9099398255348206, 0.8124723434448242, 0.7853631973266602, 0.709653913974762, 0.5999227166175842, 0.5942907929420471, 0.5858771800994873, 0.5656214952468872, 0.49047672748565674, 0.4781857430934906, 0.4467884600162506, 0.4043623208999634, 0.40048354864120483, 0.38961756229400635, 0.35605812072753906, 0.3488095998764038, 0.3194449841976166, 0.3000411093235016, 0.294520765542984, 0.2912806570529938, 0.2889115810394287, 0.2781482934951782, 0.2767323851585388, 0.2747304439544678]
```

And here you can find the relative class of the detected objects. Many cars have been spotted with lesser confidence. They actually may be cars in the image or errors. In accordance with your application of the Detection API, you may want to adjust your threshold or use another model and estimate an object only if it has been repeatedly detected by different models above a threshold: 

```py
"classes": ["car", "person", "person", "person", "person", "car", "car", "person", "person", "person", "person", "person", "person", "person", "car", "car", "person", "person", "car", "car", "person", "car", "car", "car"]
```

Applying detection to videos uses the same scripting approach. This time you just point to the appropriate method, `video_pipeline`, the path to the video, and set whether the resulting video should have audio or not (by default audio will be filtered out). The script will do everything by itself, saving, on the same directory path as the original video, a modified and annotated video (you can spot it because it has the same filename but with the addition of `annotated_` before it):

```py
from TensorFlow_detection import DetectionObj
if __name__ == "__main__":
    detection = DetectionObj(model='ssd_mobilenet_v1_coco_11_06_2017')
    detection.video_pipeline(video="./sample_videos/ducks.mp4", audio=False)
```

Finally, you can also leverage the exact same approach for images acquired by a webcam. This time you will be using the method `webcam_pipeline`:

```py
from TensorFlow_detection import DetectionObj
if __name__ == "__main__":
    detection = DetectionObj(model='ssd_mobilenet_v1_coco_11_06_2017')
    detection.webcam_pipeline()
```

The script will activate the webcam, adjust the light, pick a snapshot, save the resulting snapshot and its annotation JSON file in the current directory, and finally represent the snapshot on your screen with bounding boxes on detected objects.

# Real-time webcam detection

The previous `webcam_pipeline` is not a real-time detection system because it just takes snapshots and applies detection to the single taken image. This is a necessary limitation because dealing with webcam streaming requires intensive I/O data exchange. In particular, the problem is the queue of images arriving from the webcam to the Python interpreter that locks down Python until the transfer is completed. Adrian Rosebrock on his website pyimagesearch proposes a simple solution based on threads that you can read about at this Web address: [http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/](http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/).

The idea is very simple. In Python,  because of the **global interpreter lock** (**GIL**), only one thread can execute at a time. If there is some I/O operation that blocks the thread (such as downloading a file or getting an image from the webcam), all the remaining commands are just delayed for it to complete causing a very slow execution of the program itself. It is then a good solution to move the blocking I/O operation to another thread. Since threads share the same memory, the program thread can proceed with its instructions and inquiry from time to time the I/O thread in order to check if it has completed its operations.  Therefore, if moving images from the webcam to the memory of the program is a blocking operation, letting another thread dealing with I/O could be the solution. The main program will just inquiry the I/O thread, pick the image from a buffer containing only the latest received image and plot it on the screen.  

```py
from tensorflow_detection import DetectionObj
from threading import Thread
import cv2

def resize(image, new_width=None, new_height=None):
    """
    Resize an image based on a new width or new height
    keeping the original ratio
    """
    height, width, depth = image.shape
    if new_width:
        new_height = int((new_width / float(width)) * height)
    elif new_height:
        new_width = int((new_height / float(height)) * width)
    else:
        return image
    return cv2.resize(image, (new_width, new_height), \
                      interpolation=cv2.INTER_AREA)

class webcamStream:
    def __init__(self):
        # Initialize webcam
        self.stream = cv2.VideoCapture(0)
        # Starting TensorFlow API with SSD Mobilenet
        self.detection = DetectionObj(model=\
                        'ssd_mobilenet_v1_coco_11_06_2017')
        # Start capturing video so the Webca, will tune itself
        _, self.frame = self.stream.read()
        # Set the stop flag to False
        self.stop = False
        #
        Thread(target=self.refresh, args=()).start()

    def refresh(self):
        # Looping until an explicit stop is sent 
        # from outside the function
        while True:
            if self.stop:
                return
            _, self.frame = self.stream.read()

    def get(self):
        # returning the annotated image
        return self.detection.annotate_photogram(self.frame)

    def halt(self):
        # setting the halt flag
        self.stop = True

if __name__ == "__main__":
    stream = webcamStream()
    while True:
        # Grabbing the frame from the threaded video stream 
        # and resize it to have a maximum width of 400 pixels
        frame = resize(stream.get(), new_width=400)
        cv2.imshow("webcam", frame)
        # If the space bar is hit, the program will stop
        if cv2.waitKey(1) & 0xFF == ord(" "):
            # First stopping the streaming thread
            stream.halt()
            # Then halting the while loop
            break
```

The above code implements this solution using a `webcamStream` class that instantiates a thread for the webcam I/O, allowing the main Python program to always have at hand the latest received image, processed by the TensorFlow API (using `ssd_mobilenet_v1_coco_11_06_2017`). The processed image is fluidly plotted on the screen using an `OpenCV` function, listening to the space bar keystroke in order to terminate the program. 

# Acknowledgements

Everything related to this project started from the following paper: *Speed/accuracy trade-offs for modern convolutional object detectors(*[https://arxiv.org/abs/1611.10012](https://arxiv.org/abs/1611.10012)*)* by Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z, Song Y, Guadarrama S, Murphy K, CVPR 2017\. Concluding this chapter, we have to thank all the contributors of the TensorFlow object detection API for their great job programming the API and making it open-source and thus free and accessible to anyone: Jonathan Huang, Vivek Rathod, Derek Chow, Chen Sun, Menglong Zhu, Matthew Tang, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Jasper Uijlings, Viacheslav Kovalevskyi, Kevin Murphy. We also cannot forget to thank Dat Tran for his inspirational posts on medium of two MIT licensed projects on how to use the TensorFlow object detection API for real-time recognition even on custom ([https://towardsdatascience.com/building-a-real-time-object-recognition-app-with-tensorflow-and-opencv-b7a2b4ebdc32](https://towardsdatascience.com/building-a-real-time-object-recognition-app-with-tensorflow-and-opencv-b7a2b4ebdc32) and [https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9))

# Summary

This project has helped you to start immediately classifying objects in images with confidence without much hassle.  It helps you to see what a ConvNet could do for your problem, focusing more on the wrap up (possibly a larger application) you have in mind,and annotating many images for training more ConvNets with fresh images of a selected class.

During the project, you have learned quite a few useful technicalities you can reuse in many projects dealing with images. First of all, you now know how to process different kinds of visual inputs from images, videos, and webcam captures. You also know how to load a frozen model and put it to work, and also how to use a class to access a TensorFlow model.

On the other hand, clearly, the project has some limitations that you may encounter sooner or later, and that may spark the idea to try to integrate your code and make it shine even more. First of all, the models we have discussed will soon be surpassed by newer and more efficient ones (you can check here for newly available models: [https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)), and you will need to incorporate new ones or create your own architecture ([https://github.com/tensorflow/models/blob/master/object_detection/g3doc/defining_your_own_model.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/defining_your_own_model.md)). Then you may need to combine the model to reach the accuracy you need in your project (the paper *Speed/accuracy trade-offs for modern convolutional object detectors* reveals how researchers at Google have done it). Finally,  you may need to tune a ConvNet to recognize a new class (you can read how to do that here, but beware, it is a long process and a project by itself: [https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md)).

In the next chapter, we will look at state-of-the-art object detection in images, devising a project that will lead you to produce complete discursive captions describing submitted images, not just simple labels and bounding boxes.
