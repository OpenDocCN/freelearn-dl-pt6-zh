# Preface

TensorFlow is one of the most popular frameworks used for machine learning and, more recently, deep learning. It provides a fast and efficient framework for training different kinds of deep learning models with very high accuracy. This book is your guide to mastering deep learning with TensorFlow with the help of 12 real-world projects.

*TensorFlow Deep Learning Projects* starts with setting up the right TensorFlow environment for deep learning. You'll learn to train different types of deep learning models using TensorFlow, including CNNs, RNNs, LSTMs, and generative adversarial networks. While doing so, you will build end-to-end deep learning solutions to tackle different real-world problems in image processing, enterprise AI, and natural language processing, to name a few. You'll train high-performance models to generate captions for images automatically, predict the performance of stocks, and create intelligent chatbots. Some advanced aspects, such as recommender systems and reinforcement learning, are also covered in this book.

By the end of this book, you will have mastered all the concepts of deep learning and their implementation with TensorFlow, and will be able to build and train your own deep learning models with TensorFlow to tackle any kind of problem.

# Who this book is for

This book is for data scientists, machine learning and deep learning practitioners, and AI enthusiasts who want a go-to guide to test their knowledge and expertise in building real-world intelligent systems. If you want to master the different deep learning concepts and algorithms associated with it by implementing practical projects in TensorFlow, this book is what you need!

# What this book covers

[Chapter 1](8ce252a5-b645-440d-8e60-14fd93a1b64d.xhtml), *Recognizing traffic signs using Convnets,* shows how to extract the proper features from images with all the necessary preprocessing steps. For our convolutional neural network, we will use simple shapes generated with matplotlib. For our image preprocessing exercises, we will use the Yale Face Database.

[Chapter 2](1d7b7046-edb2-4e25-84fb-4871fa9c0ea6.xhtml), *Annotating Images with Object Detection API*, details a the building of a real-time object detection application that can annotate images, videos, and webcam captures using TensorFlow's new object detection API (with its selection of pretrained convolutional networks, the so-called TensorFlow detection model zoo) and OpenCV.

[Chapter 3](d6b4a8de-45e4-4d3a-adac-aceb176da4c8.xhtml), *Caption Generation for Images*, enables readers to learn caption generation with or without pretrained models.

[Chapter 4](9d138f1d-6fea-4c59-9691-be7b18d73c99.xhtml), *Building GANs for Conditional Image Creation*, guides you step by step through building a selective GAN to reproduce new images of the favored kind. The used datasets that GANs will reproduce will be of handwritten characters (both numbers and letters in Chars74K).

[Chapter 5](95c80f98-b5f4-4e07-9054-0c968dae1e76.xhtml), *Stock Price Prediction with LSTM*, explores how to predict the future of a mono-dimensional signal, a stock price. Given its past, we will learn how to forecast its future with an LSTM architecture, and how we can make our prediction's more and more accurate.

[Chapter 6](82354f0c-2256-4384-8270-f0f36c7cba7d.xhtml), *Create and Train Machine Translation Systems*, shows how to create and train a bleeding-edge machine translation system with TensorFlow.

[Chapter 7](d21268f3-324d-4380-9c36-fec5caf82ffb.xhtml), *Train and Set up a Chatbot, Able to Discuss Like a Human*, tells you how to build an intelligent chatbot from scratch and how to *discuss* with it.

[Chapter 8](263aea4b-fe7a-4d9b-a0c8-27ced8b9422b.xhtml), *Detecting Duplicate Quora Questions*, discusses methods that can be used to detect duplicate questions using the Quora dataset. Of course, these methods can be used for other similar datasets.

[Chapter 9](65b5a924-8bd4-413c-9f9c-e1861efd784c.xhtml), *Building a TensorFlow Recommender System*, covers large-scale applications with practical examples. We'll learn how to implement cloud GPU computing capabilities on AWS with very clear instructions. We'll also utilize H2O's wonderful API for deep networks on a large scale.

[Chapter 10](c425ccd5-3ffe-423d-acad-4286a5f756d9.xhtml), *Video Games by Reinforcement Learning*, details a project where you build an AI capable of playing *Lunar Lander* by itself. The project revolves around the existing OpenAI Gym project and integrates it using TensorFlow. OpenAI Gym is a project that provides different gaming environments to explore how to use AI agents that can be powered by, among other algorithms, TensorFlow neural models. 

# To get the most out of this book

The examples covered in this book can be run with Windows, Ubuntu, or Mac. All the installation instructions are covered. You will need basic knowledge of Python, machine learning and deep learning, and familiarity with TensorFlow.

# Download the example code files

You can download the example code files for this book from your account at [www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files emailed directly to you.

You can download the code files by following these steps:

1.  Log in or register at [www.packtpub.com](http://www.packtpub.com/support).
2.  Select the SUPPORT tab.
3.  Click on Code Downloads & Errata.
4.  Enter the name of the book in the Search box and follow the onscreen instructions.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR/7-Zip for Windows
*   Zipeg/iZip/UnRarX for Mac
*   7-Zip/PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/TensorFlow-Deep-Learning-Projects](https://github.com/PacktPublishing/TensorFlow-Deep-Learning-Projects). We also have other code bundles from our rich catalog of books and videos available at **[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**. Check them out!

# Conventions used

There are a number of text conventions used throughout this book.

`CodeInText`: Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. Here is an example: "The class `TqdmUpTo` is just a `tqdm` wrapper that enables the use of the progress display also for downloads."

A block of code is set as follows:

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

Any command-line input or output is written as follows:

```py
epoch 01: precision: 0.064
epoch 02: precision: 0.086
epoch 03: precision: 0.106
epoch 04: precision: 0.127
epoch 05: precision: 0.138
epoch 06: precision: 0.145
epoch 07: precision: 0.150
epoch 08: precision: 0.149
epoch 09: precision: 0.151
epoch 10: precision: 0.152
```

**Bold**: Indicates a new term, an important word, or words that you see onscreen. For example, words in menus or dialog boxes appear in the text like this. Here is an example: "Select System info from the Administration panel."

Warnings or important notes appear like this.

Tips and tricks appear like this.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: Email `feedback@packtpub.com` and mention the book title in the subject of your message. If you have questions about any aspect of this book, please email us at `questions@packtpub.com`.

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you would report this to us. Please visit [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the Errata Submission Form link, and entering the details.

**Piracy**: If you come across any illegal copies of our works in any form on the Internet, we would be grateful if you would provide us with the location address or website name. Please contact us at `copyright@packtpub.com` with a link to the material.

**If you are interested in becoming an author**: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com/).

# Reviews

Please leave a review. Once you have read and used this book, why not leave a review on the site that you purchased it from? Potential readers can then see and use your unbiased opinion to make purchase decisions, we at Packt can understand what you think about our products, and our authors can see your feedback on their book. Thank you!

For more information about Packt, please visit [packtpub.com](https://www.packtpub.com/).