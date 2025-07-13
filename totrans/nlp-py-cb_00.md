# Preface

Dear reader, thank you for choosing this book to pursue your interest in natural language processing. This book will give you a practical viewpoint to understand and implement NLP solutions from scratch. We will take you on a journey that will start with accessing inbuilt data sources and creating your own sources. And then you will be writing complex NLP solutions that will involve text normalization, preprocessing, POS tagging, parsing, and much more.

In this book, we will cover the various fundamentals necessary for applications of deep learning in natural language processing, and they are state-of-the-art techniques. We will discuss applications of deep learning using Keras software.

This book is motivated by the following goals:

*   The content is designed to help newbies get up to speed with various fundamentals explained in a detailed way; and for experienced professionals, it will refresh various concepts to get more clarity when applying algorithms to chosen data
*   There is an introduction to new trends in the applications of deep learning in NLP

# What this book covers

[Chapter 1](2220360f-3247-4028-b812-5a238d3804c7.xhtml), *Corpus and WordNet*, teaches you access to built-in corpora of NLTK and frequency distribution. We shall also learn what WordNet is and explore its features and usage.

[Chapter 2](dbb8fe87-d853-4e97-8812-75849452307b.xhtml), *Raw Text, Sourcing, and Normalization*, shows how to extract text from various formats of data sources. We will also learn to extract raw text from web sources. And finally we will normalize raw text from these heterogeneous sources and organize it in corpus.

[Chapter 3](114762c5-3506-47a2-a25f-8e8b55c0b30d.xhtml), *Pre-Processing*, introduces some critical preprocessing steps, such as tokenization, stemming, lemmatization, and edit distance.

[Chapter 4](bdfe8ef1-c7dd-42ff-895d-84c90c5ccfb3.xhtml), *Regular Expressions*, covers one of the most basic and simple, yet most important and powerful, tools that you will ever learn. In this chapter, you will learn the concept of pattern matching as a way to do text analysis, and for this, there is no better tool than regular expressions.

[Chapter 5](d81001da-02ad-4499-a128-913770e0833e.xhtml), *POS Tagging and Grammars*. POS tagging forms the basis of any further syntactic analyses, and grammars can be formed and deformed using POS tags and chunks. We will learn to use and write our own POS taggers and grammars.

[Chapter 6](21c490ed-f6ea-4336-8c44-814f2390de55.xhtml), *Chunking, Sentence Parse, and Dependencies, h*elps you to learn how to use the inbuilt chunker as well as train/write your own chunker: dependency parser. In this chapter, you will learn to evaluate your own trained models.

[Chapter 7](e2c9e657-c1bd-4f43-a33e-1392dc4efc3f.xhtml), *Information Extraction and Text Classification,* tells you more about named entities recognition. We will be using inbuilt NEs and also creating your own named entities using dictionaries. Let's learn to use inbuilt text classification algorithms and simple recipes around its application.

[Chapter 8](997e2fbe-1c09-4f31-ac5d-8cc4498dc6e1.xhtml), *Advanced NLP Recipes,* is about combining all your lessons so far and creating applicable recipes that can be easily plugged into any of your real-life application problems. We will write recipes such as text similarity, summarization, sentiment analysis, anaphora resolution, and so on.

[Chapter 9](6c909c5d-cd28-40c2-a1c0-60aacd2fd2cf.xhtml), *Application of Deep Learning in NLP*, presents the various fundamentals necessary for working on deep learning with applications in NLP problems such as classification of emails, sentiment classification with CNN and LSTM, and finally visualizing high-dimensional words in low dimensional space.

[Chapter 10](585767f4-c71d-437e-80f0-42d4c051ab33.xhtml), *Advanced Application of Deep Learning in NLP*, describes state-of-the-art problem solving using deep learning. This consists of automated text generation, question and answer on episodic data, language modeling to predict the next best word, and finally chatbot development using generative principles.

# What you need for this book

To perform the recipes of this book successfully, you will need Python 3.x or higher running on any Windows- or Unix-based operating system with a processor of 2.0 GHz or higher and minimum 4 GB RAM. As far as the IDE for Python development are concerned, there are many available in the market but my personal favorite is PyCharm community edition. It's free, it's open source, and it's developed by Jetbrains. That means support is excellent, advancement and fixes are distributed at a steady pace, and familiarity with IntelliJ keeps the learning curve pretty flat.

This book assumes you know Keras's basics and how to install the libraries. We do not expect that readers are already equipped with knowledge of deep learning and mathematics, such as linear algebra and so on.

We have used the following versions of software throughout this book, but it should run fine with any of the more recent ones also:

*   Anaconda 3 – 4.3.1 (all Python and its relevant packages are included in Anaconda, Python – 3.6.1, NumPy – 1.12.1, pandas – 0.19.2)
*   Theano – 0.9.0
*   Keras – 2.0.2
*   feedparser – 5.2.1
*   bs4 – 4.6.0
*   gensim – 3.0.1

# Who this book is for

This book is intended for data scientists, data analysts, and data science professionals who want to upgrade their existing skills to implement advanced text analytics using NLP. Some basic knowledge of natural language processing is recommended.

This book is intended for any newbie with no knowledge of NLP or any experienced professional who would like to expand their knowledge from traditional NLP techniques to state-of-the-art deep learning techniques in the application of NLP.

# Sections

In this book, you will find several headings that appear frequently (Getting ready, How to do it…, How it works…, There's more…, and See also). To give clear instructions on how to complete a recipe, we use these sections as follows.

# Getting ready

This section tells you what to expect in the recipe, and describes how to set up any software or any preliminary settings required for the recipe.

# How to do it…

This section contains the steps required to follow the recipe.

# How it works…

This section usually consists of a detailed explanation of what happened in the previous section.

# There's more…

This section consists of additional information about the recipe in order to make the reader more knowledgeable about the recipe.

# See also

This section provides helpful links to other useful information for the recipe.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "Create a new file named `reuters.py` and add the following import line in the file" A block of code is set as follows:

```py
for w in reader.words(fileP):
  print(w + ' ', end='')
  if (w is '.'):
    print()
```

Any command-line input or output is written as follows:

```py
# Deep Learning modules
>>> import numpy as np
>>> from keras.models import Sequential
```

**New terms** and **important words** are shown in bold.

Warnings or important notes appear like this.

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book-what you liked or disliked. Reader feedback is important for us as it helps us develop titles that you will really get the most out of. To send us general feedback, simply e-mail `feedback@packtpub.com`, and mention the book's title in the subject of your message. If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide at [www.packtpub.com/authors](http://www.packtpub.com/authors) .

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

# Downloading the example code

You can download the example code files for this book from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you. You can download the code files by following these steps:

1.  Log in or register to our website using your e-mail address and password.
2.  Hover the mouse pointer on the SUPPORT tab at the top.
3.  Click on Code Downloads & Errata.
4.  Enter the name of the book in the Search box.
5.  Select the book for which you're looking to download the code files.
6.  Choose from the drop-down menu where you purchased this book from.
7.  Click on Code Download.

You can also download the code files by clicking on the Code Files button on the book's webpage at the Packt Publishing website. This page can be accessed by entering the book's name in the Search box. Please note that you need to be logged in to your Packt account. Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR / 7-Zip for Windows
*   Zipeg / iZip / UnRarX for Mac
*   7-Zip / PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/Natural-Language-Processing-with-Python-Cookbook](https://github.com/PacktPublishing/Natural-Language-Processing-with-Python-Cookbook). We also have other code bundles from our rich catalog of books and videos available at **[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**. Check them out!

# Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books-maybe a mistake in the text or the code-we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the Errata Submission Form link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title. To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the Errata section.

# Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy. Please contact us at `copyright@packtpub.com` with a link to the suspected pirated material. We appreciate your help in protecting our authors and our ability to bring you valuable content.

# Questions

If you have a problem with any aspect of this book, you can contact us at `questions@packtpub.com`, and we will do our best to address the problem.

