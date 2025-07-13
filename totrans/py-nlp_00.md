# Preface

The book title, **Python Natural Language Processing**, gives you a broad idea about the book. As a reader, you will get the chance to learn about all the aspects of **natural language processing** (**NLP**) from scratch. In this book, I have specified NLP concepts in a very simple language, and there are some really cool practical examples that enhance your understanding of this domain. By implementing these examples, you can improve your NLP skills. Don't you think that sounds interesting?

Now let me answer some of the most common questions I have received from my friends and colleagues about the NLP domain. These questions really inspired me to write this book. For me, it's really important that all my readers understand why I am writing this book. Let's find out!

Here, I would like answer some of the questions that I feel are critical to my readers. So, I'll begin with some of the questions, followed by the answers. The first question I usually get asked is--what is NLP? The second one is--why is Python mainly used for developing NLP applications? And last but not least, the most critical question is--what are the resources I can use for learning NLP? Now let's look at the answers!

The answer to the first question is that NLP, simply put, is the language you speak, write, read, or understand as a human; natural language is, thus, a medium of communication. Using computer science algorithms, mathematical concepts, and statistical techniques, we try to process the language so machines can also understand language as humans do; this is called **NLP**.

Now let's answer the second question--why do people mainly use Python to develop NLP applications? So, there are some facts that I want to share with you. The very simple and straightforward thing is that Python has a lot of libraries that make your life easy when you develop NLP applications. The second reason is that if you are coming from a C or C++ coding background, you don't need to worry about memory leakage. The Python interpreter will handle this for you, so you can just focus on the main coding part. Besides, Python is a coder-friendly language. You can do much more by writing just a few lines of codes, compared to other object-oriented languages. So all these facts drive people to use Python for developing NLP and other data science-related applications for rapid prototyping.

The last question is critical to me because I used to explain the previous answers to my friends, but after hearing all these and other fascinating things, they would come to me and say that they want to learn NLP, so what are the resources available? I used to recommend books, blogs, YouTube videos, education platforms such as Udacity and Coursera, and a lot more, but after a few days, they would ask me if there is a single resource in the form of book, blog, or anything that they could use. Unfortunately, for them, my answer was no. At that stage, I really felt that juggling all these resources would always be difficult for them, and that painful realization became my inspiration to write this book.

So in this book, I have tried to cover most of the essential parts of NLP, which will be useful for everyone. The great news is that I have provided practical examples using Python so readers can understand all the concepts theoretically as well as practically. Reading, understanding, and coding are the three main processes that I have followed in this book to make readers lives easier.

# What this book covers

[Chapter 1](238411c1-88cf-4377-a6c6-7451e0f48daa.xhtml), *Introduction*, provides an introduction to NLP and the various branches involved in the NLP domain. We will see the various stages of building NLP applications and discuss NLTK installation.

[Chapter 2](837b4260-d60f-474a-a208-68a1eaa8e1bb.xhtml), *Practical Understanding of Corpus and Dataset*, shows all the aspects of corpus analysis. We will see the different types of corpus and data attributes present in corpuses. We will touch upon different corpus formats such as CSV, JSON, XML, LibSVM, and so on. We will see a web scraping example.

[Chapter 3](f65e61fc-1d20-434f-b606-f36cf401fc41.xhtml), *Understanding Structure of Sentences*, helps you understand the most essential aspect of natural language, which is linguistics. We will see the concepts of lexical analysis, syntactic analysis, semantic analysis, handling ambiguities, and so on. We will use NLTK to understand all the concepts practically.

[Chapter 4](59dc896d-b0de-44c9-9cbe-18c63e510897.xhtml), *Preprocessing*, helps you get to know the various types of preprocessing techniques and how you can customize them. We will see the stages of preprocessing such as data preparation, data processing, and data transformation. Apart from this, you will understand the practical aspects of preprocessing.

[Chapter 5](07f71ca1-6c8a-492d-beb3-a47996e93f04.xhtml), *Feature Engineering and NLP Algorithms*, is the core part of an NLP application. We will see how different algorithms and tools are used to generate input for machine learning algorithms, which we will be using to develop NLP applications. We will also understand the statistical concepts used in feature engineering, and we will get into the customization of tools and algorithms.

[Chapter 6](c4861b9e-2bcf-4fce-94d4-f1e2010831de.xhtml), *Advance Feature Engineering and NLP Algorithms*, gives you an understanding of the most recent concepts in NLP, which are used to deal with semantic issues. We will see word2vec, doc2vec, GloVe, and so on, as well as some practical implementations of word2vec by generating vectors from a Game of Thrones dataset.

[Chapter 7](0dc5bd44-3b7d-47ac-8b0d-51134007b483.xhtml), *Rule-Based System for NLP*, details how we can build a rule-based system and all the aspects you need to keep in mind while developing the same for NLP. We will see the rule-making process and code the rules too. We will also see how we can develop a template-based chatbot.

[Chapter 8](97808151-90d2-4034-8d53-b94123154265.xhtml), *Machine Learning for NLP Problems*, provides you fresh aspects of machine learning techniques. We will see the various algorithms used to develop NLP applications. We will also implement some great NLP applications using machine learning.

[Chapter 9](f414d38e-b88e-4239-88bd-2d90e5ce67ab.xhtml), *Deep Learning for NLU and NLG Problems*, introduces you to various aspects of artificial intelligence. We will look at the basic concepts of **artificial neural networks** (**ANNs**) and how you can build your own ANN. We will understand hardcore deep learning, develop the mathematical aspect of deep learning, and see how deep learning is used for **natural language understanding**(**NLU**) and **natural language generation** (**NLG**). You can expect some cool practical examples here as well.

[Appendix A](5665c471-381c-4e91-986f-3db8d4ed94f5.xhtml), *Advance Tools*, gives you a brief introduction to various frameworks such as Apache Hadoop, Apache Spark, and Apache Flink.

[Appendix B](c0abe5d3-090a-41ee-9e3f-521425bdc02b.xhtml), *How to Improve Your NLP Skills*, is about suggestions from my end on how to keep your NLP skills up to date and how constant learning will help you acquire new NLP skills.

[Appendix C](d6c902de-6399-4654-9e38-c387ba111eda.xhtml), *Installation Guide*, has instructions for installations required.

# What you need for this book

Let's discuss some prerequisites for this book. Don't worry, it's not math or statistics, just basic Python coding syntax is all you need to know. Apart from that, you need Python 2.7.X or Python 3.5.X installed on your computer; I recommend using any Linux operating system as well.

The list of Python dependencies can be found at GitHub repository at [https://github.com/jalajthanaki/NLPython/blob/master/pip-requirements.txt.](https://github.com/jalajthanaki/NLPython/blob/master/pip-requirements.txt)

Now let's look at the hardware required for this book. A computer with 4 GB RAM and at least a two-core CPU is good enough to execute the code, but for machine learning and deep learning examples, you may have more RAM, perhaps 8 GB or 16 GB, and computational power that uses GPU(s).

# Who this book is for

This book is intended for Python developers who wish to start with NLP and want to make their applications smarter by implementing NLP in them.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, path names, dummy URLs, user input, and Twitter handles are shown as follows: "The `nltk` library provides some inbuilt corpuses."

A block of code is set as follows:

```py
import nltk
from nltk.corpus import brown as cb
from nltk.corpus import gutenberg as cg

```

Any command-line input or output is written as follows:

```py
pip install nltk or sudo pip install nltk

```

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "This will open an additional dialog window, where you can choose specific libraries, but in our case, click on All packages, and you can choose the path where the packages reside. Wait till all the packages are downloaded."

Warnings or important notes appear like this.

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book-what you liked or disliked. Reader feedback is important for us as it helps us develop titles that you will really get the most out of. To send us general feedback, simply e-mail `feedback@packtpub.com`, and mention the book's title in the subject of your message. If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide at [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

# Downloading the example code

You can download the example code files for this book from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files emailed directly to you. You can download the code files by following these steps:

1.  Log in or register to our website using your e-mail address and password.
2.  Hover the mouse pointer on the SUPPORT tab at the top.
3.  Click on Code Downloads & Errata.
4.  Enter the name of the book in the Search box.
5.  Select the book for which you're looking to download the code files.
6.  Choose from the drop-down menu where you purchased this book from.
7.  Click on Code Download.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR / 7-Zip for Windows
*   Zipeg / iZip / UnRarX for Mac
*   7-Zip / PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/Python-Natural-Language-Processing](https://github.com/PacktPublishing/Python-Natural-Language-Processing). We also have other code bundles from our rich catalog of books and videos available at [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/). Check them out!

# Downloading the color images of this book

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from [https://www.packtpub.com/sites/default/files/downloads/PythonNaturalLanguageProcessing_ColorImages.pdf](https://www.packtpub.com/sites/default/files/downloads/PythonNaturalLanguageProcessing_ColorImages.pdf).

# Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books-maybe a mistake in the text or the code-we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the Errata Submission Form link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title. To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the Errata section.

# Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy. Please contact us at `copyright@packtpub.com` with a link to the suspected pirated material. We appreciate your help in protecting our authors and our ability to bring you valuable content.

# Questions

If you have a problem with any aspect of this book, you can contact us at `questions@packtpub.com`, and we will do our best to address the problem.