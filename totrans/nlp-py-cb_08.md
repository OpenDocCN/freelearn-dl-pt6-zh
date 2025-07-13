# Advanced NLP Recipes

In this chapter, we will go through the following recipes:

*   Creating an NLP pipeline
*   Solving the text similarity problem
*   Identifying topics
*   Summarizing text
*   Resolving anaphora
*   Disambiguating word sense
*   Perform sentiment analysis
*   Exploring advanced sentiment analysis
*   Creating a conversational assistant or chatbot

# Introduction

So far, we have seen how to process input text, identify parts of speech, and extract important information (named entities). We've learned a few computer science concepts also, such as grammars, parsers, and so on. In this chapter, we will dig deeper into advanced topics in **natural language processing** (**NLP**), which need several techniques to properly understand and solve them.

#  Creating an NLP pipeline

In computing, a pipeline can be thought of as a multi-phase data flow system where the output from one component is fed to the input of another component.

These are the things that happen in a pipeline:

*   Data is flowing all the time from one component to another
*   The component is a black box that should worry about the input data and output data

A well-defined pipeline takes care of the following things.

*   The input format of the data that is flowing through each of the components
*   The output format of the data that is coming out of each of the components
*   Making sure that data flow is controlled between components by adjusting the velocity of data inflow and outflow

For example, if you are familiar with Unix/Linux systems and have some exposure to working on a shell, you'd have seen the | operator, which is the shell's abstraction of a data pipe. We can leverage the | operator to build pipelines in the Unix shell.

Let's take an example in Unix (for a quick understanding): how do I find the number of files in a given directory ?

To solve this, we need the following things:

*   We need a component (or a command in the Unix context) that reads the directory and lists all the files in it
*   We need another component (or a command in the Unix context) that reads the lines and prints the count of lines

So, we have the solutions to these two requirements. Which are :

*   The `ls` command
*   The `wc` command

If we can build a pipeline where we take the output from `ls` and feed it to `wc`, we are done.

In terms of Unix commands, `ls -l  | wc -l` is a simple pipeline that counts the files in a directory.

With this knowledge, let's get back to the NLP pipeline requirements:

*   Input data acquisition
*   Breaking the input data into words
*   Identifying the POS of words in the input data
*   Extracting the named entities from the words
*   Identifying the relationships between named entities

In this recipe, let's try to build the simplest possible pipeline; it acquires data from a remote RSS feed and then prints the identified named entities in each document.

# Getting ready

You should have Python installed, along with the `nltk`, `queue`, `feedparser`, and `uuid` libraries.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `PipelineQ.py`.

3.  Type the following source code:

![](img/c2691bf7-2c41-45ec-b6fb-a508383a6e71.png)

3.  Save the file.
4.  Run the program using the Python interpreter.
5.  You will see this output:

![](img/2432e7f0-57ca-4b03-99d9-b1ac745705af.png)

# How it works...

Let's see how to build this pipeline:

```py
import nltk
import threading
import queue
import feedparser
import uuid
```

These five instructions import five Python libraries into the current program:

*   `nltk`: Natural language toolkit
*   `threading`: A threading library used to create lightweight tasks within a single program
*   `queue`: A queue library that can be used in a multi-threaded program
*   `feedparser`: An RSS feed parsing library
*   `uuid`: An RFC-4122-based uuid version 1, 3, 4, 5-generating library

```py
threads = []
```

Creating a new empty list to keep track of all the threads in the program:

```py
queues = [queue.Queue(), queue.Queue()]
```

This instruction creates a list of two queues in a variable `queue`?

Why do we need two queues:

*   The first queue is used to store tokenized sentences
*   The second queue is used to store all the POS analyzed words

This instruction defines a new function, `extractWords()`, which reads a sample RSS feed from the internet and stores the words, along with a unique identifier for this text:

```py
def extractWords():
```

We are defining a sample URL (entertainment news) from the India Times website:

```py
url = 'https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms'
```

This instruction invokes the `parse()` function of the `feedparser` library. This `parse()` function downloads the content of the URL and converts it into a list of news items. Each news item is a dictionary with title and summary keys:

```py
feed = feedparser.parse(url)
```

We are taking the first five entries from the RSS feed and storing the current item in a variable called `entry`:

```py
for entry in feed['entries'][:5]:
```

The title of the current RSS feed item is stored in a variable called `text`:

```py
text = entry['title']
```

This instruction skips the titles that contain sensitive words. Since we are reading the data from the internet, we have to make sure that the data is properly sanitized:

```py
if 'ex' in text:
  continue
```

Break the input text into words using the `word_tokenize()` function and store the result into a variable called `words`:

```py
words = nltk.word_tokenize(text)
```

Create a dictionary called `data` with two key-value pairs, where we are storing the UUID and input words under the UUID and input keys respectively:

```py
data = {'uuid': uuid.uuid4(), 'input': words}
```

This instruction stores the dictionary in the first queue, `queues[0]`. The second argument is set to true, which indicates that if the queue is full, pause the thread:

```py
queues[0].put(data, True)
```

A well-designed pipeline understands that it should control the inflow and outflow of the data according to the component's computation capacity. If not, the entire pipeline collapses. This instruction prints the current RSS item that we are processing along with its unique ID:

```py
print(">> {} : {}".format(data['uuid'], text))
```

This instruction defines a new function called `extractPOS()`, which reads from the first queue, processes the data, and saves the POS of the words in the second queue:

```py
def extractPOS():
```

This is an infinite loop:

```py
while True:
```

These instructions check whether the first queue is empty. When the queue is empty, we stop processing:

```py
if queues[0].empty():
  break
```

In order to make this program robust, pass the feedback from the first queue. This is left as an exercise to the reader. This is the else part, which indicates there is some data in the first queue:

```py
else:
```

Take the first item from the queue (in FIFO order):

```py
data = queues[0].get()
```

Identify the parts of speech in the words:

```py
words = data['input']
postags = nltk.pos_tag(words)
```

Update the first queue, mentioning that we are done with processing the item that is just extracted by this thread:

```py
queues[0].task_done()
```

Store the POS tagged word list in the second queue so that the next phase in the pipeline will execute things. Here also, we are using true for the second parameter, which will make sure that the thread will wait if there is no free space in the queue:

```py
queues[1].put({'uuid': data['uuid'], 'input': postags}, True)
```

This instruction defines a new function, `extractNE()`, which reads from the second queue, processes the POS tagged words, and prints the named entities on screen:

```py
def extractNE():
```

This is an infinite loop instruction:

```py
while True:
```

If the second queue is empty, then we exit the infinite loop:

```py
if queues[1].empty():
  break
```

This instruction picks an element from the second queue and stores it in a data variable:

```py
else:
  data = queues[1].get()
```

This instruction marks the completion of data processing on the element that was just picked from the second queue:

```py
postags = data['input']
queues[1].task_done()
```

This instruction extracts the named entities from the `postags` variable and stores it in a variable called `chunks`:

```py
chunks = nltk.ne_chunk(postags, binary=False)

print("  << {} : ".format(data['uuid']), end = '')
  for path in chunks:
    try:
      label = path.label()
      print(path, end=', ')
      except:
        pass
      print()
```

These instructions do the following

*   Print the UUID from the data dictionary
*   Iterate over all chunks that are identified
*   We are using a try/except block because not all elements in the tree have the `label()` function (they are tuples when no NE is found)
*   Finally, we call a `print()` function, which prints a newline on screen

This instruction defines a new function, `runProgram`, which does the pipeline setup using threads:

```py
def runProgram():
```

These three instructions create a new thread with `extractWords()` as the function, start the thread and add the thread object (`e`) to the list called `threads`:

```py
e = threading.Thread(target=extractWords())
e.start()
threads.append(e)
```

These instructions create a new thread with `extractPOS()` as the function, start the thread, and add the thread object (`p`) to the list variable `threads`:

```py
p = threading.Thread(target=extractPOS())
p.start()
threads.append(p)
```

These instructions create a new thread using `extractNE()` for the code, start the thread, and add the thread object (`n`) to the list `threads`:

```py
    n = threading.Thread(target=extractNE())
    n.start()
    threads.append(n)
```

These two instructions release the resources that are allocated to the queues once all the processing is done:

```py
    queues[0].join()
    queues[1].join()
```

These two instructions iterate over the threads list, store the current thread object in a variable, `t`, call the `join()` function to mark the completion of the thread, and release resources allocated to the thread:

```py
    for t in threads:
        t.join()
```

This is the section of the code that is invoked when the program is run with the main thread. The `runProgram()` is called, which simulates the entire pipeline:

```py
if __name__ == '__main__':
    runProgram()
```

#  Solving the text similarity problem

The text similarity problem deals with the challenge of finding how close given text documents are. Now, when we say close, there are many dimensions in which we can say they are closer or far:

*   Sentiment/emotion dimension
*   Sense dimension
*   Mere presence of certain words

There are many algorithms available for this; all of them vary in the degree of complexity, the resources needed, and the volume of data we are dealing with.

In this recipe, we will use the TF-IDF algorithm to solve the similarity problem. So first, let's understand the basics:

*   **Term frequency (TF)**: This technique tries to find the relative importance (or frequency) of the word in a given document

Since we are talking about relative importance, we generally normalize the frequency with respect to the total words that are present in the document to compute the TF value of a word.

*   **Inverse document frequency (IDF)** : This technique makes sure that words that are frequently used (a, the, and so on) should be given lower weight when compared to the words that are rarely used.

Since both TF and IDF values are decomposed to numbers (fractions), we will do a multiplication of these two values for each term against every document and build *M* vectors of *N* dimensions (where *N* is the total number of documents and *M* are the unique words in all the documents).

Once we have these vectors, we need to find the cosine similarity using the following formula on these vectors:

![](img/4acc3f1a-3759-4c0a-acbe-2e297e6f41be.png)

# Getting ready

You should have Python installed, along with the `nltk` and `scikit` libraries. Having some understanding of mathematics is helpful.

# How to do it...

1.  Open atom editor (or your favorite programming editor).
2.  Create a new file called `Similarity.py`.

3.  Type the following source code:

![](img/21458fab-7f37-42a8-8e06-5e6a227a7eed.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/f9774ed3-f05c-43c4-9fb6-49a8bba75d10.png)

# How it works...

Let's see how we are solving the text similarity problem. These four instructions import the necessary libraries that are used in the program:

```py
import nltk
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

We are defining a new class, `TextSimilarityExample`:

```py
class TextSimilarityExample:
```

This instruction defines a new constructor for the class:

```py
def __init__(self):
```

This instruction defines sample sentences on which we want to find the similarity.

```py
self.statements = [
  'ruled india',
  'Chalukyas ruled Badami',
  'So many kingdoms ruled India',
  'Lalbagh is a botanical garden in India'
]
```

We are defining the TF of all the words in a given sentence:

```py
    def TF(self, sentence):
        words = nltk.word_tokenize(sentence.lower())
        freq = nltk.FreqDist(words)
        dictionary = {}
        for key in freq.keys():
            norm = freq[key]/float(len(words))
            dictionary[key] = norm
        return dictionary
```

 This function does the following things:

*   Converts the sentence to lower case and extracts all the words
*   Finds the frequency distribution of these words using the nltk `FreqDist` function
*   Iterates over all the dictionary keys, builds the normalized floating values, and stores them in a dictionary
*   Returns the dictionary that contains the normalized score for each word in the sentence

We are defining an IDF that finds the IDF value for all the words in all the documents:

```py
    def IDF(self):
        def idf(TotalNumberOfDocuments, NumberOfDocumentsWithThisWord):
            return 1.0 + math.log(TotalNumberOfDocuments/NumberOfDocumentsWithThisWord)
        numDocuments = len(self.statements)
        uniqueWords = {}
        idfValues = {}
        for sentence in self.statements:
            for word in nltk.word_tokenize(sentence.lower()):
                if word not in uniqueWords:
                    uniqueWords[word] = 1
                else:
                    uniqueWords[word] += 1
        for word in uniqueWords:
            idfValues[word] = idf(numDocuments, uniqueWords[word])
        return idfValues
```

This function does the following things:

*   We define a local function called `idf()`, which is the formula to find the IDF of a given word
*   We iterate over all the statements and convert them to lowercase
*   Find how many times each word is present across all the documents
*   Build the IDF value for all words and return the dictionary containing these IDF values

We are now defining a `TF_IDF` (TF multiplied by IDF) for all the documents against a given search string.

```py
    def TF_IDF(self, query):
        words = nltk.word_tokenize(query.lower())
        idf = self.IDF()
        vectors = {}
        for sentence in self.statements:
            tf = self.TF(sentence)
            for word in words:
                tfv = tf[word] if word in tf else 0.0
                idfv = idf[word] if word in idf else 0.0
                mul = tfv * idfv
                if word not in vectors:
                    vectors[word] = []
                vectors[word].append(mul)
        return vectors
```

 Let's see what we are doing here:

*   Break the search string into tokens
*   Build `IDF()` for all sentences in the `self.statements` variable
*   Iterate over all sentences and find the TF for all words in this sentence
*   Filter and use only the words that are present in the input search string and build vectors that consist of *tf*idf* values against each document
*   Return the list of vectors for each word in the search query

This function displays the contents of vectors on screen:

```py
    def displayVectors(self, vectors):
        print(self.statements)
        for word in vectors:
            print("{} -> {}".format(word, vectors[word]))
```

Now, in order to find the similarity, as we discussed initially, we need to find the Cosine similarity on all the input vectors. We can do all the math ourselves. But this time, let's try to use scikit to do all the computations for us.

```py
    def cosineSimilarity(self):
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(self.statements)
        for j in range(1, 5):
            i = j - 1
            print("\tsimilarity of document {} with others".format(i))
            similarity = cosine_similarity(matrix[i:j], matrix)
            print(similarity)
```

In the previous functions, we learned how to build TF and IDF values and finally get the TF x IDF values for all the documents.

Let's see what we are doing here:

*   Defining a new function: `cosineSimilarity()`
*   Creating a new vectorizer object
*   Building a matrix of TF-IDF values for all the documents that we are interested in, using the `fit_transform()` function
*   Later we compare each document with all other documents and see how close they are to each other

This is the `demo()` function and it runs all the other functions we have defined before:

```py
    def demo(self):
        inputQuery = self.statements[0]
        vectors = self.TF_IDF(inputQuery)
        self.displayVectors(vectors)
        self.cosineSimilarity()
```

 Let's see what we are doing here

*   We take the first statement as our input query.
*   We build vectors using our own handwritten `TF_IDF()` function.
*   We display our TF x IDF vectors for all sentences on screen.
*   We print the cosine similarity computed for all the sentences using the `scikit` library by invoking the `cosineSimilarity()` function.

We are creating a new object for the `TextSimilarityExample()` class and then invoking the `demo()` function.

```py
similarity = TextSimilarityExample()
similarity.demo()
```

# Identifying topics

In the previous chapter, we learned how to do document classification. Beginners might think document classification and topic identification are the same, but there is a slight difference.

Topic identification is the process of discovering topics that are present in the input document set. These topics can be multiple words that occur uniquely in a given text.

Let's take an example. When we read arbitrary text that contains a mention of Sachin Tendulkar, score, win we can understand that the sentence is describing cricket. But we may be wrong as well.

In order to find all these types of topics in a given input text, we use the Latent Dirichlet allocation algorithm (we could use TF-IDF as well, but since we have already explored it in a previous recipe, let's see how LDA works in identifying the topic).

# Getting ready

You should have Python installed, along with the `nltk`, `gensim`, and `feedparser` libraries.

# How to do it...

1.  Open atom editor (or your favorite programming editor).
2.  Create a new file called `IdentifyingTopic.py`.
3.  Type the following source code:

![](img/0b4aa55c-b327-4ad5-aa01-d0e95e7a21bd.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/c3ba63c9-117a-4bb1-a6a1-0d78dfb1869f.png)

# How it works...

Let's see how the topic identification program works. These five instructions import the necessary libraries into the current program.

```py
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim import corpora, models
import nltk
import feedparser
```

This instruction defines a new class, `IdentifyingTopicExample`:

```py
class IdentifyingTopicExample:
```

This instruction defines a new function, `getDocuments()`, whose responsibility is to download few documents from the internet using `feedparser`:

```py
    def getDocuments(self):
```

Download all the documents mentioned in the URL and store the list of dictionaries into a variable called `feed`:

```py
        url = 'https://sports.yahoo.com/mlb/rss.xml'
        feed = feedparser.parse(url)
```

Empty the list to keep track of all the documents that we are going to analyze further:

```py
        self.documents = []
```

Take the top five documents from the `feed` variable and store the current news item into a variable called `entry`:

```py
        for entry in feed['entries'][:5]:
```

Store the news summary into a variable called `text`:

```py
            text = entry['summary']
```

If the news article contains any sensitive words, skip those:

```py
            if 'ex' in text:
                continue
```

Store the document in the `documents` variable:

```py
            self.documents.append(text)
```

Display the current document on the screen:

```py
            print("-- {}".format(text))
```

Display an informational message to the user that we have collected *N* documents from the given `url`:

```py
        print("INFO: Fetching documents from {} completed".format(url))
```

This instruction defines a new function, `cleanDocuments()`, whose responsibility is to clean the input text (since we are downloading it from the internet, it can contain any type of data).

```py
    def cleanDocuments(self):
```

We are interested in extracting words that are in the English alphabet. So, this tokenizer is defined to break the text into tokens, where each token consists of letters from a to z and A-Z. By doing so, we can be sure that punctuation and other bad data doesn't come into the processing.

```py
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
```

Store the stop words of English in a variable, `en_stop`:

```py
        en_stop = set(stopwords.words('english'))
```

Define a empty list called `cleaned`, which is used to store all the cleaned and tokenized documents:

```py
        self.cleaned = []
```

Iterate over all the documents we have collected using the `getDocuments()` function:

```py
        for doc in self.documents:
```

Convert the document to lowercase to avoid treating the same word differently because they are case sensitive:

```py
            lowercase_doc = doc.lower()
```

Break the sentence into words. The output is a list of words stored in a variable called `words`:

```py
            words = tokenizer.tokenize(lowercase_doc)
```

Ignore all the words from the sentence if they belong to the English stop word category and store all of them in the `non_stopped_words` variable:

```py
            non_stopped_words = [i for i in words if not i in en_stop]
```

Store the sentence that is tokenized and cleaned in a variable called `self.cleaned` (class member).

```py
            self.cleaned.append(non_stopped_words)
```

Show a diagnostic message to the user that we have finished cleaning the documents:

```py
        print("INFO: Cleaning {} documents completed".format(len(self.documents)))
```

This instruction defines a new function, `doLDA`, which runs the LDA analysis on the cleaned documents:

```py
    def doLDA(self):
```

Before we directly process the cleaned documents, we create a dictionary from these documents:

```py
        dictionary = corpora.Dictionary(self.cleaned)
```

The input corpus is defined as a bag of words for each cleaned sentence:

```py

     corpus = [dictionary.doc2bow(cleandoc) for cleandoc in self.cleaned]
```

Create a model on the corpus with the number of topics defined as `2` and set the vocabulary size/mapping using the `id2word` parameter:

```py
        ldamodel = models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary)
```

Print two topics, where each topic should contain four words on the screen:

```py
        print(ldamodel.print_topics(num_topics=2, num_words=4))
```

This is the function that does all the steps in order:

```py
    def run(self):
        self.getDocuments()
        self.cleanDocuments()
        self.doLDA()
```

When the current program is invoked as the main program, create a new object called `topicExample` from the `IdentifyingTopicExample()` class and invoke the `run()` function on the object.

```py
if __name__ == '__main__':
    topicExample = IdentifyingTopicExample()
    topicExample.run()
```

# Summarizing text

In this information overload era, there is so much information that is available in print/text form. Its humanly impossible for us to consume all this data. In order to make the consumption of this data easier, we have been trying to invent algorithms that can help simplify large text into a summary (or a gist) that we can easily digest.

By doing this, we will save time and also make things easier for the network.

In this recipe, we will use the gensim library, which has built-in support for this summarization using the TextRank algorithm ([https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)).

# Getting ready

You should have Python installed, along with the `bs4` and `gensim` libraries.

# How to do it...

1.  Open atom editor (or your favorite programming editor).
2.  Create a new file called `Summarize.py`.

3.  Type the following source code:

![](img/a95816f9-4168-4eb3-85ae-7e56797626ce.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/b7fa82ee-3668-4db0-98c8-b5d5b9571fc8.png)

# How it works...

Let's see how we our summarization program works. 

```py
from gensim.summarization import summarize
from bs4 import BeautifulSoup
import requests
```

These three instructions import the necessary libraries into the current program:

*   `gensim.summarization.summarize`: Text-rank-based summarization algorithm
*   `bs4`: A `BeautifulSoup` library for parsing HTML documents
*   `requests`: A library to download HTTP resources

We are defining a dictionary called URLs whose keys are the title of the paper that is auto generated and the value is the URL to the paper:

```py
urls = {
    'Daff: Unproven Unification of Suffix Trees and Redundancy': 'http://scigen.csail.mit.edu/scicache/610/scimakelatex.21945.none.html',
    'CausticIslet: Exploration of Rasterization': 'http://scigen.csail.mit.edu/scicache/790/scimakelatex.1499.none.html'
}
```

Iterate through all the keys of the dictionary:

```py
for key in urls.keys():
```

Store the URL of the current paper in a variable called `url`:

```py
    url = urls[key]
```

Download the content of the url using the `requests` library's `get()` method and store the response object into a variable, `r`:

```py
    r = requests.get(url)
```

Use `BeautifulSoup()` to parse the text from the `r` object using the HTML parser and store the return object in a variable called `soup`:

```py
    soup = BeautifulSoup(r.text, 'html.parser')
```

Strip out all the HTML tags and extract only the text from the document into the variable `data`:

```py
    data = soup.get_text()
```

Find the position of the text `Introduction` and skip past towards end of string, to mark is as starting offset from which we want to extract the substring.

```py
    pos1 = data.find("1  Introduction") + len("1  Introduction")
```

Find the second position in the document, exactly at the beginning of the related work section:

```py
    pos2 = data.find("2  Related Work")
```

Now, extract the introduction of the paper, which is between these two offsets:

```py
    text = data[pos1:pos2].strip()
```

Display the URL and the title of the paper on the screen:

```py
    print("PAPER URL: {}".format(url))
    print("TITLE: {}".format(key))
```

Call the `summarize()` function on the text, which returns shortened text as per the text rank algorithm:

```py
    print("GENERATED SUMMARY: {}".format(summarize(text)))
```

Print an extra newline for more readability of the screen output.

```py
    print()
```

#  Resolving anaphora

In many natural languages, while forming sentences, we avoid the repeated use of certain nouns with pronouns to simplify the sentence construction.

For example:

Ravi is a boy. He often donates money to the poor.

In this example, there are two statements:

*   Ravi is a boy.
*   He often donates money to the poor.

When we start analyzing the second statement, we cannot make a decision about who is donating the money  without knowing about the first statement. So, we should associate He with Ravi to get the complete sentence meaning. All this reference resolution happens naturally in our mind.

If we observe the previous example carefully, first the subject is present; then the pronoun comes up. So the direction of the flow is from left to right. Based on this flow, we can call these types of sentences anaphora.

Let's take another example:

He was already on his way to airport. Realized Ravi

This is another class of example where the direction of expression is the reverse order (first the pronoun and then the noun). Here too, He is associated with Ravi. These types of sentences are called as Cataphora.

The earliest available algorithm for this anaphora resolution dates back to the 1970; Hobbs has presented a paper on this. An online version of this paper is available here: [https://www.isi.edu/~hobbs/pronoun-papers.html](https://www.isi.edu/~hobbs/pronoun-papers.html).

In this recipe, we will try to write a very simple Anaphora resolution algorithm using what we have learned just now.

# Getting ready

You should have python installed, along with the `nltk` library and `gender` datasets.

You can use `nltk.download()` to download the corpus.

# How to do it...

1.  Open atom editor (or your favorite programming editor).
2.  Create a new file called `Anaphora.py`.

3.  Type the following source code:

![](img/2f596d1e-74a7-4788-bcf8-0db0cf82e547.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/7d6cf7f5-a322-479c-b628-5d50d524c98d.png)

# How it works...

Let see how our simple Anaphora resolution algorithm works.

```py
import nltk
from nltk.chunk import tree2conlltags
from nltk.corpus import names
import random
```

These four instructions import the necessary modules and functions that are used in the program. We are defining a new class called `AnaphoraExample`:

```py
class AnaphoraExample:
```

We are defining a new constructor for this class, which doesn't take any parameters:

```py
    def __init__(self):
```

These two instructions load all the male and female names from the `nltk.names` corpus and tag them as male/female before storing them in two lists called male/female.

```py
        males = [(name, 'male') for name in names.words('male.txt')]
        females = [(name, 'female') for name in names.words('female.txt')]
```

This instruction creates a unique list of males and females. `random.shuffle()` ensures that all of the data in the list is randomized:

```py
        combined = males + females
        random.shuffle(combined)
```

This instruction invokes the `feature()` function on the gender and stores all the names in a variable called `training`:

```py
        training = [(self.feature(name), gender) for (name, gender) in combined]
```

We are creating a `NaiveBayesClassifier` object called `_classifier` using the males and females features that are stored in a variable called `training`:

```py
        self._classifier = nltk.NaiveBayesClassifier.train(training)
```

This function defines the simplest possible feature, which categorizes the given name as male or female just by looking at the last letter of the name:

```py
    def feature(self, word):
        return {'last(1)' : word[-1]}
```

This function takes a word as an argument and tries to detect the gender as male or female using the classifier we have built:

```py
    def gender(self, word):
        return self._classifier.classify(self.feature(word))
```

This is the main function that is of interest to us, as we are going to detect anaphora on the sample sentences:

```py
    def learnAnaphora(self):
```

These are four examples with mixed complexity expressed in anaphora form:

```py
        sentences = [
            "John is a man. He walks",
            "John and Mary are married. They have two kids",
            "In order for Ravi to be successful, he should follow John",
            "John met Mary in Barista. She asked him to order a Pizza"
        ]
```

This instruction iterates over all the sentences by taking one sentence at a time to a local variable called `sent`:

```py

        for sent in sentences:
```

This instruction tokenizes, assigns parts of speech, extracts chunks (named entities), and returns the chunk tree to a variable called `chunks`:

```py
            chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)), binary=False)
```

This variable is used to store all the names and pronouns that help us resolve anaphora:

```py
            stack = []
```

This instruction shows the current sentence that is being processed on the user's screen:

```py
            print(sent)
```

This instruction flattens the tree chunks to a list of items expressed in IOB format:

```py
            items = tree2conlltags(chunks)
```

We are traversing through all chunked sentences that are in IOB format (tuple with three elements):

```py
            for item in items:
```

If the POS of the word is `NNP` and IOB letter for this word is `B-PERSON` or `O`, then we mark this word as a `Name`:

```py
                if item[1] == 'NNP' and (item[2] == 'B-PERSON' or item[2] == 'O'):
                    stack.append((item[0], self.gender(item[0])))
```

If the POS of the word is `CC`, then also we will add this to the `stack` variable:

```py
                elif item[1] == 'CC':
                    stack.append(item[0])
```

If the POS of the word is `PRP`, then we will add this to the `stack` variable:

```py
                elif item[1] == 'PRP':
                    stack.append(item[0])
```

Finally we print the stack on the screen:

```py
            print("\t {}".format(stack))
```

We are creating a new object called `anaphora` from `AnaphoraExample()` and invoking the `learnAnaphora()` function on the anaphora object. Once this function execution completes, we see the list of words for every sentence.

```py
anaphora = AnaphoraExample()
anaphora.learnAnaphora()
```

# Disambiguating word sense

In previous chapters, we learned how to identify POS of the words, find named entities, and so on. Just like a word in English behaves as both a noun and a verb, finding the sense in which a word is used is very difficult for computer programs.

Let's take a few examples to understand this sense portion:

| **Sentence** | **Description** |
| *She is my date* | Here the sense of the word *date* is not the calendar date but expresses a human relationship. |
| *You have taken too many leaves to skip cleaning leaves in the garden* | Here the word *leaves* has multiple senses: 
*   The first word *leave* means taking a break

*   The second one actually refers to tree leaves

 |

Like this, there are many combinations of senses possible in sentences.

One of the challenges we have faced for senses identification is to find a proper nomenclature to describe these senses. There are many English dictionaries available that describe the behavior of words and all possible combinations of those. Of them all, WordNet is the most structured, preferred, and widely accepted source of sense usage.

In this recipe, we will see examples of senses from the WordNet library and use the built-in `nltk` library to find out the sense of words.

Lesk is the oldest algorithm that was coined to tackle this sense detection. You will see, however, that this one too is not accurate in some cases.

# Getting ready

You should have Python installed, along with the `nltk` library.

# How to do it...

1.  Open atom editor (or your favorite programming editor).
2.  Create a new file called `WordSense.py`.
3.  Type the following source code:

![](img/25897007-6789-40c0-b6b6-1ea605c35b4e.png)

4.  Save the file.
5.  Run the program using the Python interpreter.

6.  You will see the following output:

![](img/984afb10-d504-4899-b396-6d0a1d9d7cdb.png)

# How it works...

Let's see how our program works. This instruction imports the `nltk` library into the program:

```py
import nltk
```

We are defining a function with the name `understandWordSenseExamples()`, which uses the WordNet corpus to showcase the possible senses of the words that we are interested in.

```py

def understandWordSenseExamples():
```

These are the three words with different senses of expression. They are stored as a list in a variable called `words`.

```py
    words = ['wind', 'date', 'left']

   print("-- examples --")
    for word in words:
        syns = nltk.corpus.wordnet.synsets(word)
        for syn in syns[:2]:
            for example in syn.examples()[:2]:
                print("{} -> {} -> {}".format(word, syn.name(), example))
```

These instructions do the following:

*   Iterate over all the words in the list by storing the current word in a variable called `word`.
*   Invoke the `synsets()` function from the `wordne`t module and store the result in the `syns` variable.
*   Take the first three synsets from the list, iterate through them, and take the current one in a variable called `syn`.
*   Invoke the `examples()` function on the `syn` object and take the first two examples as the iterator. The current value of the iterator is available in the variable example.
*   Print the word, synset's name, and example sentence finally.

Define a new function, `understandBuiltinWSD()`, to explore the NLTK built-in lesk algorithm's performance on sample sentences.

```py
def understandBuiltinWSD():
```

Define a new variable called `maps`, a list of tuples.

```py
    print("-- built-in wsd --")
    maps = [
        ('Is it the fish net that you are using to catch fish ?', 'fish', 'n'),
        ('Please dont point your finger at others.', 'point', 'n'),
        ('I went to the river bank to see the sun rise', 'bank', 'n'),
    ]
```

 Each tuple consists of three elements:

*   The sentence we want to analyze
*   The word in the sentence for which we want to find the sense
*   The POS of the word

In these two instructions, we are traversing through the `maps` variable, taking the current tuple into variable `m`, invoking the `nltk.wsd.lesk()` function, and displaying the formatted results on screen.

```py
    for m in maps:
        print("Sense '{}' for '{}' -> '{}'".format(m[0], m[1], nltk.wsd.lesk(m[0], m[1], m[2])))
```

When the program is run, call the two functions that show the results on the user's screen.

```py
if __name__ == '__main__':
    understandWordSenseExamples()
    understandBuiltinWSD()
```

#  Performing sentiment analysis

Feedback is one of the most powerful measures for understanding relationships. Humans are very good at understanding feedback in verbal communication as the analysis happens unconsciously. In order to write computer programs that can measure and find the emotional quotient, we should have some good understanding of the ways these emotions are expressed in these natural languages.

Let's take a few examples:

| **Sentence** | **Description** |
| *I am very happy* | Indicates a happy emotion |
| *She is so :(* | We know there is an iconic sadness expression here |

With the increased use of text, icons, and emojis in written natural language communication, it's becoming increasingly difficult for computer programs to understand the emotional meaning of a sentence.

Let's try to write a program to understand the facilities nltk provides to build our own algorithm.

# Getting ready

You should have Python installed, along with the `nltk` library.

# How to do it...

1.  Open atom editor (or your favorite programming editor).
2.  Create a new file called `Sentiment.py`.

3.  Type the following source code:

![](img/842ffb61-bece-46e2-9d4f-1db5e6c5ca33.png)

4.  Save the file.
5.  Run the program using the Python interpreter.

6.  You will see the following output:

![](img/8487c0f4-10ac-4de5-b4f2-f61f307f889f.png)

# How it works...

Let's see how our sentiment analysis program works. These instructions import the `nltk` module and `sentiment_analyzer` module respectively.

```py
import nltk
import nltk.sentiment.sentiment_analyzer
```

We are defining a new function, `wordBasedSentiment()`, which we will use to learn how to do sentiment analysis based on the words that we already know and which mean something important to us.

```py
def wordBasedSentiment():
```

We are defining a list of three words that are special to us as they represent some form of happiness. These words are stored in the `positive_words` variable.

```py
    positive_words = ['love', 'hope', 'joy']
```

This is the sample text that we are going to analyze; the text is stored in a variable called `text`.

```py
    text = 'Rainfall this year brings lot of hope and joy to Farmers.'.split()
```

We are calling the `extract_unigram_feats()` function on the text using the words that we have defined. The result is a dictionary of input words that indicate whether the given words are present in the text or not.

```py
    analysis = nltk.sentiment.util.extract_unigram_feats(text, positive_words)
```

This instruction displays the dictionary on the user's screen.

```py
    print(' -- single word sentiment --')
    print(analysis)
```

This instruction defines a new function that we will use to understand whether some pairs of words occur in a sentence.

```py
def multiWordBasedSentiment():
```

This instruction defines a list of two-word tuples. We are interested in finding if these pairs of words occur together in a sentence.

```py
    word_sets = [('heavy', 'rains'), ('flood', 'bengaluru')]
```

This is the sentence we are interested in processing and finding the features of.

```py
    text = 'heavy rains cause flash flooding in bengaluru'.split()
```

We are calling the `extract_bigram_feats()` on the input sentence against the sets of words in the `word_sets` variable. The result is a dictionary that tells whether these pairs of words are present in the sentence or not.

```py

    analysis = nltk.sentiment.util.extract_bigram_feats(text, word_sets)
```

This instruction displays the dictionary on screen.

```py
    print(' -- multi word sentiment --')
    print(analysis)
```

We are defining a new function, `markNegativity()`, which helps us understand how we can find negativity in a sentence.

```py
def markNegativity():
```

Next is the sentence on which we want to run the negativity analysis. It's stored in a variable, `text`.

```py
    text = 'Rainfall last year did not bring joy to Farmers'.split()
```

We are calling the `mark_negation()` function on the text. This returns a list of all the words in the sentence along with a special suffix `_NEG` for all the words that come under the negative sense. The result is stored in the `negation` variable.

```py
    negation = nltk.sentiment.util.mark_negation(text)
```

This instruction displays the list negation on screen.

```py

    print(' -- negativity --')
    print(negation)
```

When the program is run, these functions are called and we see the output of three functions in the order they are executed (top-down).

```py
if __name__ == '__main__':
    wordBasedSentiment()
    multiWordBasedSentiment()
    markNegativity()
```

#  Exploring advanced sentiment analysis

We are seeing that more and more businesses are going online to increase their target customer base and the customers are given the ability to leave feedback via various channels. It's becoming more and more important for businesses to understand the emotional response of their customers with respect to the businesses they run.

In this recipe, we will write our own sentiment analysis program based on what we have learned in the previous recipe. We will also explore the built-in vader sentiment analysis algorithm, which helps evaluate in finding the sentiment of complex sentences.

# Getting ready

You should have Python installed, along with the `nltk` library.

# How to do it...

1.  Open atom editor (or your favorite programming editor).
2.  Create a new file called `AdvSentiment.py`.

3.  Type the following source code:

![](img/b2a2b51b-c606-442c-8639-148cea230656.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/49f760f3-9ef7-4291-a3a7-f727caab0996.png)

# How it works...

Now, let's see how our sentiment analysis program works. These four instructions import the necessary modules that we are going to use as part of this program.

```py
import nltk
import nltk.sentiment.util
import nltk.sentiment.sentiment_analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
```

Defining a new function, `mySentimentAnalyzer()`:

```py
def mySentimentAnalyzer():
```

This instruction defines a new subfunction, `score_feedback()`, which takes a sentence as input and returns the score for the sentence in terms of `-1` negative, `0` neutral, and `1` positive.

```py

    def score_feedback(text):
```

Since we are just experimenting, we are defining the three words using which we are going to find the sentiment. In real-world use cases, we might use these from the corpus of a larger dictionary.

```py

        positive_words = ['love', 'genuine', 'liked']
```

This instruction breaks the input sentence into words. The list of words is fed to the `mark_negation()` function to identify the presence of any negativity in the sentence. Join the result from `mark_negation()` to the string and see if the `_NEG` suffix is present; then set the score as `-1`.

```py

        if '_NEG' in ' '.join(nltk.sentiment.util.mark_negation(text.split())):
            score = -1
```

Here we are using `extract_unigram_feats()` on the input text against `positive_words` and storing the dictionary into a variable called `analysis`:

```py

        else:
            analysis = nltk.sentiment.util.extract_unigram_feats(text.split(), positive_words)
```

The value of score is decided to be `1` if there is a presence of the positive word in the input text.

```py
            if True in analysis.values():
                score = 1
            else:
                score = 0
```

Finally this `score_feedback()` function returns the computed score:

```py
        return score
```

These are the four reviews that we are interested in processing using our algorithm to print the score.

```py

    feedback = """I love the items in this shop, very genuine and quality is well maintained.
    I have visited this shop and had samosa, my friends liked it very much.
    ok average food in this shop.
    Fridays are very busy in this shop, do not place orders during this day."""
```

These instructions extract the sentences from the variable feedback by splitting on newline (`\n`) and calling the `score_feedback()` function on this text.

```py
    print(' -- custom scorer --')
    for text in feedback.split("\n"):
        print("score = {} for >> {}".format(score_feedback(text), text))
```

The result will be the score and sentence on the screen. This instruction defines the `advancedSentimentAnalyzer()` function, which will be used to understand the built-in features of NLTK sentiment analysis.

```py
def advancedSentimentAnalyzer():
```

We are defining five sentences to analyze. you'll note that we are also using emoticons (icons) to see how the algorithm works.

```py
    sentences = [
        ':)',
        ':(',
        'She is so :(',
        'I love the way cricket is played by the champions',
        'She neither likes coffee nor tea',
    ]
```

This instruction creates a new object for `SentimentIntensityAnalyzer()` and stores the object in the variable `senti`.

```py
    senti = SentimentIntensityAnalyzer()

  print(' -- built-in intensity analyser --')
    for sentence in sentences:
        print('[{}]'.format(sentence), end=' --> ')
        kvp = senti.polarity_scores(sentence)
        for k in kvp:
            print('{} = {}, '.format(k, kvp[k]), end='')
        print()
```

These instructions do the following things:

*   Iterate over all the sentences and store the current one in the variable `sentence`
*   Display the currently processed sentence on screen
*   Invoke the `polarity_scores()` function on this sentence; store the result in a variable called `kvp`
*   Traverse through the dictionary `kvp` and print the key (negativity, neutral, positivity, or compound types) and the score computed for these types

When the current program is invoked, call these two functions to display the results on screen.

```py

if __name__ == '__main__':
    advancedSentimentAnalyzer()
    mySentimentAnalyzer()
```

# Creating a conversational assistant or chatbot

Conversational assistants or chatbots are not very new. One of the foremost of this kind is ELIZA, which was created in the early 1960s and is worth exploring.

In order to successfully build a conversational engine, it should take care of the following things:

*   Understand the target audience
*   Understand the natural language in which communication happens
*   Understand the intent of the user
*   Come up with responses that can answer the user and give further clues

NLTK has a module, `nltk.chat`, which simplifies building these engines by providing a generic framework.

Let's see the available engines in NLTK:

| **Engines** | **Modules** |
| Eliza | `nltk.chat.eliza` Python module |
| Iesha | `nltk.chat.iesha` Python module |
| Rude | `nltk.chat.rudep` Python module |
| Suntsu | `nltk.chat.suntsu` module |
| Zen | `nltk.chat.zen` module |

In order to interact with these engines we can just load these modules in our Python program and invoke the `demo()` function.

This recipe will show us how to use built-in engines and also write our own simple conversational engine using the framework provided by the `nltk.chat` module.

# Getting ready

You should have Python installed, along with the `nltk` library. Having an understanding of regular expressions also helps.

# How to do it...

1.  Open atom editor (or your favorite programming editor).
2.  Create a new file called `Conversational.py`.

3.  Type the following source code:

![](img/1d727273-2841-4170-af3a-992ef49b141c.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/f142875f-b0c3-4faf-b7e7-58ca26b1cbaa.png)

# How it works...

Let's try to understand what we are trying to achieve here. This instruction imports the `nltk` library into the current program.

```py
import nltk
```

This instruction defines a new function called `builtinEngines` that takes a string parameter, `whichOne`:

```py
def builtinEngines(whichOne):
```

These if, elif, else instructions are typical branching instructions that decide which chat engine's `demo()` function is to be invoked depending on the argument that is present in the `whichOne` variable. When the user passes an unknown engine name, it displays a message to the user that it's not aware of this engine.

```py
    if whichOne == 'eliza':
        nltk.chat.eliza.demo()
    elif whichOne == 'iesha':
        nltk.chat.iesha.demo()
    elif whichOne == 'rude':
        nltk.chat.rude.demo()
    elif whichOne == 'suntsu':
        nltk.chat.suntsu.demo()
    elif whichOne == 'zen':
        nltk.chat.zen.demo()
    else:
        print("unknown built-in chat engine {}".format(whichOne))
```

It's a good practice to handle all known and unknown cases also; it makes our programs more robust in handling unknown situations.

This instruction defines a new function called `myEngine()`; this function does not take any parameters.

```py
def myEngine():
```

This is a single instruction where we are defining a nested tuple data structure and assigning it to chat pairs.

```py
    chatpairs = (
        (r"(.*?)Stock price(.*)",
            ("Today stock price is 100",
            "I am unable to find out the stock price.")),
        (r"(.*?)not well(.*)",
            ("Oh, take care. May be you should visit a doctor",
            "Did you take some medicine ?")),
        (r"(.*?)raining(.*)",
            ("Its monsoon season, what more do you expect ?",
            "Yes, its good for farmers")),
        (r"How(.*?)health(.*)",
            ("I am always healthy.",
            "I am a program, super healthy!")),
        (r".*",
            ("I am good. How are you today ?",
            "What brings you here ?"))
    )
```

Let's pay close attention to the data structure:

*   We are defining a tuple of tuples
*   Each subtuple consists of two elements:
    *   The first member is a regular expression (this is the user's question in regex format)
    *   The second member of the tuple is another set of tuples (these are the answers)

We are defining a subfunction called `chat()` inside the `myEngine()` function. This is permitted in Python. This `chat()` function displays some information to the user on the screen and calls the nltk built-in `nltk.chat.util.Chat()` class with the chatpairs variable. It passes `nltk.chat.util.reflections` as the second argument. Finally we call the `chatbot.converse()` function on the object that's created using the `chat()` class.

```py
    def chat():
        print("!"*80)
        print(" >> my Engine << ")
        print("Talk to the program using normal english")
        print("="*80)
        print("Enter 'quit' when done")
        chatbot = nltk.chat.util.Chat(chatpairs, nltk.chat.util.reflections)
        chatbot.converse()
```

This instruction calls the `chat()` function, which shows a prompt on the screen and accepts the user's requests. It shows responses according to the regular expressions that we have built before:

```py
    chat()
```

These instructions will be called when the program is invoked as a standalone program (not using import). 

```py
if __name__ == '__main__':
    for engine in ['eliza', 'iesha', 'rude', 'suntsu', 'zen']:
        print("=== demo of {} ===".format(engine))
        builtinEngines(engine)
        print()
    myEngine()
```

They do these two things:

*   Invoke the built-in engines one after another (so that we can experience them)
*   Once all the five built-in engines are excited, they call our `myEngine()`, where our customer engine comes into play

