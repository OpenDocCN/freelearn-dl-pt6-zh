

# Practical Understanding of a Corpus and Dataset

In this chapter, we'll explore the first building block of natural language processing. We are going to cover the following topics to get a practical understanding of a corpus or dataset:

*   What is corpus?
*   Why do we need corpus?
*   Understanding corpus analysis
*   Understanding types of data attributes
*   Exploring different file formats of datasets
*   Resources for access free corpus
*   Preparing datasets for NLP applications
*   Developing the web scrapping application

# What is a corpus?

Natural language processing related applications are built using a huge amount of data. In layman's terms, you can say that a large collection of data is called **corpus**. So, more formally and technically, corpus can be defined as follows:

Corpus is a collection of written or spoken natural language material, stored on computer, and used to find out how language is used. So more precisely, a corpus is a systematic computerized collection of authentic language that is used for linguistic analysis as well as corpus analysis. If you have more than one corpus, it is called **corpora**.

In order to develop NLP applications, we need corpus that is written or spoken natural language material. We use this material or data as input data and try to find out the facts that can help us develop NLP applications. Sometimes, NLP applications use a single corpus as the input, and at other times, they use multiple corpora as input.

There are many reasons of using corpus for developing NLP applications, some of which are as follows:

*   With the help of corpus, we can perform some statistical analysis such as frequency distribution, co-occurrences of words, and so on. Don't worry, we will see some basic statistical analysis for corpus later in this chapter.
*   We can define and validate linguistics rules for various NLP applications. If you are building a grammar correction system, you will use the text corpus and try to find out the grammatically incorrect instances, and then you will define the grammar rules that help us to correct those instances.
*   We can define some specific linguistic rules that depend on the usage of the language. With the help of the rule-based system, you can define the linguistic rules and validate the rules using corpus.

In a corpus, the large collection of data can be in the following formats:

*   Text data, meaning written material
*   Speech data, meaning spoken material

Let's see what exactly text data is and how can we collect the text data. Text data is a collection of written information. There are several resources that can be used for getting written information such as news articles, books, digital libraries, email messages, web pages, blogs, and so on. Right now, we all are living in a digital world, so the amount of text information is growing rapidly. So, we can use all the given resources to get the text data and then make our own corpus. Let's take an example: if you want to build a system that summarizes news articles, you will first gather various news articles present on the web and generate a collection of new articles so that the collection is your corpus for news articles and has text data. You can use web scraping tools to get information from raw HTML pages. In this chapter, we will develop one.

Now we will see how speech data is collected. A speech data corpus generally has two things: one is an audio file, and the other one is its text transcription. Generally, we can obtain speech data from audio recordings. This audio recording may have dialogues or conversations of people. Let me give you an example: in India, when you call a bank customer care department, if you pay attention, you get to know that each and every call is recorded. This is the way you can generate speech data or speech corpus. For this book, we are concentrating just on text data and not on speech data.

A corpus is also referred to as a dataset in some cases.

There are three types of corpus:

*   **Monolingual corpus:** This type of corpus has one language
*   **Bilingual corpus:** This type of corpus has two languages
*   **Multilingual corpus:** This type of corpus has more than one language

A few examples of the available corpora are given as follows:

*   Google Books Ngram corpus
*   Brown corpus
*   American National corpus

# Why do we need a corpus?

In any NLP application, we need data or corpus to building NLP tools and applications. A corpus is the most critical and basic building block of any NLP-related application. It provides us with quantitative data that is used to build NLP applications. We can also use some part of the data to test and challenge our ideas and intuitions about the language. Corpus plays a very big role in NLP applications. Challenges regarding creating a corpus for NLP applications are as follows:

*   Deciding the type of data we need in order to solve the problem statement
*   Availability of data
*   Quality of the data
*   Adequacy of the data in terms of amount

Now you may want to know the details of all the preceding questions; for that, I will take an example that can help you to understand all the previous points easily. Consider that you want to make an NLP tool that understands the medical state of a particular patient and can help generate a diagnosis after proper medical analysis.

Here, our aspect is more biased toward the corpus level and generalized. If you look at the preceding example as an NLP learner, you should process the problem statement as stated here:

*   What kind of data do I need if I want to solve the problem statement?
    *   Clinical notes or patient history
    *   Audio recording of the conversation between doctor and patient
*   Do you have this kind of corpus or data with you?
    *   If yes, great! You are in a good position, so you can proceed to the next question.
    *   If not, OK! No worries. You need to process one more question, which is probably a difficult but interesting one.
*   Is there an open source corpus available?
    *   If yes, download it, and continue to the next question.
    *   If not, think of how you can access the data and build the corpus. Think of web scraping tools and techniques. But you have to explore the ethical as well as legal aspects of your web scraping tool.
*   What is the quality level of the corpus?
    *   Go through the corpus, and try to figure out the following things:
        *   If you can't understand the dataset at all, then what to do?
            *   Spend more time with your dataset.
            *   Think like a machine, and try to think of all the things you would process if you were fed with this kind of a dataset. Don't think that you will throw an error!
            *   Find one thing that you feel you can begin with.
            *   Suppose your NLP tool has diagnosed a human disease, think of what you would ask the patient if you were the doctor's machine. Now you can start understanding your dataset and then think about the preprocessing part. Do not rush to the it.
        *   If you can understand the dataset, then what to do?
            *   Do you need each and every thing that is in the corpus to build an NLP system?
                *   If yes, then proceed to the next level, which we will look at in Chapter 5, *Feature Engineering and NLP Algorithms*.
                *   If not, then proceed to the next level, which we will look at in Chapter 4, *Preprocessing*.
*   Will the amount of data be sufficient for solving the problem statement on at least a **proof of concept** (**POC**) basis?
    *   According to my experience, I would prefer to have at least 500 MB to 1 GB of data for a small POC.
    *   For startups, to collect 500 MB to 1 GB data is also a challenge for the following reasons:
        *   Startups are new in business.
        *   Sometimes they are very innovative, and there is no ready-made dataset available.
        *   Even if they manage to build a POC, to validate their product in real life is also challenging.

Refer to *Figure 2.1* for a description of the preceding process:

![](img/e79a2f33-2530-47d0-95fe-98d93f406f93.png)

Figure 2.1: Description of the process defined under why do we need corpus?

# Understanding corpus analysis

In this section, we will first understand what corpus analysis is. After this, we will briefly touch upon speech analysis. We will also understand how we can analyze text corpus for different NLP applications. At the end, we will do some practical corpus analysis for text corpus. Let's begin!

Corpus analysis can be defined as a methodology for pursuing in-depth investigations of linguistic concepts as grounded in the context of authentic and communicative situations. Here, we are talking about the digitally stored language corpora, which is made available for access, retrieval, and analysis via computer.

Corpus analysis for speech data needs the analysis of phonetic understanding of each of the data instances. Apart from phonetic analysis, we also need to do conversation analysis, which gives us an idea of how social interaction happens in day-to-day life in a specific language. Suppose in real life, if you are doing conversational analysis for casual English language, maybe you find a sentence such as *What's up, dude?* more frequently used in conversations compared to *How are you, sir (or madam)?*.

Corpus analysis for text data consists in statistically probing, manipulating, and generalizing the dataset. So for a text dataset, we generally perform analysis of how many different words are present in the corpus and what the frequency of certain words in the corpus is. If the corpus contains any noise, we try to remove that noise. In almost every NLP application, we need to do some basic corpus analysis so we can understand our corpus well. `nltk` provides us with some inbuilt corpus. So, we perform corpus analysis using this inbuilt corpus. Before jumping to the practical part, it is very important to know what type of corpora is present in `nltk`.

`nltk` has four types of corpora. Let's look at each of them:

*   **Isolate corpus**: This type of corpus is a collection of text or natural language. Examples of this kind of corpus are `gutenberg`, `webtext`, and so on.

*   **Categorized corpus**: This type of corpus is a collection of texts that are grouped into different types of categories.
    An example of this kind of corpus is the `brown` corpus, which contains data for different categories such as news, hobbies, humor, and so on.

*   **Overlapping corpus**: This type of corpus is a collection of texts that are categorized, but the categories overlap with each other. An example of this kind of corpus is the `reuters` corpus, which contains data that is categorized, but the defined categories overlap with each other.
    More explicitly, I want to define the example of the `reuters` corpus. For example, if you consider different types of coconuts as one category, you can see subcategories of coconut-oil, and you also have cotton oil. So, in the `reuters` corpus, the various data categories are overlapped.

*   **Temporal corpus**: This type of corpus is a collection of the usages of natural language over a period of time.
    An example of this kind of corpus is the `inaugural address` corpus.
    Suppose you recode the usage of a language in any city of India in 1950\. Then you repeat the same activity to see the usage of the language in that particular city in 1980 and then again in 2017\. You will have recorded the various data attributes regarding how people used the language and what the changes over a period of time were.

Now enough of theory, let's jump to the practical stuff. You can access the following links to see the codes:

This chapter code is on the GitHub directory URL at [https://github.com/jalajthanaki/NLPython/tree/master/ch2](https://github.com/jalajthanaki/NLPython/tree/master/ch2).

Follow the Python code on this URL: [https://nbviewer.jupyter.org/github/jalajthanaki/NLPython/blob/master/ch2/2_1_Basic_corpus_analysis.html](https://nbviewer.jupyter.org/github/jalajthanaki/NLPython/blob/master/ch2/2_1_Basic_corpus_analysis.html)

The Python code has basic commands of how to access corpus using the `nltk` API. We are using the `brown` and `gutenberg` corpora. We touch upon some of the basic corpus-related APIs.

A description of the basic API attributes is given in the following table:

| **API Attributes** | **Description** |
| `fileids()` | This results in files of the corpus |
| `fileids([categories])` | This results in files of the corpus corresponding to these categories |
| `categories()` | This lists categories of the corpus |
| `categories([fileids])` | This shows categories of the corpus corresponding to these files |
| `raw()` | This shows the raw content of the corpus |
| `raw(fileids=[f1,f2,f3])` | This shows the raw content of the specified files |
| `raw(categories=[c1,c2])` | This shows the raw content of the specified categories |
| `words()` | This shows the words of the whole corpus |
| `words(fileids=[f1,f2,f3])` | This shows the words of specified `fileids` |
| `words(categories=[c1,c2])` | This shows the words of the specified categories |
| `sents()` | This shows the sentences of the whole corpus |
| `sents(fileids=[f1,f2,f3])` | This shows the sentences of specified `fileids` |
| `sents(categories=[c1,c2])` | This shows the sentences of the specified categories |
| `abspath(fileid)` | This shows the location of the given file on disk |
| `encoding(fileid)` | This shows the encoding of the file (if known) |
| `open(fileid)` | This basically opens a stream for reading the given corpus file |
| `root` | This shows a path, if it is the path to the root of the locally installed corpus |
| `readme()` | This shows the contents of the `README` file of the corpus |

We have seen the code for loading your customized corpus using `nltk` as well as done the frequency distribution for the available corpus and our custom corpus.

The `FreqDist` class is used to encode frequency distributions, which count the number of times each word occurs in a corpus.

All `nltk` corpora are not that noisy. A basic kind of preprocessing is required for them to generate features out of them. Using a basic corpus-loading API of `nltk` helps you identify the extreme level of junk data. Suppose you have a bio-chemistry corpus, then you may have a lot of equations and other complex names of chemicals that cannot be parsed accurately using the existing parsers. You can then, according to your problem statement, make a decision as to whether you should remove them in the preprocessing stage or keep them and do some customization on parsing in the **part-of-speech tagging** (**POS**) level.

In real-life applications, corpora are very dirty. Using `FreqDist`,you can take a look at how words are distributed and what we should and shouldn't consider. At the time of preprocessing, you need to check many complex attributes such as whether the results of parsing, POS tagging, and sentence splitting are appropriate or not. We will look at all these in a detailed manner in Chapter 4, *Preprocessing*, and Chapter 5, *Feature Engineering and NLP Algorithms*.

Note here that the corpus analysis is in terms of the technical aspect. We are not focusing on corpus linguistics analysis, so guys, do not confuse the two.
If you want to read more on corpus linguistics analysis, refer to this URL:
[https://en.wikipedia.org/wiki/Corpus_linguistics](https://en.wikipedia.org/wiki/Corpus_linguistics)
If you want to explore the `nltk` API more, the URL is [http://www.nltk.org/](http://www.nltk.org/).

# Exercise

1.  Calculate the number of words in the `brown` corpus with `fileID: fileidcc12`.
2.  Create your own corpus file, load it using `nltk`, and then check the frequency distribution of that corpus.

# Understanding types of data attributes

Now let's focus on what kind of data attributes can appear in the corpus. *Figure 2.3* provides you with details about the different types of data attributes:

![](img/ef5c1724-9a60-484c-a823-b2318f0368db.png)

Figure 2.3: Types of data attributes

I want to give some examples of the different types of corpora. The examples are generalized, so you guys can understand the different type of data attributes.

# Categorical or qualitative data attributes

Categorical or qualitative data attributes are as follows:

*   These kinds of data attributes are more descriptive
*   Examples are our written notes, corpora provided by `nltk`, a corpus that has recorded different types of breeds of dogs, such as collie, shepherd, and terrier

There are two sub-types of categorical data attributes:

*   **Ordinal data**:
    *   This type of data attribute is used to measure non-numeric concepts such as satisfaction level, happiness level, discomfort level, and so on.
    *   Consider the following questions, for example, which you're to answer from the options given:
        *   Question 1: How do you feel today?
        *   Options for Question 1:
            *   Very bad
            *   Bad
            *   Good
            *   Happy
            *   Very happy
        *   Now you will choose any of the given options. Suppose you choose Good, nobody can convert how good you feel to a numeric value.
    *   All the preceding options are non-numeric concepts. Hence, they lie under the category of ordinal data.
        *   Question 2: How would you rate our hotel service?
        *   Options for Question 2:
            *   Bad
            *   Average
            *   Above average
            *   Good
            *   Great
    *   Now suppose you choose any of the given options. All the aforementioned options will measure your satisfaction level, and it is difficult to convert your answer to a numeric value because answers will vary from person to person.
    *   Because one person says Good and another person says Above average, there may be a chance that they both feel the same about the hotel service but give different responses. In simple words, you can say that the difference between one option and the other is unknown. So you can't precisely decide the numerical values for these kinds of data.
*   **Nominal data**:
    *   This type of data attribute is used to record data that doesn't overlap.
    *   Example: What is your gender? The answer is either male or female, and the answers are not overlapping.
    *   Take another example: What is the color of your eyes? The answer is either black, brown, blue, or gray. (By the way, we are not considering the color lenses available in the market!)

In NLP-related applications, we will mainly deal with categorical data attributes. So, to derive appropriate data points from a corpus that has categorical data attributes is part of feature engineering. We will see more on this in Chapter 5, *Feature Engineering and NLP Algorithms*.

Some corpora contain both sub-types of categorical data.

# Numeric or quantitative data attributes

The following are numeric or quantitative data attributes:

*   These kinds of data attributes are numeric and represent a measurable quantity
*   Examples: Financial data, population of a city, weight of people, and so on

There are two sub-types of numeric data attributes:

*   **Continuous data**:
    *   These kinds of data attributes are continuous
    *   Examples: If you are recording the weight of a student, from 10 to 12 years of age, whatever data you collect about the student's weight is continuous data; Iris flower corpus
*   **Discrete data**:
    *   Discrete data can only take certain values
    *   Examples: If you are rolling two dice, you can only have the resultant values of 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, and 12; you never get 1 or 1.5 as a result if you are rolling two dice
    *   Take another example: If you toss a coin, you will get either heads or tails

These kinds of data attributes are a major part of analytics applications.

# Exploring different file formats for corpora

Corpora can be in many different formats. In practice, we can use the following file formats. All these file formats are generally used to store features, which we will feed into our machine learning algorithms later. Practical stuff regarding dealing with the following file formats will be incorporated from Chapter 4, *Preprocessing* onward. Following are the aforementioned file formats:

*   `.txt`: This format is basically given to us as a raw dataset. The `gutenberg` corpus is one of the example corpora. Some of the real-life applications have parallel corpora. Suppose you want to make Grammarly a kind of grammar correction software, then you will need a parallel corpus.
*   `.csv`: This kind of file format is generally given to us if we are participating in some hackathons or on Kaggle. We use this file format to save our features, which we will derive from raw text, and the feature `.csv` file will be used to train our machines for NLP applications.
*   `.tsv`: To understand this kind of file format usage, we will take an example. Suppose you want to make an NLP system that suggests where we should put a comma in a sentence. In this case, we cannot use the `.csv` file format to store our features because some of our feature attributes contain commas, and this will affect the performance when we start processing our feature file. You can also use any customized delimiter as well. You can put \t, ||, and so on for ease of further processing.
*   `.xml`: Some well-known NLP parsers and tools provide results in the `.xml` format. For example, the Stanford CoreNLP toolkit provides parser results in the `.xml` format. This kind of file format is mainly used to store the results of NLP applications.
*   `.json`: The Stanford CoreNLP toolkit provides its results in the `.json` format. This kind of file format is mainly used to store results of NLP applications, and it is easy to display and integrate with web applications.
*   `LibSVM`: This is one of the special file formats. Refer to the following *Figure 2.4***:**

![](img/a2f37adf-81c8-4133-95ed-33fb9385ec83.png)

Figure 2.4: LibSVM file format example

*   `LibSVM` allows for sparse training data. The non-zero values are the only ones that are included in the training dataset. Hence, the index specifies the column of the instance data (feature index). To convert from a conventional dataset, just iterate over the data, and if the value of `X(i,j)` is non-zero, print `j + 1: X(i,j)`.
*   `X(i,j)`: This is a sparse matrix:
    *   If the value of `X(i,j)` is equal to non-zero, include it in the `LibSVM` format
        *   `j+1`: This is the value of `X(i,j)`, where `j` is the column index of the matrix starting with `0`, so we add `1`
    *   Otherwise, do not include it in the `LibSVM` format
*   Let's take the following example:
    *   Example: 1 5:1 7:1 14:1 19:1
        *   Here, *1* is the class or label
        *   In the preceding example, let's focus on *5:1*, where *5* is the key, and *1* is the value; *5:1* is the key : value pair
        *   *5* is the column number or data attribute number, which is the key and is in the `LibSVM` format; we are considering only those data columns that contain non-zero values, so here, *1* is the value
        *   The values of parameters with indexes 1, 2, 3, 4, 6, and others unmentioned are 0s, so we are not including these in our example
*   This kind of data format is used in Apache Spark to train your data, and you will learn how to convert text data to the `LibSVM` format from Chapter 5, *Feature Engineering and NLP Algorithms* onwards.
*   **Customized format**: You can make your feature file using the customized file format. (Refer to the `CoNLL` dataset.) It is kind of a customized file format.
    There are many different `CoNLL` formats since `CoNLL` is a different shared task each year. *Figure 2.5* shows a data sample in the `CoNLL` format:

![](img/0a1793c5-fe04-49a4-8f14-eee22a26f6a9.png)

Figure 2.5: Data sample in CoNLL format

# Resources for accessing free corpora

Getting the corpus is a challenging task, but in this section, I will provide you with some of the links from which you can download a free corpus and use it to build NLP applications.

The `nltk` library provides some inbuilt `corpus`. To list down all the corpus names, execute the following commands:

```py
 import nltk.corpus
    dir(nltk.corpus) # Python shell print dir(nltk.corpus) # Pycharm IDE syntax

```

In *Figure 2.2*, you can see the output of the preceding code; the highlighted part indicates the name of the corpora that are already installed:

![](img/3da1717d-bc7b-4cdf-853f-920898bfb117.png)

Figure 2.2: List of all available corpora in nltk

If you guys want to use IDE to develop an NLP application using Python, you can use the PyCharm community version. You can follow its installation steps by clicking on the following URL: [https://github.com/jalajthanaki/NLPython/blob/master/ch2/Pycharm_installation_guide.md](https://github.com/jalajthanaki/NLPython/blob/master/ch2/Pycharm_installation_guide.md)

If you want to explore more corpus resources, take a look at *Big Data: 33 Brilliant and Free Data Sources for 2016*, Bernard Marr ([https://www.forbes.com/sites/bernardmarr/2016/02/12/big-data-35-brilliant-and-free-data-sources-for-2016/#53369cd5b54d](https://www.forbes.com/sites/bernardmarr/2016/02/12/big-data-35-brilliant-and-free-data-sources-for-2016/#53369cd5b54d)).

Until now, we have looked at a lot of basic stuff. Now let me give you an idea of how we can prepare a dataset for a natural language processing applications, which will be developed with the help of machine learning.

# Preparing a dataset for NLP applications

In this section, we will look at the basic steps that can help you prepare a dataset for NLP or any data science applications. There are basically three steps for preparing your dataset, given as follows:

*   Selecting data
*   Preprocessing data
*   Transforming data

# Selecting data

Suppose you are working with world tech giants such as Google, Apple, Facebook, and so on. Then you could easily get a large amount of data, but if you are not working with giants and instead doing independent research or learning some NLP concepts, then how and from where can you get a dataset? First, decide what kind of dataset you need as per the NLP application that you want to develop. Also, consider the end result of the NLP application that you are trying to build. If you want to make a chatbot for the healthcare domain, you should not use a dialog dataset of banking customer care. So, understand your application or problem statement thoroughly.

You can use the following links to download free datasets:
[https://github.com/caesar0301/awesome-public-datasets](https://github.com/caesar0301/awesome-public-datasets).
[https://www.kaggle.com/datasets](https://www.kaggle.com/datasets).
[https://www.reddit.com/r/datasets/](https://www.reddit.com/r/datasets/).

You can also use the Google Advanced Search feature, or you can use Python web scraping libraries such as `beautifulsoup` or `scrapy`.

After selecting the dataset as per the application, you can move on to the next step.

# Preprocessing the dataset

In this step, we will do some basic data analysis, such as which attributes are available in the dataset. This stage has three sub-stages, and we will look at each of them. You will find more details about the preprocessing stage in Chapter 4, *Preprocessing*. Here, I'll give you just the basic information.

# Formatting

In this step, generate the dataset format that you feel most comfortable working with. If you have a dataset in the JSON format and you feel that you are most comfortable working with CSV, then convert the dataset from JSON to CSV.

# Cleaning

In this step, we clean the data. If the dataset has missing values, either delete that data record or replace it with the most appropriate nearest value. If you find any unnecessary data attributes, you can remove them as well. Suppose you are making a grammar correction system, then you can remove the mathematical equations from your dataset because your grammar correction application doesn't use equations.

# Sampling

In this stage, you can actually try to figure out which of the available data attributes our present dataset has and which of the data attributes can be derived by us. We are also trying to figure out what the most important data attributes are as per our application. Suppose we are building a chatbot. We will then try to break down sentences into words so as to identify the keywords of the sentence. So, the word-level information can be derived from the sentence, and both word-level and sentence level information are important for the chatbot application. As such, we do not remove sentences, apart from junk sentences. Using sampling, we try to extract the best data attributes that represent the overall dataset very well.

Now we can look at the last stage, which is the transformation stage.

# Transforming data

In this stage, we will apply some feature engineering techniques that help us convert the text data into numeric data so the machine can understand our dataset and try to find out the pattern in the dataset. So, this stage is basically a data manipulation stage. In the NLP domain, for the transformation stage, we can use some encoding and vectorization techniques. Don't get scared by the terminology. We will look at all the data manipulation techniques and feature extraction techniques in Chapter 5, *Feature Engineering and NLP Algorithms* and Chapter 6, *Advance Feature Engineering and NLP Algorithms*.

All the preceding stages are basic steps to prepare the dataset for any NLP or data science related applications. Now, let's see how you can generate data using web scraping.

# Web scraping

To develop a web scraping tool, we can use libraries such as `beautifulsoup` and `scrapy`. Here, I'm giving some of the basic code for web scraping.

Take a look at the code snippet in *Figure 2.6,* which is used to develop a basic web scraper using `beautifulsoup`:

![](img/b319fd5a-5d4e-4fc7-a6e0-0f461633119e.png)

Figure 2.6: Basic web scraper tool using beautifulsoup

The following *Figure 2.7* demonstrates the output:

![](img/f53b1140-5385-448d-a2f1-3e45601e6d69.png)

Figure 2.7: Output of basic web scraper using beautifulsoup

You can find the installation guide for `beautifulsoup` and `scrapy` at this link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch2/Chapter_2_Installation_Commands.txt](https://github.com/jalajthanaki/NLPython/blob/master/ch2/Chapter_2_Installation_Commands.txt).

You can find the code at this link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch2/2_2_Basic_webscraping_byusing_beautifulsuop.py](https://github.com/jalajthanaki/NLPython/blob/master/ch2/2_2_Basic_webscraping_byusing_beautifulsuop.py).

If you get any warning while running the script, it will be fine; don't worry about warnings.

Now, let's do some web scraping using `scrapy`. For that, we need to create a new scrapy project.

Follow the command to create the scrapy project. Execute the following command on your terminal:

```py
  $ scrapy startproject project_name

```

I'm creating a scrapy project with the `web_scraping_test` name; the command is as follows:

```py
  $ scrapy startproject web_scraping_test

```

Once you execute the preceding command, you can see the output as shown in *Figure 2.8*:

![](img/772c2f47-79c0-4e47-a8ac-41d87f2cfaf3.png)

Figure 2.8: Output when you create a new scrapy project

After creating a project, perform the following steps:

1.  Edit your `items.py` file, which has been created already.
2.  Create the `WebScrapingTestspider` file inside the `spiders` directory.
3.  Go to the website page that you want to scrape, and select `xpath` of the element. You can read more on the `xpath` selector by clicking at this link:
    [https://doc.scrapy.org/en/1.0/topics/selectors.html](https://doc.scrapy.org/en/1.0/topics/selectors.html)[](https://doc.scrapy.org/en/1.0/topics/selectors.html)

Take a look at the code snippet in *Figure 2.9.* Its code is available at the GitHub URL:

[https://github.com/jalajthanaki/NLPython/tree/master/web_scraping_test](https://github.com/jalajthanaki/NLPython/tree/master/web_scraping_test)

![](img/d1164bb1-7d61-4f42-8ecd-4e4b24036caf.png)

Figure 2.9: The items.py file where we have defined items we need to scrape

*Figure 2.10* is used to develop a basic web scraper using `scrapy`:

![](img/087e8c20-48b8-4769-848c-6669c2aa842a.png)

Figure 2.10: Spider file containing actual code

*Figure 2.11* demonstrates the output, which is in the form of a CSV file:

![](img/10379c97-edd6-44dd-a3b8-614e61d26f2e.png)

Figure 2.11: Output of scraper is redirected to a CSV file

If you get any SSL-related warnings, refer to the answer at this link:

[https://stackoverflow.com/questions/29134512/insecureplatformwarning-a-true-sslcontext-object-is-not-available-this-prevent](https://stackoverflow.com/questions/29134512/insecureplatformwarning-a-true-sslcontext-object-is-not-available-this-prevent)

You can develop a web scraper that bypasses AJAX and scripts, but you need to be very careful when you do this because you need to keep in mind that you are not doing anything unethical. So, here, we are not going to cover the part on bypassing AJAX and scripts and scraping data. Out of curiosity, you can search on the web how people actually do this. You can use the `Selenium` library to do automatic clicking to perform web events.

# Summary

In this chapter, we saw that a corpus is the basic building block for NLP applications. We also got an idea about the different types of corpora and their data attributes. We touched upon the practical analysis aspects of a corpus. We used the `nltk` API to make corpus analysis easy.

In the next chapter, we will address the basic and effective aspects of natural language using linguistic concepts such as parts of speech, lexical items, and tokenization, which will further help us in preprocessing and feature engineering.
