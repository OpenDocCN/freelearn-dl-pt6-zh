
# Preprocessing

From here, all our chapters will mostly contain code. I want to remind all my readers to run and develop the code at their end. Let's start the coding ninja journey.

In this chapter, we will be learning how to do preprocessing according to the different NLP applications. We will learn the following topics:

*   Handling corpus-raw text
*   Handling corpus-raw sentences
*   Basic preprocessing
*   Practical and customized preprocessing

# Handling corpus-raw text

In this section, we will see how to get the raw text and, in the following section, we will preprocess text and identify the sentences.

The process for this section is given in *Figure 4.1*:

![](img/85eda8bb-a6e5-4c06-9569-9b370ab22d78.png)

Figure 4.1: Process of handling corpus-raw text

# Getting raw text

In this section, we will use three sources where we can get the raw text data.

The following are the data sources:

*   Raw text file
*   Define raw data text inside a script in the form of a local variable
*   Use any of the available corpus from `nltk`

Let's begin:

*   Raw text file access: I have a `.txt` file saved on my local computer which contains text data in the form of a paragraph. I want to read the content of that file and then load the content as the next step. I will run a sentence tokenizer to get the sentences out of it.
*   Define raw data text inside a script in the form of a local variable: If we have a small amount of data, then we can assign the data to a local string variable. For example: **Text = This is the sentence, this is another example**.
*   Use an available corpus from `nltk`: We can import an available corpus such as the `brown` corpus, `gutenberg` corpus, and so on from `nltk` and load the content.

I have defined three functions:

*   `fileread()`: This reads the content of a file
*   `localtextvalue()`: This loads locally defined text
*   `readcorpus()`: This reads the `gutenberg` corpus content

Refer to the code snippet given in *Figure 4.2*, which describes all the three cases previously defined:

![](img/fa9692ee-16ab-4ad2-be27-3a636801b560.png)

Figure 4.2: Various ways to get the raw data

You can find the code by clicking on the GitHub link: [https://github.com/jalajthanaki/NLPython/blob/master/ch4/4_1_processrawtext.py](https://github.com/jalajthanaki/NLPython/blob/master/ch4/4_1_processrawtext.py)

# Lowercase conversion

Converting all your data to lowercase helps in the process of preprocessing and in later stages in the NLP application, when you are doing parsing.

So, converting the text to its lowercase format is quite easy. You can find the code on this GitHub link: [https://github.com/jalajthanaki/NLPython/blob/master/ch4/4_4_wordtokenization.py](https://github.com/jalajthanaki/NLPython/blob/master/ch4/4_4_wordtokenization.py)

You can find the code snippet in *Figure 4.3*:

![](img/0ab65a96-ba98-434d-925a-2309c62f409f.png)

Figure 4.3: Converting data to lowercase

The output of the preceding code snippet is as follows:

```py
----------converting data to lower case ---------- 
i am a person. do you know what is time now? 

```

# Sentence tokenization

In raw text data, data is in paragraph form. Now, if you want the sentences from the paragraph, then you need to tokenize at sentence level.

Sentence tokenization is the process of identifying the boundary of the sentences. It is also called **sentence boundary detection** or **sentence segmentation** or **sentence boundary disambiguation**. This process identifies the sentences starting and ending points.

Some of the specialized cases need a customized rule for the sentence tokenizer as well.

The following open source tools are available for performing sentence tokenization:

*   OpenNLP
*   Stanford CoreNLP
*   GATE
*   nltk

Here we are using the `nltk` sentence tokenizer.

We are using `sent_tokenize` from `nltk` and will import it as `st`:

*   `sent_tokenize(rawtext)`: This takes a raw data string as an argument
*   `st(filecontentdetails)`: This is our customized raw data, which is provided as an input argument

You can find the code on this GitHub Link: [https://github.com/jalajthanaki/NLPython/blob/master/ch4/4_1_processrawtext.py](https://github.com/jalajthanaki/NLPython/blob/master/ch4/4_1_processrawtext.py).

You can see the code in the following code snippet in *Figure 4.4*:

![](img/9b94d00c-8a6e-480b-bebf-722e48dc8940.png)

Figure 4.4: Code snippet for nltk sentence tokenizer

# Challenges of sentence tokenization

At first glance, you would ask, what's the big deal about finding out the sentence boundary from the given raw text?

Sentence tokenization varies from language to language.

Things get complicated when you have the following scenarios to handle. We are using examples to explain the cases:

*   If there is small letter after a dot, then the sentence should not split after the dot. The following is an example:
    *   Sentence: He has completed his Ph.D. degree. He is happy.
    *   In the preceding example, the sentence tokenizer should split the sentence after **degree**, not after **Ph.D.**
*   If there is a small letter after the dot, then the sentence should be split after the dot. This is a common mistake. Let's take an example:
    *   Sentence: This is an apple.an apple is good for health.
    *   In the preceding example, the sentence tokenizer should split the sentence after **apple**.
*   If there is an initial name in the sentence, then the sentence should not split after the initials:
    *   Sentence: Harry Potter was written by J.K. Rowling. It is an entertaining one.
    *   In the preceding example, the sentence should not split after **J.** It should ideally split after **Rowling**.

*   Grammarly Inc., the grammar correction software, customized a rule for the identification of sentences and achieves high accuracy for sentence boundary detection. See the blog link:
    [https://tech.grammarly.com/blog/posts/How-to-Split-Sentences.html](https://tech.grammarly.com/blog/posts/How-to-Split-Sentences.html).

To overcome the previous challenges, you can take the following approaches, but the accuracy of each approach depends on the implementation. The approaches are as follows:

*   You can develop a rule-based system to increase the performance of the sentence tokenizer:
    *   For the previous approach, you can use **name entity recognition** (**NER**) tools, POS taggers, or parsers, and then analyze the output of the described tools, as well as the sentence tokenizer output and rectify where the sentence tokenizer went wrong. With the help of NER tools, POS taggers, and parsers, can you fix the wrong output of the sentence tokenizer. In this case, write a rule, then code it, and check whether the output is as expected.
    *   Test your code! You need to check for exceptional cases. Does your code perform well? If yes, great! And, if not, change it a bit:
        *   You can improve the sentence tokenizer by using **machine learning** (**ML**) or deep learning techniques:
            *   If you have enough data that is annotated by a human, then you can train the model using an annotated dataset. Based on that trained model, we can generate a new prediction from where the sentence boundary should end.
            *   In this method, you need to check how the model will perform.

# Stemming for raw text

As we saw in [Chapter 3](f65e61fc-1d20-434f-b606-f36cf401fc41.xhtml), *Understanding Structure of Sentences*, stemming is the process of converting each word of the sentence to its root form by deleting or replacing suffixes.

In this section, we will apply the `Stemmer` concept on the raw text.

Here, we have code where we are using the `PorterStemmer` available in `nltk`. Refer to *Figure 4.5*:

![](img/7c3b44ed-01b0-4319-a4d1-9548c1221033.png)

Figure 4.5: PorterStemmer code for raw text

The output of the preceding code is:

```py
stem is funnier than a bummer say the sushi love comput scientist. she realli want to buy cars. she told me angrily. 

```

When you compare the preceding output with the original text, then we can see the following changes:

```py
Stemming is funnier than a bummer says the sushi loving computer scientist. She really wants to buy cars. She told me angrily. 

```

If you want to see the difference, then you can refer to the highlighted words to see the difference.

# Challenges of stemming for raw text

Initially, stemming tools were made for the English language. The accuracy of stemming tools for the English language is high, but for languages such as Urdu and Hebrew, stemming tools do not perform well. So, to develop stemming tools for other languages is quite challenging. It is still an open research area.

# Lemmatization of raw text

Lemmatization is the process that identifies the correct intended **part-of-speech** (**POS**) and the meaning of words that are present in sentences.

In lemmatization, we remove the inflection endings and convert the word into its base form, present in a dictionary or in the vocabulary. If we use vocabulary and morphological analysis of all the words present in the raw text properly, then we can get high accuracy for lemmatization.

Lemmatization transforms words present in the raw text to its lemma by using a tagged dictionary such as WordNet.

Lemmatization is closely related to stemming.

In lemmatization, we consider POS tags, and in stemming we do not consider POS tags and the context of words.

Let's take some examples to make the concepts clear. The following are the sentences:

*   Sentence 1: It is better for you.
    *   There is a word **better** present in sentence 1\. So, the lemma of word **better** is as **good** as a lemma. But stemming is missing as it requires a dictionary lookup.
*   Sentence 2: Man is walking.
    *   The word **walking** is derived from the base word walk and here, stemming and lemmatization are both the same.
*   Sentence 3: We are meeting tomorrow.
    *   Here, to meet is the base form. The word **meeting** is derived from the base form. The base form meet can be a noun or it can be a verb. So it depends on the context it will use. So, lemmatization attempts to select the right lemma based on their POS tags.
*   Refer to the code snippet in *Figure 4.6* for the lemmatization of raw text:

![](img/e48d0681-646c-4cf2-b4de-42c921b1f663.png)
Figure 4.6: Stemming and lemmatization of raw text

The output of the preceding code is given as follows:

The given input is:

```py
text = """Stemming is funnier than a bummer says the sushi loving computer scientist.She really wants to buy cars. She told me angrily.It is better for you. Man is walking. We are meeting tomorrow.""" 

```

The output is given as:

```py
Stemmer 
stem is funnier than a bummer say the sushi love comput scientist. she realli want to buy cars. she told me angrily. It is better for you. man is walking. We are meet tomorrow. 
Verb lemma 
Stemming be funnier than a bummer say the sushi love computer scientist. She really want to buy cars. She tell me angrily. It be better for you. Man be walking. We be meet tomorrow. 
Noun lemma 
Stemming is funnier than a bummer say the sushi loving computer scientist. She really want to buy cars. She told me angrily. It is better for you. Man is walking. We are meeting tomorrow. 
Adjective lemma 
Stemming is funny than a bummer says the sushi loving computer scientist. She really wants to buy cars. She told me angrily. It is good for you. Man is walking. We are meeting tomorrow. 
Satellite adjectives lemma 
Stemming is funny than a bummer says the sushi loving computer scientist. She really wants to buy cars. She told me angrily. It is good for you. Man is walking. We are meeting tomorrow. 
Adverb lemma 
Stemming is funnier than a bummer says the sushi loving computer scientist. She really wants to buy cars. She told me angrily. It is well for you. Man is walking. We are meeting tomorrow. 

```

In lemmatization, we use different POS tags. The abbreviation description is as follows:

*   `v` stands for verbs
*   `n` stands for nouns
*   `a` stands for adjectives
*   `s` stands for satellite adjectives
*   `r` stands for adverbs

You can see that, inside the `lemmatizer()`function, I have used all the described POS tags.

You can download the code from the GitHub link at: [https://github.com/jalajthanaki/NLPython/blob/master/ch4/4_2_rawtext_Stemmers.py](https://github.com/jalajthanaki/NLPython/blob/master/ch4/4_2_rawtext_Stemmers.py).

# Challenges of lemmatization of raw text

Lemmatization uses a tagged dictionary such as WordNet. Mostly, it's a human-tagged dictionary. So human efforts and the time it takes to make WordNet for different languages is challenging.

# Stop word removal

Stop word removal is an important preprocessing step for some NLP applications, such as sentiment analysis, text summarization, and so on.

Removing stop words, as well as removing commonly occurring words, is a basic but important step. The following is a list of stop words which are going to be removed. This list has been generated from `nltk`. Refer to the following code snippet in Figure 4.7:

![](img/1982b360-e918-4f5b-b9cd-f476c91600cd.png)

Figure 4.7: Code to see the list of stop words for the English language

The output of the preceding code is a list of stop words available in `nltk`, refer to *Figure 4.8*:

![](img/6fba5fc2-d654-4d6a-b4de-e384e8b48194.png)

Figure 4.8: Output of nltk stop words list for the English language

The `nltk` has a readily available list of stop words for the English language. You can also customize which words you want to remove according to the NLP application that you are developing.

You can see the code snippet for removing customized stop words in *Figure 4.9*:

![](img/deeb5441-8493-4495-a956-3c8034eba7c3.png)

Figure 4.9: Removing customized stop words

The output of the code given in *Figure 4.9* is as follows:

```py
this is foo. 

```

The code snippet in *Figure 4.10* performs actual stop word removal from raw text and this raw text is in the English language:

![](img/e99189e3-380d-4a09-9297-c824f24be3ce.png)

Figure 4.10: Stop words removal from raw text

The output of the preceding code snippet is as follows:

```py
Input raw sentence: ""this is a test sentence. I am very happy today."" 
--------Stop word removal from raw text--------- 
test sentence. happy today. 

```

# Exercise

Take a file which is placed in the data folder with the name `rawtextcorpus.txt`, open the file in read mode, load the content, and then remove the stop words by using the nltk stop word list. Please analyze the output to get a better idea of how things are working out.

Up until this section, we have analyzed raw text. In the next section, we will do preprocessing on sentence levels and word levels.

# Handling corpus-raw sentences

In the previous section, we were processing on raw text and looked at concepts at the sentence level. In this section, we are going to look at the concepts of tokenization, lemmatization, and so on at the word level.

# Word tokenization

Word tokenization is defined as the process of chopping a stream of text up into words, phrases, and meaningful strings. This process is called **word tokenization**. The output of the process are words that we will get as an output after tokenization. This is called a **token**.

Let's see the code snippet given in *Figure 4.11* of tokenized words:

![](img/d6b59dcf-3062-40f4-8d65-d0203b591ff3.png)

Figure 4.11: Word tokenized code snippet

The output of the code given in *Figure 4.11* is as follows:

The input for word tokenization is:

```py
Stemming is funnier than a bummer says the sushi loving computer scientist.She really wants to buy cars. She told me angrily. It is better for you.Man is walking. We are meeting tomorrow. You really don''t know..! 

```

The output for word tokenization is:

```py
[''Stemming'', ''is'', ''funnier'', ''than'', ''a'', ''bummer'', ''says'', ''the'', ''sushi'', ''loving'', ''computer'', ''scientist'', ''.'', ''She'', ''really'', ''wants'', ''to'', ''buy'', ''cars'', ''.'', ''She'', ''told'', ''me'', ''angrily'', ''.'', ''It'', ''is'', ''better'', ''for'', ''you'', ''.'', ''Man'', ''is'', ''walking'', ''.'', ''We'', ''are'', ''meeting'', ''tomorrow'', ''.'', ''You'', ''really'', ''do'', ""n''t"", ''know..'', ''!''] 

```

# Challenges for word tokenization

If you analyze the preceding output, then you can observe that the word `don't` is tokenized as `do, n't know`. Tokenizing these kinds of words is pretty painful using the `word_tokenize` of `nltk`.

To solve the preceding problem, you can write exception codes and improvise the accuracy. You need to write pattern matching rules, which solve the defined challenge, but are so customized and vary from application to application.

Another challenge involves some languages such as Urdu, Hebrew, Arabic, and so on. They are quite difficult in terms of deciding on the word boundary and find out meaningful tokens from the sentences.

# Word lemmatization

Word lemmatization is the same concept that we defined in the first section. We will just do a quick revision of it and then we will implement lemmatization on the word level.

Word lemmatization is converting a word from its inflected form to its base form. In word lemmatization, we consider the POS tags and, according to the POS tags, we can derive the base form which is available to the lexical WordNet.

You can find the code snippet in *Figure 4.12*:

![](img/6822ae40-ec0e-4d7b-bb99-5a1ceedd8318.png)

Figure 4.12: Word lemmatization code snippet

The output of the word lemmatization is as follows:

```py
Input is: wordlemma.lemmatize(''cars'')  Output is: car 
Input is: wordlemma.lemmatize(''walking'',pos=''v'') Output is: walk 
Input is: wordlemma.lemmatize(''meeting'',pos=''n'') Output is: meeting 
Input is: wordlemma.lemmatize(''meeting'',pos=''v'') Output is: meet 
Input is: wordlemma.lemmatize(''better'',pos=''a'') Output is: good 

```

# Challenges for word lemmatization

It is time consuming to build a lexical dictionary. If you want to build a lemmatization tool that can consider a larger context, taking into account the context of preceding sentences, it is still an open area in research.

# Basic preprocessing

In basic preprocessing, we include things that are simple and easy to code but seek our attention when we are doing preprocessing for NLP applications.

# Regular expressions

Now we will begin some of the interesting concepts of preprocessing, which are the most useful. We will look at some of the advanced levels of regular expression.

For those who are new to regular expression, I want to explain the basic concept of **regular expression** (**regex**).

Regular expression is helpful to find or find-replace specific patterns from a sequence of characters. There is particular syntax which you need to follow when you are making regex.

There are many online tools available which can give you the facility to develop and test your regex. One of my favorite online regex development tool links is given here: [https://regex101.com/](https://regex101.com/)

You can also refer to the Python regex library documentation at: [https://docs.python.org/2/library/re.html](https://docs.python.org/2/library/re.html)

# Basic level regular expression

Regex is a powerful tool when you want to do customized preprocessing or when you have noisy data with you.

Here, I'm presenting some of the basic syntax and then we will see the actual implementation on Python. In Python, the `re` library is available and by using this library we can implement regex. You can find the code on this GitHub link: [https://github.com/jalajthanaki/NLPython/blob/master/ch4/4_5_regualrexpression.py](https://github.com/jalajthanaki/NLPython/blob/master/ch4/4_5_regualrexpression.py)

# Basic flags

The basic flags are `I`, `L`, `M`, `S`, `U`, `X`:

*   `re.I`: This flag is used for ignoring casing
*   `re.M`: This flag is useful if you want to find patterns throughout multiple lines
*   `re.L`: This flag is used to find a local dependent
*   `re.S`: This flag is used to find dot matches
*   `re.U`: This flag is used to work for unicode data
*   `re.X`: This flag is used for writing regex in a more readable format

We have mainly used `re.I`, `re.M`, `re.L`, and `re.U` flags.

We are using the `re.match()` and `re.search()` functions. Both are used to find the patterns and then you can process them according to the requirements of your application.

Let's look at the differences between `re.match()` and `re.search()`:

*   `re.match()`: This checks for a match of the string only at the beginning of the string. So, if it finds the pattern at the beginning of the input string then it returns the matched pattern, otherwise; it returns a noun.
*   `re.search()`: This checks for a match of the string anywhere in the string. It finds all the occurrences of the pattern in the given input string or data.

Refer to the code snippet given in *Figure 4.13*:

![](img/9f086497-45e5-4f5c-8820-635c14dfa4e1.png)

Figure 4.13: Code snippet to see the difference between re.match() versus re.search()

The output of the code snippet of *Figure 4.13* is given in *Figure 4.14*:

![](img/40421689-c806-48dc-89a5-8d7fc8d6814c.png)

Figure 4.14: Output of the re.match() versus re.search()

The syntax is as follows:

Find the single occurrence of character `a` and `b`:

```py
Regex: [ab] 

```

Find characters except `a` and `b`:

```py
Regex: [^ab] 

```

Find the character range of `a` to `z`:

```py
Regex: [a-z] 

```

Find range except to `z`:

```py
Regex: [^a-z] 

```

Find all the characters `a` to `z` as well as `A` to `Z`:

```py
Regex: [a-zA-Z] 

```

Any single character:

```py
Regex: . 

```

Any whitespace character:

```py
Regex: \s 

```

Any non-whitespace character:

```py
Regex: \S 

```

Any digit:

```py
Regex: \d 

```

Any non-digit:

```py
Regex: \D 

```

Any non-words:

```py
Regex: \W 

```

Any words:

```py
Regex: \w 

```

Either match `a` or `b`:

```py
Regex: (a|b) 

```

Occurrence of `a` is either zero or one:

```py
Regex: a? ; ? Matches  zero or one occurrence not more than 1 occurrence 

```

Occurrence of `a` is zero time or more than that:

```py
Regex: a* ; * matches zero or more than that 

```

Occurrence of `a` is one time or more than that:

```py
Regex: a+ ; + matches occurrences one or more that one time 

```

Exactly match three occurrences of `a`:

```py
Regex: a{3} 

```

Match simultaneous occurrences of `a` with `3` or more than `3`:

```py
Regex: a{3,} 

```

Match simultaneous occurrences of `a` between `3` to `6`:

```py
Regex: a{3,6} 

```

Starting of the string:

```py
Regex: ^ 

```

Ending of the string:

```py
Regex: $ 

```

Match word boundary:

```py
Regex: \b 

```

Non-word boundary:

```py
Regex: \B 

```

The basic code snippet is given in *Figure 4.15*:

![](img/e4e39a15-d25a-497a-b904-8a9ccc4da46c.png)

Figure 4.15: Basic regex functions code snippet

The output of the code snippet of *Figure 4.15* is given in *Figure 4.16*:

![](img/6a83361e-378b-4931-b811-311f6be3cce7.png)

Figure 4.16: Output of the basic regex function code snippet

# Advanced level regular expression

There are advanced concepts of regex which will be very useful.

The **lookahead** and **lookbehind** are used to find out substring patterns from your data. Let's begin. We will understand the concepts in the basic language. Then we will look at the implementation of them.

# Positive lookahead

Positive lookahead matches the substring from a string if the defined pattern is followed by the substring. If you don't understand, then let's look at the following example:

*   Consider a sentence: I play on the playground.
*   Now, you want to extract *play* as a pattern but only if it follows *ground*. In this situation, you can use positive lookahead.

The syntax of positive lookahead is `(?=pattern)`

The regex `rplay(?=ground)` matches *play*, but only if it is followed by *ground*. Thus, the first *play* in the text string won't be matched.

# Positive lookbehind

Positive lookbehind matches the substring from a string if the defined pattern is preceded by the substring. Refer to the following example:

*   Consider the sentence: I play on the playground. It is the best ground.
*   Now you want to extract *ground*, if it is preceded by the string *play*. In this case, you can use positive lookbehind.

The syntax of positive lookbehind is `(?<=pattern)`

The regex `r(?<=play)ground` matches *ground*, but only if it is preceded by *play*.

# Negative lookahead

Negative lookahead matches the string which is definitely not followed by the pattern which we have defined in the regex pattern part.

Let's give an example to understand the negative lookahead:

*   Consider the sentence: I play on the playground. It is the best ground.
*   Now you want to extract *play* only if it is not followed by the string *ground*. In this case, you can use negative lookahead.

The syntax of negative lookahead is `(?!pattern)`

The regex `r play(?!ground)` matches *play*, but only if it is not followed by *ground*. Thus the *play* just before *on* is matched.

# Negative lookbehind

Negative lookbehind matches the string which is definitely not preceded by the pattern which we have defined in the regex pattern part.

Let's see an example to understand the negative lookbehind:

*   Consider the sentence: I play on the playground. It is the best ground.
*   Now you want to extract *ground* only if it is not preceded by the string *play*. In this case, you can use negative lookbehind.

The syntax of negative lookbehind is `(?<!pattern)`

The regex `r(?<!play)ground` matches *ground*, but only if it is not preceded by *play*.

You can see the code snippet which is an implementation of `advanceregex()` in *Figure 4.17*:

![](img/74f4c83a-883b-4b81-89ca-d529aa3f8f23.png)

Figure 4.17: Advanced regex code snippet

The output of the code snippet of *Figure 4.17* is given in *Figure 4.18*:

![](img/6efff615-cc21-4535-9207-b0fd8f01e31d.png)

Figure 4.18: Output of code snippet of advanced regex

# Practical and customized preprocessing

When we start preprocessing for NLP applications, sometimes you need to do some customization according to your NLP application. At that time, it might be possible that you need to think about some of the points which I have described as follows.

# Decide by yourself

This section is a discussion of how to approach preprocessing when you don't know what kind of preprocessing is required for developing an NLP application. In this kind of situation, what you can do is simply ask the following questions to yourself and make a decision.

What is your NLP application and what kind of data do you need to build the NLP application?

*   Once you have understood the problem statement, as well as having clarity on what your output should be, then you are in a good situation.
*   Once you know about the problem statement and the expected output, now think what all the data points are that you need from your raw data set.
*   To understand the previous two points, let's take an example. If you want to make a text summarization application, suppose you are using a news articles that are on the web, which you want to use for building news text summarization application. Now, you have built a scraper that scrapes news articles from the web. This raw news article dataset may contain HTML tags, long texts, and so on.

For news text summarization, how will we do preprocessing? In order to answer that, we need to ask ourselves a few questions. So, let's jump to a few questions about preprocessing.

# Is preprocessing required?

*   Now you have raw-data for text summarization and your dataset contains HTML tags, repeated text, and so on.
*   If your raw-data has all the content that I described in the first point, then preprocessing is required and, in this case, we need to remove HTML tags and repeated sentences; otherwise, preprocessing is not required.
*   You also need to apply lowercase convention.
*   After that, you need to apply sentence tokenizer on your text summarization dataset.
*   Finally, you need to apply word tokenizer on your text summarization dataset.
*   Whether your dataset needs preprocessing depends on your problem statement and what data your raw dataset contains.

You can see the flowchart in *Figure 4.19:*

![](img/36192456-01df-4e30-91bc-3261e57a43e2.png)

Figure 4.19: Basic flowchart for performing preprocessing of text-summarization

# What kind of preprocessing is required?

In our example of text summarization, if a raw dataset contains HTML tags, long text, repeated text, then during the process of developing your application, as well as in your output, you don't need the following data:

*   You don't need HTML tags, so you can remove them
*   You don't need repeated sentences, so you can remove them as well
*   If there is long text content then if you can find stop words and high frequency small words, you should remove them

# Understanding case studies of preprocessing

Whatever I have explained here regarding customized preprocessing will make more sense to you when you have some real life case studies explained.

# Grammar correction system

*   You are making a grammar correction system. Now, think of the sub-task of it. You want to build a system which predicts the placement of articles a, an, and the in a particular sentence.
*   For this kind of system, if you are thinking I need to remove stop words every time, then, OOPs, you are wrong because this time we really can't remove all the stop words blindly. In the end, we need to predict the articles a, an, and the.
*   You can remove words which are not meaningful at all, such as when your dataset contains math symbols, then you can remove them. But this time, you need to do a detailed analysis as to whether you can remove the small length words, such as abbreviations, because your system also needs to predict which abbreviations don't take an article and which do.

Now, let's look at a system where you can apply all the preprocessing techniques that we have described here. Let's follow the points inside sentiment analysis.

# Sentiment analysis

Sentiment analysis is all about evaluating the reviews of your customers and categorizing them into positive, negative, and neutral categories:

*   For this kind of system, your dataset contains user reviews so user writing generally contains casual language.
*   The data contains informal language so we need to remove stop words such as Hi, Hey, Hello, and so on. We do not use Hi, Hey, How are u? to conclude whether the user review is positive, negative, or neutral.
*   Apart from that, you can remove the repeated reviews.
*   You can also preprocess data by using word tokenization and lemmatization.

# Machine translation

Machine translation is also one of the widely used NLP applications. In machine translation, our goal is to translate one language to another language in a logical manner. So, if we want to translate the English language to the German language, then you may the apply the following preprocessing steps:

1.  We can apply convert to the whole dataset to be converted into lowercase.
2.  Apply sentence splitter on the dataset so you can get the boundary for each of the sentences.
3.  Now, suppose you have corpus where all English sentences are in `English_Sentence_File` and all German sentence are in `German_Sentence_File`. Now, you know for each English sentence there is a corresponding German sentence present in `German_Sentence_File`. This kind of corpus is called **parallel** corpus. So in this case, you also need to check that all sentences in both files are aligned appropriately.
4.  You can also apply stemming for each of the words of the sentences.

# Spelling correction

Spelling correction can be a very useful tool for preprocessing as well, as it helps to improve your NLP application.

# Approach

The concept of spelling correction came from the concept of how much similarity is contained by two strings. This concept is used to compare two strings. The same concept has been used everywhere nowadays. We will consider some examples to better understand how this concept of checking the similarity of two strings can be helpful to us.

When you search on Google, if you make a typing mistake in your search query, then you get a suggestion on the browser, Did you mean: with your corrected query with the right spelling. This mechanism rectifies your spelling mistake and Google has its own way of providing almost perfect results every time. Google does not just do a spelling correction, but is also indexes on your submitted query and displays the best result for you. So, the concept behind the spelling correction is the similarity between two strings.

Take another example: If you are developing a machine translation system, then when you see the string translated by the machine, your next step is probably to validate your output. So now you will compare the output of the machine with a human translator and situation, which may not be perfectly similar to the output of the machine.

If the machine translated string is: **She said could she help me?**, the human string translated would say: **She asked could she help me?** When you are checking the similarity between a system string and a human string, you may find that *said* is replace by asked.

So, this concept of the similarity of two strings can be used in many applications, including speech recognition, NLP applications, and so on.

There are three major operations when we are talking about measuring the similarity of two strings. The operations are insertion, deletion, and substitution. These operations are used for the implementation of the spelling correction operation. Right now, to avoid complexity, we are not considering transpose and long string editing operations.

Let's start with the operations and then we will look at the algorithm specifically for the spelling correction.

**Insertion operation**

If you have an incorrect string, now after inserting one or more characters, you will get the correct string or expected string.

Let's see an example.

If I have entered a string `aple`,then after inserting `p` we will get `apple`, which is right. If you have entered a string `staemnt` then after inserting `t` and `e` you will get `statement`, which is right.

**Deletion operation**

You may have an incorrect string which can be converted into a correct string after deleting one or more characters of the string.

An example is as follows:

If I have entered `caroot`, then to get the correct string we need to delete one `o`. After that, we will get the correct string `carrot`.

**Substitution operation**

If you get the correct string by substituting one or more characters, then it is called a **substitution operation**.

Suppose you have a string `implemantation`. To make it correct, you need to substitute the first `a` to `e` and you will get the correct string `implementation`.

**Algorithms for spelling corrections**

We are using the minimum edit distance algorithm to understand spelling corrections.

**Minimum edit distance**

This algorithm is for converting one string `X` into another string `Y` and we need to find out what the minimum edit cost is to convert string `X` to string `Y`. So, here you can either do insertion, deletion, or substitution operations to convert string `X` to `Y` with the minimum possible sequences of the character edits.

Suppose you have a string `X` with a length of `n`,and string `Y` with a length of `m`.

Follow the steps of the algorithm:

```py
Input: Two String, X and Y  
Output: cheapest possible sequences of the character edits for converting string from X to Y. D( i , j ) = minimum distance cost for converting X string to Y  

```

Let's look at the following steps:

1.  Set `n` to a length of P.
    Set `m` to a length of Q.
2.  If `n = 0`, return `m` and exit.
    If `m = 0`, return `n` and exit.
3.  Create a matrix containing 0..*m* rows and 0..*n* columns.

4.  Initialize the first row to 0..*n*.
    Initialize first column to 0..*m*.
5.  Iterate each character of P (`i` from 1 to *n*).
    Iterate each character of Q (`j` from 1 to *m*).
6.  If P[i] equals Q[j], the cost is 0.
    If Q[i] doesn't equal Q[j], the cost is 1.

 Set the value at cell `v[i,j]` of the matrix equal to the minimum of all three of the following  points:

7.  The cell immediately previous plus 1: `v[i-1,j] + 1`
8.  The cell immediately to the left plus 1: `v[i,j-1] + 1`
9.  The cell diagonally previous and to the left plus the cost: `v[i-1,j-1] +1` for minimum edit distance. If you are using the Levenshtein distance then `v[i-1,j-1] +` cost should be considered
10.  After the iteration in *step 7* to *step 9* has been completed, the distance is found in cell `v[n,m]`.

The previous steps are the basic algorithm to develop the logic of spelling corrections but we can use probability distribution of words and take a consideration of that as well. This kind of algorithmic approach is based on dynamic programing.

Let's convert the string `tutour` to `tutor` by understanding that we need to delete `u`. The edit distance is therefore 1\. The table which is developed by using the defined algorithm is shown in *Figure 4.20* for computing the minimum edit distance:

![](img/3044f3f6-aa26-4af7-857e-846d55589d53.png)

Figure 4.20: Computing minimum edit distance

**Implementation**

Now, for the spelling correction, we need to add a dictionary or extract the words from the large documents. So, in the implementation, we have used a big document from where we have extracted words. Apart from that, we have used the probability of occurring words in the document to get an idea about the distribution. You can see more details regarding the implementation part by clicking on this link: [http://norvig.com/spell-correct.html](http://norvig.com/spell-correct.html)

We have implemented the spelling correction for the minimum edit distance 2.

See the implementation of the spelling correction in *Figure 4.21*:

![](img/aa9bb806-49ed-4658-8f9e-8acb601484be.png)

Figure 4.21: Implementation of spelling correction

See the output of the spelling correction in *Figure 4.22*.

We are providing the string `aple`, which is converted to `apple` successfully:

![](img/9d23d7f2-847a-4f5d-a2f6-def409aaface.png)

Figure 4.22: Output of spelling correction

# Summary

In this chapter, we have looked at all kinds of preprocessing techniques which will be useful to you when you are developing an NLP system or an NLP application. We have also touched upon a spelling correction system which you can consider as part of the preprocessing technique because it will be useful for many of the NLP applications that you develop in the future. By the way, you can access the code on GitHub by clicking the following link: [https://github.com/jalajthanaki/NLPython/tree/master/ch4](https://github.com/jalajthanaki/NLPython/tree/master/ch4)

In the next chapter, we will look at the most important part for any NLP system: feature engineering. The performance of an NLP system mainly depends on what kind of data we provide to the NLP system. Feature engineering is an art and skill which you are going to adopt from the next chapter onwards and, trust me, it is the most important ingredient in developing the NLP systems, so read it and definitely implement it to enrich your skills.