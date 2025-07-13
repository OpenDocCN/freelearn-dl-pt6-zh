# Pre-Processing

In this chapter, we will cover the following recipes:

*   Tokenization – learning to use the inbuilt tokenizers of NLTK
*   Stemming – learning to use the inbuilt stemmers of NLTK
*   Lemmatization – learning to use the WordnetLemmatizer of NLTK
*   Stopwords – learning to use the stopwords corpus and seeing the difference it can make
*   Edit distance – writing your own algorithm to find edit distance between two strings
*   Processing two short stories and extracting the common vocabulary between two of them

# Introduction

In the previous chapter, we learned to read, normalize, and organize raw data coming from heterogeneous forms and formats into uniformity. In this chapter, we will go a step forward and prepare the data for consumption in our NLP applications. Preprocessing is the most important step in any kind of data processing task, or else we fall prey to the age old computer science cliché of *garbage in, garbage out*. The aim of this chapter is to introduce some of the critical preprocessing steps such as tokenization, stemming, lemmatization, and so on.

In this chapter, we will be seeing six different recipes. We will build up the chapter by performing each preprocessing task in individual recipes—tokenization, stemming, lemmatization, stopwords treatment, and edit distance—in that order. In the last recipe, we will look at an example of how we can combine some of these preprocessing techniques to find common vocabulary between two free-form texts.

# Tokenization – learning to use the inbuilt tokenizers of NLTK

Understand the meaning of tokenization, why we need it, and how to do it.

# Getting ready

Let's first see what a token is. When you receive a document or a long string that you want to process or work on, the first thing you'd want to do is break it into words and punctuation marks. This is what we call the process of tokenization. We will see what types of tokenizers are available with NLTK and implement them as well.

# How to do it…

1.  Create a file named `tokenizer.py` and add the following import lines to it:

```py
from nltk.tokenize import LineTokenizer, SpaceTokenizer, TweetTokenizer
from nltk import word_tokenize
```

Import the four different types of tokenizers that we are going to explore in this recipe.

2.  We will start with `LineTokernizer`. Add the following two lines:

```py
lTokenizer = LineTokenizer();
print("Line tokenizer output :",lTokenizer.tokenize("My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius. \nFather to a murdered son, husband to a murdered wife. \nAnd I will have my vengeance, in this life or the next."))
```

3.  As the name implies, this tokenizer is supposed to divide the input string into lines (not sentences, mind you!). Let's see the output and what the tokenizer does:

```py
Line tokenizer output : ['My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius. ', 'Father to a murdered son, husband to a murdered wife. ', 'And I will have my vengeance, in this life or the next.']
```

As you can see, it has returned a list of three strings, meaning the given input has been divided in to three lines on the basis of where the newlines are. `LineTokenizer` simply divides the given input string into new lines.

4.  Now we will see `SpaceTokenizer`. As the name implies, it is supposed to divide on split on space characters. Add the following lines:

```py
rawText = "By 11 o'clock on Sunday, the doctor shall open the dispensary."
sTokenizer = SpaceTokenizer()
print("Space Tokenizer output :",sTokenizer.tokenize(rawText))
```

5.  The `sTokenizer` object is an object of `SpaceTokenize`. We have invoked the `tokenize()` method and we shall see the output now:

```py
Space Tokenizer output : ['By', '11', "o'clock", 'on', 'Sunday,', 'the', 'doctor', 'shall', 'open', 'the', 'dispensary.']
```

6.  As expected, the input `rawText` is split on the space character  `""`.  On to the next one! It's the `word_tokenize()` method. Add the following line:

```py
print("Word Tokenizer output :", word_tokenize(rawText))
```

7.  See the difference here. The other two we have seen so far are classes, whereas this is a method of the `nltk` module. This is the method that we will be using most of the time going forward as it does exactly what we've defined to be tokenization. It breaks up words and punctuation marks. Let's see the output:

```py
Word Tokenizer output : ['By', '11', "o'clock", 'on', 'Sunday', ',', 'the', 'doctor', 'shall', 'open', 'the', 'dispensary', '.']
```

8.  As you can see, the difference between `SpaceTokenizer` and `word_tokenize()` is clearly visible.

9.  Now, on to the last one. There's a special `TweetTokernizer` that we can use when dealing with special case strings:

```py
tTokenizer = TweetTokenizer()
print("Tweet Tokenizer output :",tTokenizer.tokenize("This is a cooool #dummysmiley: :-) :-P <3"))
```

10.  Tweets contain special words, special characters, hashtags, and smileys that we want to keep intact. Let's see the output of these two lines:

```py
Tweet Tokenizer output : ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3']
```

As we see, the `Tokenizer` kept the hashtag word intact and didn't break it; the smileys are also kept intact and are not lost. This is one special little class that can be used when the application demands it.

11.  Here's the output of the program in full. We have already seen it in detail, so I will not be going into it again:

```py
Line tokenizer output : ['My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius. ', 'Father to a murdered son, husband to a murdered wife. ', 'And I will have my vengeance, in this life or the next.']
Space Tokenizer output : ['By', '11', "o'clock", 'on', 'Sunday,', 'the', 'doctor', 'shall', 'open', 'the', 'dispensary.']
Word Tokenizer output : ['By', '11', "o'clock", 'on', 'Sunday', ',', 'the', 'doctor', 'shall', 'open', 'the', 'dispensary', '.']
Tweet Tokenizer output : ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3']
```

# How it works…

We saw three tokenizer classes and a method implemented to do the job in the NLTK module. It's not very difficult to understand how to do it, but it is worth knowing why we do it. The smallest unit to process in language processing task is a token. It is very much like a divide-and-conquer strategy, where we try to make sense of the smallest units at a granular level and add them up to understand the semantics of the sentence, paragraph, document, and the corpus (if any) by moving up the level of detail.

# Stemming – learning to use the inbuilt stemmers of NLTK

Let's understand the concept of a stem and the process of stemming. We will learn why we need to do it and how to perform it using inbuilt NLTK stemming classes.

# Getting ready

So what is a stem supposed to be? A stem is the base form of a word without any suffixes. And a stemmer is what removes the suffixes and returns the stem of the word. Let's see what types of stemmers are available with NLTK.

# How to do it…

1.  Create a file named `stemmers.py` and add the following import lines to it:

```py
from nltk import PorterStemmer, LancasterStemmer, word_tokenize
```

Importing the four different types of tokenizers that we are going to explore in this recipe

2.  Before we apply any stems, we need to tokenize the input text. Let's quickly get that done with the following code:

```py
raw = "My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius. Father to a murdered son, husband to a murdered wife. And I will have my vengeance, in this life or the next."
tokens = word_tokenize(raw)
```

The token list contains all the `tokens` generated from the `raw` input string.

3.  First we shall `seePorterStemmer`. Let's add the following three lines:

```py
porter = PorterStemmer()
pStems = [porter.stem(t) for t in tokens]
print(pStems)
```

4.  First, we initialize the stemmer object. Then we apply the stemmer to all `tokens` of the input text, and finally we `print` the output. Let's see the output and we will know more:

```py
['My', 'name', 'is', 'maximu', 'decimu', 'meridiu', ',', 'command', 'of', 'the', 'armi', 'of', 'the', 'north', ',', 'gener', 'of', 'the', 'felix', 'legion', 'and', 'loyal', 'servant', 'to', 'the', 'true', 'emperor', ',', 'marcu', 'aureliu', '.', 'father', 'to', 'a', 'murder', 'son', ',', 'husband', 'to', 'a', 'murder', 'wife', '.', 'and', 'I', 'will', 'have', 'my', 'vengeanc', ',', 'in', 'thi', 'life', 'or', 'the', 'next', '.']
```

As you can see in the output, all the words have been rid of the trailing `'s'`, `'es'`, `'e'`, `'ed'`, `'al'`, and so on.

5.  The next one is `LancasterStemmer`. This one is supposed to be even more error prone as it contains many more suffixes to be removed than `porter`:

```py
lancaster = LancasterStemmer()
lStems = [lancaster.stem(t) for t in tokens]
print(lStems)
```

6.  The same drill! Just that this time we have `LancasterStemmer` instead of `PrterStemmer`. Let's see the output:

```py
['my', 'nam', 'is', 'maxim', 'decim', 'meridi', ',', 'command', 'of', 'the', 'army', 'of', 'the', 'nor', ',', 'gen', 'of', 'the', 'felix', 'leg', 'and', 'loy', 'serv', 'to', 'the', 'tru', 'emp', ',', 'marc', 'aureli', '.', 'fath', 'to', 'a', 'murd', 'son', ',', 'husband', 'to', 'a', 'murd', 'wif', '.', 'and', 'i', 'wil', 'hav', 'my', 'veng', ',', 'in', 'thi', 'lif', 'or', 'the', 'next', '.']
```

We shall discuss the difference in the output section, but we can make out that the suffixes that are dropped are bigger than Porter. `'us'`, `'e'`, `'th'`, `'eral'`, `"ered"`, and many more!

7.  Here's the output of the program in full. We will compare the output of both the stemmers:

```py
['My', 'name', 'is', 'maximu', 'decimu', 'meridiu', ',', 'command', 'of', 'the', 'armi', 'of', 'the', 'north', ',', 'gener', 'of', 'the', 'felix', 'legion', 'and', 'loyal', 'servant', 'to', 'the', 'true', 'emperor', ',', 'marcu', 'aureliu', '.', 'father', 'to', 'a', 'murder', 'son', ',', 'husband', 'to', 'a', 'murder', 'wife', '.', 'and', 'I', 'will', 'have', 'my', 'vengeanc', ',', 'in', 'thi', 'life', 'or', 'the', 'next', '.']
['my', 'nam', 'is', 'maxim', 'decim', 'meridi', ',', 'command', 'of', 'the', 'army', 'of', 'the', 'nor', ',', 'gen', 'of', 'the', 'felix', 'leg', 'and', 'loy', 'serv', 'to', 'the', 'tru', 'emp', ',', 'marc', 'aureli', '.', 'fath', 'to', 'a', 'murd', 'son', ',', 'husband', 'to', 'a', 'murd', 'wif', '.', 'and', 'i', 'wil', 'hav', 'my', 'veng', ',', 'in', 'thi', 'lif', 'or', 'the', 'next', '.']
```

As we compare the output of both the stemmers, we see that `lancaster` is clearly the greedier one when dropping suffixes. It tries to remove as many characters from the end as possible, whereas `porter` is non-greedy and removes as little as possible.

# How it works…

For some language processing tasks, we ignore the form available in the input text and work with the stems instead. For example, when you search on the Internet for *cameras*, the result includes documents containing the word *camera* as well as *cameras*, and vice versa. In hindsight though, both words are the same; the stem is *camera*.

Having said this, we can clearly see that this method is quite error prone, as the spellings are quite meddled with after a stemmer is done reducing the words. At times, it might be okay, but if you really want to understand the semantics, there is a lot of data loss here. For this reason, we shall next see what is called **lemmatization**.

# Lemmatization – learning to use the WordnetLemmatizer of NLTK

Understand what lemma and lemmatization are. Learn how lemmatization differs from Stemming, why we need it, and how to perform it using `nltk` library's `WordnetLemmatizer`.

# Getting ready

A lemma is a lexicon headword or, more simply, the base form of a word. We have already seen what a stem is, but a lemma is a dictionary-matched base form unlike the stem obtained by removing/replacing the suffixes. Since it is a dictionary match, lemmatization is a slower process than Stemming.

# How to do it…

1.  Create a file named `lemmatizer.py` and add the following import lines to it:

```py
from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
```

We will need to tokenize the sentences first, and we shall use the `PorterStemmer` to compare the output.

2.  Before we apply any stems, we need to tokenize the input text. Let's quickly get that done with the following code:

```py
raw = "My name is Maximus Decimus Meridius, commander of the armies of the north, General of the Felix legions and loyal servant to the true emperor, Marcus Aurelius. Father to a murdered son, husband to a murdered wife. And I will have my vengeance, in this life or the next."
tokens = word_tokenize(raw)
```

The token list contains all the tokens generated from the `raw` input string.

3.  First we will `applyPorterStemmer`, which we have already seen in the previous recipe. Let's add the following three lines:

```py
porter = PorterStemmer()
stems = [porter.stem(t) for t in tokens]
print(stems)
```

First, we initialize the stemmer object. Then we apply the stemmer on all `tokens` of the input text, and finally we print the output. We shall check the output at the end of the recipe.

4.  Now we apply the `lemmatizer`. Add the following three lines:

```py
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(t) for t in tokens]
print(lemmas)
```

5.  Run, and the output of these three lines will be like this:

```py
['My', 'name', 'is', 'Maximus', 'Decimus', 'Meridius', ',', 'commander', 'of', 'the', 'army', 'of', 'the', 'north', ',', 'General', 'of', 'the', 'Felix', 'legion', 'and', 'loyal', 'servant', 'to', 'the', 'true', 'emperor', ',', 'Marcus', 'Aurelius', '.', 'Father', 'to', 'a', 'murdered', 'son', ',', 'husband', 'to', 'a', 'murdered', 'wife', '.', 'And', 'I', 'will', 'have', 'my', 'vengeance', ',', 'in', 'this', 'life', 'or', 'the', 'next', '.']
```

As you see, it understands that for nouns it doesn't have to remove the trailing `'s'`. But for non-nouns, for example, legions and armies, it removes suffixes and also replaces them. However, what it’s essentially doing is a dictionary match. We shall discuss the difference in the output section.

6.  Here's the output of the program in full. We will compare the output of both the stemmers:

```py
['My', 'name', 'is', 'maximu', 'decimu', 'meridiu', ',', 'command', 'of', 'the', 'armi', 'of', 'the', 'north', ',', 'gener', 'of', 'the', 'felix', 'legion', 'and', 'loyal', 'servant', 'to', 'the', 'true', 'emperor', ',', 'marcu', 'aureliu', '.', 'father', 'to', 'a', 'murder', 'son', ',', 'husband', 'to', 'a', 'murder', 'wife', '.', 'and', 'I', 'will', 'have', 'my', 'vengeanc', ',', 'in', 'thi', 'life', 'or', 'the', 'next', '.']
['My', 'name', 'is', 'Maximus', 'Decimus', 'Meridius', ',', 'commander', 'of', 'the', 'army', 'of', 'the', 'north', ',', 'General', 'of', 'the', 'Felix', 'legion', 'and', 'loyal', 'servant', 'to', 'the', 'true', 'emperor', ',', 'Marcus', 'Aurelius', '.', 'Father', 'to', 'a', 'murdered', 'son', ',', 'husband', 'to', 'a', 'murdered', 'wife', '.', 'And', 'I', 'will', 'have', 'my', 'vengeance', ',', 'in', 'this', 'life', 'or', 'the', 'next', '.']
```

As we compare the output of the stemmer and the `lemmatizer`, we see that the stemmer makes a lot of mistakes and the `lemmatizer` makes very few mistakes. However, it doesn't do anything with the word `'murdered'`, and that is an error. Yet, as an end product, `lemmatizer` does a far better job of getting us the base form than the stemmer.

# How it works…

`WordNetLemmatizer` removes affixes only if it can find the resulting word in the dictionary. This makes the process of lemmatization slower than Stemming. Also, it understands and treats capitalized words as special words; it doesn’t do any processing for them and returns them as is. To work around this, you may want to convert your input string to lowercase and then run lemmatization on it.

All said and done, lemmatization is still not perfect and will make mistakes. Check the input string and the result of this recipe; it couldn't convert `'murdered'` to `'murder'`. Similarly, it will handle the word `'women'` correctly but can't handle `'men'`.

# Stopwords – learning to use the stopwords corpus and seeing the difference it can make

We will be using the Gutenberg corpus as an example in this recipe. The Gutenberg corpus is part of the NLTK data module. It contains a selection of 18 texts from some 25,000 electronic books from the project Gutenberg text archives. It is  `PlainTextCorpus`, meaning there are no categories involved with this corpus. It is best suited if you want to play around with the words/tokens without worrying about the affinity of the text to any particular topic. One of the objectives of this little recipe is also to introduce one of the most important preprocessing steps in text analytics—stopwords treatment.

In accordance with the objectives, we will use this corpus to elaborate the usage of Frequency Distribution of the NLTK module in Python within the context of stopwords. To give a small synopsis, a stopword is a word that, though it has significant syntactic value in sentence formation, carries very negligible or minimal semantic value. When you are not working with the syntax but with a bag-of-words kind of approach (for example, TF/IDF), it makes sense to get rid of stopwords except the ones that you are specifically interested in.

# Getting ready

The `nltk.corpus.stopwords` is also a corpus as part of the NLTK Data module that we will use in this recipe, along with `nltk.corpus.gutenberg`.

# How to do it...

1.  Create a new file named `Gutenberg.py` and add the following three lines of code to it:

```py
import nltk
from nltk.corpus import gutenberg
print(gutenberg.fileids())
```

2.  Here we are importing the required libraries and the Gutenberg corpus in the first two lines. The second line is used to check if the corpus was loaded successfully. Run the file on the Python interpreter and you should get an output that looks similar to:

```py
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']
```

As you can see, the names of all 18 Gutenberg texts are printed on the console.

3.  Add the following two lines of code, where we are doing a little preprocessing step on the list of all words from the corpus:

```py
gb_words = gutenberg.words('bible-kjv.txt')
words_filtered = [e for e in gb_words if len(e) >= 3]
```

The first line simply copies the list of all words in the corpus from the sample bible—`kjv.txt` in the `gb_words` variable. The second, and interesting, step is where we are iterating over the entire list of words from Gutenberg, discarding all the words/tokens whose length is two characters or less.

4.  Now we will access `nltk.corpus.stopwords` and do `stopwords` treatment on the filtered words list from the previous list. Add the following lines of code for the same:

```py
stopwords = nltk.corpus.stopwords.words('english')
words = [w for w in words_filtered if w.lower() not in stopwords]
```

The first line simply loads words from the stopwords corpus into the `stopwords` variable for the `english` language. The second line is where we are filtering out all `stopwords` from the filtered word list we had developed in the previous example.

5.  Now we will simply apply `nltk.FreqDist` to the list of preprocessed `words` and the plain list of `words`. Add these lines to do the same:

```py
fdistPlain = nltk.FreqDist(words)
fdist = nltk.FreqDist(gb_words)
```

Create the `FreqDist` object by passing as argument the words list that we formulated in steps 2 and 3.

6.  Now we want to see some of the characteristics of the frequency distribution that we just made. Add the following four lines in the code and we will see what each does:

```py
print('Following are the most common 10 words in the bag')
print(fdistPlain.most_common(10))
print('Following are the most common 10 words in the bag minus the stopwords')
print(fdist.most_common(10))
```

The `most_common(10)` function will return the `10` most common words in the word bag being processed by frequency distribution. What it outputs is what we will discuss and elaborate now.

7.  After you run this program, you should get something similar to the following:

```py
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']

Following are the most common 10 words in the bag

[(',', 70509), ('the', 62103), (':', 43766), ('and', 38847), ('of', 34480), ('.', 26160), ('to', 13396), ('And', 12846), ('that', 12576), ('in', 12331)]

Following are the most common 10 words in the bag minus the stopwords

[('shall', 9838), ('unto', 8997), ('lord', 7964), ('thou', 5474), ('thy', 4600), ('god', 4472), ('said', 3999), ('thee', 3827), ('upon', 2748), ('man', 2735)]
```

# How it works...

If you look carefully at the output, the most common 10 words in the unprocessed or plain list of words won't make much sense. Whereas from the preprocessed bag of words, the most common 10 words such as `god`, `lord`, and `man` give us a quick understanding that we are dealing with a text related to faith or religion.

The foremost objective of this recipe is to introduce you to the concept of stopwords treatment for text preprocessing techniques that you would most likely have to do before running any complex analysis on your data. The NLTK stopwords corpus contains stop-words for 11 languages. When you are trying to analyze the importance of keywords in any text analytics application, treating the stopwords properly will take you a long way. Frequency distribution will help you get the importance of words. Statistically speaking, this distribution would ideally look like a bell curve if you plot it on a two-dimensional plane of frequency and importance of words.

# Edit distance – writing your own algorithm to find edit distance between two strings

Edit distance, also called as **Levenshtein distance** is a metric used to measure the similarity between two distances. Essentially, it’s a count of how many edit operations, deletions, insertions, or substitutions will transform a given String `A` to String `B`. We shall write our own algorithm to calculate the edit distance and then compare it against `nltk.metrics.distance.edit_distance()` for a sanity check.

# Getting ready

You may want to look up a little more on the Levenshtein distance part for mathematical equations. We will look at the algorithm implementation in python and why we do it, but it may not be feasible to cover the complete mathematics behind it. Here’s a link on Wikipedia: [https://en.wikipedia.org/wiki/Levenshtein_distance](https://en.wikipedia.org/wiki/Levenshtein_distance).

# How to do it…

1.  Create a file named `edit_distance_calculator.py` and add the following import lines to it:

```py
from nltk.metrics.distance import edit_distance
```

We just imported the inbuilt `nltk` library's `edit_distance` function from the `nltk.metrics.distance` module.

2.  Let's define our method to accept two strings and calculate the edit distance between the two. `str1` and `str2` are two strings that the function accepts, and we will return an integer distance value:

```py
def my_edit_distance(str1, str2):
```

3.  The next step is to get the length of the two input strings. We will be using the length to create an *m x n* table where `m` and `n` are the lengths of the two strings `s1` and `s2` respectively:

```py
m=len(str1)+1
n=len(str2)+1
```

4.  Now we will create `table` and initialize the first row and first column:

```py
    table = {}
    for i in range(m): table[i,0]=i
    for j in range(n): table[0,j]=j
```

5.  This will initialize the two-dimensional array and the contents will look like the following table in memory:

![](img/dc28e376-df91-45f1-b9a5-b400657cd191.png)

Please note that this is inside a function and I'm using the example strings we are going to pass to the function to elaborate the algorithm.

6.  Now comes the tricky part. We are going to fill up the matrix using the formula:

```py
for i in range(1, m):
  for j in range(1, n):
    cost = 0 if str1[i-1] == str2[j-1] else 1
    table[i,j] = min(table[i, j-1]+1, table[i-1, j]+1, table[i-1, j-1]+cost)
```

The `cost` is calculated on whether the characters in contention are the same or they edition, specifically deletion or insertion. The formula in the next line for is calculating the value of the cell in the matrix, the first two take care of substitution and the third one is for substitution. We also add the cost of the previous step to it and take the minimum of the three.

7.  At the end, we return the value of the last cell, that is, `table[m,n]`, as the final edit distance value:

```py
return table[i,j]
```

8.  Now we will call our function and the `nltk` library's  `edit_distance()` function on two strings and check the output:

```py
print("Our Algorithm :",my_edit_distance("hand", "and"))
print("NLTK Algorithm :",edit_distance("hand", "and"))
```

9.  Our words are `hand` and `and`. Only a single delete operation on the first string or a single insertion operation on the second string will give us a match. Hence, the expected Levenshtein score is `1`.
10.  Here's the output of the program:

```py
Our Algorithm : 1
NLTK Algorithm : 1
```

As expected, the NLTK `edit_distance()` returns `1` and so does our algorithm. Fair to say that our algorithm is doing as expected, but I would urge you guys to test it further by running it through with some more examples.

# How it works…

I've already given you a brief on the algorithm; now let’s see how the matrix *table* gets populated with the algorithm. See the attached table here:

![](img/719ea28d-7009-4ffc-a111-b8f99a86de11.png)

You've already seen how we initialized the matrix. Then we filled up the matrix using the formula in algorithm. The yellow trail you see is the significant numbers. After the first iteration, you can see that the distance is moving in the direction of 1 consistently and the final value that we return is denoted by the green background cell.

Now, the applications of the edit distance algorithm are multifold. First and foremost, it is used in spell checkers and auto-suggestions in text editors, search engines, and many such text-based applications. Since the cost of comparisons is equivalent to the product of the length of the strings to be compared, it is sometimes impractical to apply it to compare large texts.

# Processing two short stories and extracting the common vocabulary between two of them

This recipe is supposed to give you an idea of how to handle a typical text analytics problem when you come across it. We will be using multiple preprocessing techniques in the process of getting to our outcome. The recipe will end with an important preprocessing task and not a real application of text analysis. We will be using a couple of short stories from [http://www.english-for-students.com/](http://www.english-for-students.com/).

# Getting ready

We will be removing all special characters, splitting words, doing case folds, and some set and list operations in this recipe. We won’t be using any special libraries, just Python programming tricks.

# How to do it…

1.  Create a file named `lemmatizer.py` and create a couple of long strings with short stories or any news articles:

```py
story1 = """In a far away kingdom, there was a river. This river was home to many golden swans. The swans spent most of their time on the banks of the river. Every six months, the swans would leave a golden feather as a fee for using the lake. The soldiers of the kingdom would collect the feathers and deposit them in the royal treasury.
One day, a homeless bird saw the river. "The water in this river seems so cool and soothing. I will make my home here," thought the bird.
As soon as the bird settled down near the river, the golden swans noticed her. They came shouting. "This river belongs to us. We pay a golden feather to the King to use this river. You can not live here."
"I am homeless, brothers. I too will pay the rent. Please give me shelter," the bird pleaded. "How will you pay the rent? You do not have golden feathers," said the swans laughing. They further added, "Stop dreaming and leave once." The humble bird pleaded many times. But the arrogant swans drove the bird away.
"I will teach them a lesson!" decided the humiliated bird.
She went to the King and said, "O King! The swans in your river are impolite and unkind. I begged for shelter but they said that they had purchased the river with golden feathers."
The King was angry with the arrogant swans for having insulted the homeless bird. He ordered his soldiers to bring the arrogant swans to his court. In no time, all the golden swans were brought to the King’s court.
"Do you think the royal treasury depends upon your golden feathers? You can not decide who lives by the river. Leave the river at once or you all will be beheaded!" shouted the King.
The swans shivered with fear on hearing the King. They flew away never to return. The bird built her home near the river and lived there happily forever. The bird gave shelter to all other birds in the river. """
story2 = """Long time ago, there lived a King. He was lazy and liked all the comforts of life. He never carried out his duties as a King. “Our King does not take care of our needs. He also ignores the affairs of his kingdom." The people complained.
One day, the King went into the forest to hunt. After having wandered for quite sometime, he became thirsty. To his relief, he spotted a lake. As he was drinking water, he suddenly saw a golden swan come out of the lake and perch on a stone. “Oh! A golden swan. I must capture it," thought the King.
But as soon as he held his bow up, the swan disappeared. And the King heard a voice, “I am the Golden Swan. If you want to capture me, you must come to heaven."
Surprised, the King said, “Please show me the way to heaven." “Do good deeds, serve your people and the messenger from heaven would come to fetch you to heaven," replied the voice.
The selfish King, eager to capture the Swan, tried doing some good deeds in his Kingdom. “Now, I suppose a messenger will come to take me to heaven," he thought. But, no messenger came.
The King then disguised himself and went out into the street. There he tried helping an old man. But the old man became angry and said, “You need not try to help. I am in this miserable state because of out selfish King. He has done nothing for his people."
Suddenly, the King heard the golden swan’s voice, “Do good deeds and you will come to heaven." It dawned on the King that by doing selfish acts, he will not go to heaven.
He realized that his people needed him and carrying out his duties was the only way to heaven. After that day he became a responsible King.
"""
```

There we have two short stories from the aforementioned website!

2.  First, we will remove some of the special characters from the texts. We are removing all newlines (`'\n'`), commas, full stops, exclamations, question marks, and so on. At the end, we convert the entire string to lowercase with the `casefold()` function:

```py
story1 = story1.replace(",", "").replace("\n", "").replace('.', '').replace('"', '').replace("!","").replace("?","").casefold()
story2 = story2.replace(",", "").replace("\n", "").replace('.', '').replace('"', '').replace("!","").replace("?","").casefold()
```

3.  Next, we will split the texts into words:

```py
story1_words = story1.split(" ")
print("First Story words :",story1_words)
story2_words = story2.split(" ")
print("Second Story words :",story2_words)
```

4.  Using `split` on the `""` character, we split and get the list of words from `story1` and `story2.` Let's see the output after this step:

```py
First Story words : ['in', 'a', 'far', 'away', 'kingdom', 'there', 'was', 'a', 'river', 'this', 'river', 'was', 'home', 'to', 'many', 'golden', 'swans', 'the', 'swans', 'spent', 'most', 'of', 'their', 'time', 'on', 'the', 'banks', 'of', 'the', 'river', 'every', 'six', 'months', 'the', 'swans', 'would', 'leave', 'a', 'golden', 'feather', 'as', 'a', 'fee', 'for', 'using', 'the', 'lake', 'the', 'soldiers', 'of', 'the', 'kingdom', 'would', 'collect', 'the', 'feathers', 'and', 'deposit', 'them', 'in', 'the', 'royal', 'treasury', 'one', 'day', 'a', 'homeless', 'bird', 'saw', 'the', 'river', 'the', 'water', 'in', 'this', 'river', 'seems', 'so', 'cool', 'and', 'soothing', 'i', 'will', 'make', 'my', 'home', 'here', 'thought', 'the', 'bird', 'as', 'soon', 'as', 'the', 'bird', 'settled', 'down', 'near', 'the', 'river', 'the', 'golden', 'swans', 'noticed', 'her', 'they', 'came', 'shouting', 'this', 'river', 'belongs', 'to', 'us', 'we', 'pay', 'a', 'golden', 'feather', 'to', 'the', 'king', 'to', 'use', 'this', 'river', 'you', 'can', 'not', 'live', 'here', 'i', 'am', 'homeless', 'brothers', 'i', 'too', 'will', 'pay', 'the', 'rent', 'please', 'give', 'me', 'shelter', 'the', 'bird', 'pleaded', 'how', 'will', 'you', 'pay', 'the', 'rent', 'you', 'do', 'not', 'have', 'golden', 'feathers', 'said', 'the', 'swans', 'laughing', 'they', 'further', 'added', 'stop', 'dreaming', 'and', 'leave', 'once', 'the', 'humble', 'bird', 'pleaded', 'many', 'times', 'but', 'the', 'arrogant', 'swans', 'drove', 'the', 'bird', 'away', 'i', 'will', 'teach', 'them', 'a', 'lesson', 'decided', 'the', 'humiliated', 'bird', 'she', 'went', 'to', 'the', 'king', 'and', 'said', 'o', 'king', 'the', 'swans', 'in', 'your', 'river', 'are', 'impolite', 'and', 'unkind', 'i', 'begged', 'for', 'shelter', 'but', 'they', 'said', 'that', 'they', 'had', 'purchased', 'the', 'river', 'with', 'golden', 'feathers', 'the', 'king', 'was', 'angry', 'with', 'the', 'arrogant', 'swans', 'for', 'having', 'insulted', 'the', 'homeless', 'bird', 'he', 'ordered', 'his', 'soldiers', 'to', 'bring', 'the', 'arrogant', 'swans', 'to', 'his', 'court', 'in', 'no', 'time', 'all', 'the', 'golden', 'swans', 'were', 'brought', 'to', 'the', 'king’s', 'court', 'do', 'you', 'think', 'the', 'royal', 'treasury', 'depends', 'upon', 'your', 'golden', 'feathers', 'you', 'can', 'not', 'decide', 'who', 'lives', 'by', 'the', 'river', 'leave', 'the', 'river', 'at', 'once', 'or', 'you', 'all', 'will', 'be', 'beheaded', 'shouted', 'the', 'king', 'the', 'swans', 'shivered', 'with', 'fear', 'on', 'hearing', 'the', 'king', 'they', 'flew', 'away', 'never', 'to', 'return', 'the', 'bird', 'built', 'her', 'home', 'near', 'the', 'river', 'and', 'lived', 'there', 'happily', 'forever', 'the', 'bird', 'gave', 'shelter', 'to', 'all', 'other', 'birds', 'in', 'the', 'river', ''] Second Story words : ['long', 'time', 'ago', 'there', 'lived', 'a', 'king', 'he', 'was', 'lazy', 'and', 'liked', 'all', 'the', 'comforts', 'of', 'life', 'he', 'never', 'carried', 'out', 'his', 'duties', 'as', 'a', 'king', '“our', 'king', 'does', 'not', 'take', 'care', 'of', 'our', 'needs', 'he', 'also', 'ignores', 'the', 'affairs', 'of', 'his', 'kingdom', 'the', 'people', 'complained', 'one', 'day', 'the', 'king', 'went', 'into', 'the', 'forest', 'to', 'hunt', 'after', 'having', 'wandered', 'for', 'quite', 'sometime', 'he', 'became', 'thirsty', 'to', 'his', 'relief', 'he', 'spotted', 'a', 'lake', 'as', 'he', 'was', 'drinking', 'water', 'he', 'suddenly', 'saw', 'a', 'golden', 'swan', 'come', 'out', 'of', 'the', 'lake', 'and', 'perch', 'on', 'a', 'stone', '“oh', 'a', 'golden', 'swan', 'i', 'must', 'capture', 'it', 'thought', 'the', 'king', 'but', 'as', 'soon', 'as', 'he', 'held', 'his', 'bow', 'up', 'the', 'swan', 'disappeared', 'and', 'the', 'king', 'heard', 'a', 'voice', '“i', 'am', 'the', 'golden', 'swan', 'if', 'you', 'want', 'to', 'capture', 'me', 'you', 'must', 'come', 'to', 'heaven', 'surprised', 'the', 'king', 'said', '“please', 'show', 'me', 'the', 'way', 'to', 'heaven', '“do', 'good', 'deeds', 'serve', 'your', 'people', 'and', 'the', 'messenger', 'from', 'heaven', 'would', 'come', 'to', 'fetch', 'you', 'to', 'heaven', 'replied', 'the', 'voice', 'the', 'selfish', 'king', 'eager', 'to', 'capture', 'the', 'swan', 'tried', 'doing', 'some', 'good', 'deeds', 'in', 'his', 'kingdom', '“now', 'i', 'suppose', 'a', 'messenger', 'will', 'come', 'to', 'take', 'me', 'to', 'heaven', 'he', 'thought', 'but', 'no', 'messenger', 'came', 'the', 'king', 'then', 'disguised', 'himself', 'and', 'went', 'out', 'into', 'the', 'street', 'there', 'he', 'tried', 'helping', 'an', 'old', 'man', 'but', 'the', 'old', 'man', 'became', 'angry', 'and', 'said', '“you', 'need', 'not', 'try', 'to', 'help', 'i', 'am', 'in', 'this', 'miserable', 'state', 'because', 'of', 'out', 'selfish', 'king', 'he', 'has', 'done', 'nothing', 'for', 'his', 'people', 'suddenly', 'the', 'king', 'heard', 'the', 'golden', 'swan’s', 'voice', '“do', 'good', 'deeds', 'and', 'you', 'will', 'come', 'to', 'heaven', 'it', 'dawned', 'on', 'the', 'king', 'that', 'by', 'doing', 'selfish', 'acts', 'he', 'will', 'not', 'go', 'to', 'heaven', 'he', 'realized', 'that', 'his', 'people', 'needed', 'him', 'and', 'carrying', 'out', 'his', 'duties', 'was', 'the', 'only', 'way', 'to', 'heaven', 'after', 'that', 'day', 'he', 'became', 'a', 'responsible', 'king', '']
```

As you can see, all the special characters are gone and a list of words is created.

5.  Now let's create a vocabulary out of this list of words. A vocabulary is a set of words. No repeats!

```py
story1_vocab = set(story1_words)
print("First Story vocabulary :",story1_vocab)
story2_vocab = set(story2_words)
print("Second Story vocabulary",story2_vocab)
```

6.  Calling the Python internal `set()` function on the list will deduplicate the list and convert it into a set:

```py
First Story vocabulary : {'', 'king’s', 'am', 'further', 'having', 'river', 'he', 'all', 'feathers', 'banks', 'at', 'shivered', 'other', 'are', 'came', 'here', 'that', 'soon', 'lives', 'unkind', 'by', 'on', 'too', 'kingdom', 'never', 'o', 'make', 'every', 'will', 'said', 'birds', 'teach', 'away', 'hearing', 'humble', 'but', 'deposit', 'them', 'would', 'leave', 'return', 'added', 'were', 'fear', 'bird', 'lake', 'my', 'settled', 'or', 'pleaded', 'in', 'so', 'use', 'was', 'me', 'us', 'laughing', 'bring', 'rent', 'have', 'how', 'lived', 'of', 'seems', 'gave', 'day', 'no', 'months', 'down', 'this', 'the', 'her', 'decided', 'angry', 'built', 'cool', 'think', 'golden', 'spent', 'time', 'noticed', 'lesson', 'many', 'near', 'once', 'collect', 'who', 'your', 'flew', 'fee', 'six', 'most', 'had', 'to', 'please', 'purchased', 'happily', 'depends', 'belongs', 'give', 'begged', 'there', 'she', 'i', 'times', 'dreaming', 'as', 'court', 'their', 'you', 'shouted', 'shelter', 'forever', 'royal', 'insulted', 'they', 'with', 'live', 'far', 'water', 'king', 'shouting', 'a', 'brothers', 'drove', 'arrogant', 'saw', 'soldiers', 'stop', 'home', 'upon', 'can', 'decide', 'beheaded', 'do', 'for', 'homeless', 'ordered', 'be', 'using', 'not', 'feather', 'soothing', 'swans', 'humiliated', 'treasury', 'thought', 'one', 'and', 'we', 'impolite', 'brought', 'went', 'pay', 'his'}
Second Story vocabulary {'', 'needed', 'having', 'am', 'he', 'all', 'way', 'spotted', 'voice', 'realized', 'also', 'came', 'that', '“our', 'soon', '“oh', 'by', 'on', 'has', 'complained', 'never', 'ago', 'kingdom', '“do', 'capture', 'said', 'into', 'long', 'will', 'liked', 'disappeared', 'but', 'would', 'must', 'stone', 'lake', 'from', 'messenger', 'eager', 'deeds', 'fetch', 'carrying', 'in', 'because', 'perch', 'responsible', 'was', 'me', 'disguised', 'take', 'comforts', 'lived', 'of', 'tried', 'day', 'no', 'street', 'good', 'bow', 'the', 'need', 'this', 'helping', 'angry', 'out', 'thirsty', 'relief', 'wandered', 'old', 'golden', 'acts', 'time', 'an', 'needs', 'suddenly', 'state', 'serve', 'affairs', 'ignores', 'does', 'people', 'want', 'your', 'dawned', 'man', 'to', 'miserable', 'became', 'swan', 'there', 'hunt', 'show', 'i', 'heaven', 'as', 'selfish', 'after', 'suppose', 'you', 'only', 'done', 'drinking', 'then', 'care', 'it', 'him', 'come', 'swan’s', 'if', 'water', 'himself', 'nothing', '“please', 'carried', 'king', 'help', 'heard', 'up', 'try', 'a', 'held', 'saw', 'life', 'surprised', 'go', '“i', 'for', 'doing', 'our', 'some', '“now', 'sometime', 'forest', 'lazy', 'not', '“you', 'replied', 'quite', 'duties', 'thought', 'one', 'and', 'went', 'his'}
```

Here are the deduplicated sets, the vocabularies of both the stories.

7.  Now, the final step. Produce the common vocabulary between these two stories:

```py
common_vocab = story1_vocab & story2_vocab
print("Common Vocabulary :",common_vocab)
```

8.  Python allows the set operation `&` (AND), which we are using to find the set of common entries between these two vocabulary sets. Let's see the output of the final step:

```py
Common Vocabulary : {'', 'king', 'am', 'having', 'he', 'all', 'your', 'in', 'was', 'me', 'a', 'to', 'came', 'that', 'lived', 'soon', 'saw', 'of', 'by', 'on', 'day', 'no', 'never', 'kingdom', 'there', 'for', 'i', 'said', 'will', 'the', 'this', 'as', 'angry', 'you', 'not', 'but', 'would', 'golden', 'thought', 'time', 'one', 'and', 'lake', 'went', 'water', 'his'}
```

And there it is, the end-goal.

Here is the output:

```py
I won't be dumping the output of the entire program again here. It's huge so let’s save some trees!
```

# How it works…

So here, we saw how we can go from a couple of narratives to the common vocabulary between them. We didn’t use any fancy libraries, nor did we perform any complex operations. Yet we built a base from which we can take this bag-of-words forward and do many things with it.

From here on, we can think of many different applications, such as text similarity, search engine tagging, text summarization, and many more.

