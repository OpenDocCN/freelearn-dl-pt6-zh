# Corpus and WordNet

In this chapter, we will cover the following recipes:

*   Accessing in-built corpora
*   Download an external corpus, load it, and access it
*   Counting all the wh words in three different genres in the Brown corpus
*   Explore frequency distribution operations on one of the web and chat text corpus files
*   Take an ambiguous word and explore all its senses using WordNet
*   Pick two distinct synsets and explore the concepts of hyponyms and hypernyms using WordNet
*   Compute the average polysemy of nouns, verbs, adjectives, and adverbs according to WordNet

# Introduction

To solve any real-world **Natural Language Processing** (**NLP**) problems, you need to work with huge amounts of data. This data is generally available in the form of a corpus out there in the open diaspora and as an add-on of the NLTK package. For example, if you want to create a spell checker, you need a huge corpus of words to match against.

The goal of this chapter is to cover the following:

*   Introducing various useful textual corpora available with NLTK
*   How to access these in-built corpora from Python
*   Working with frequency distributions
*   An introduction to WordNet and its lexical features

We will try to understand these things from a practical standpoint. We will perform some exercises that will fulfill all of these goals through our recipes.

# Accessing in-built corpora

As already explained, we have many corpuses available for use with NLTK. We will assume that you have already downloaded and installed NLTK data on your computer. If not, you can find the same at [http://www.nltk.org/data.html](http://www.nltk.org/data.html). Also, a complete list of corpora that you can use from within NLTK data is available at [http://www.nltk.org/nltk_data/](http://www.nltk.org/nltk_data/).

Now, our first task/recipe involves us learning how to access any one of these corpora. We have decided to do some tests on the Reuters corpus or the same. We will import the corpus into our program and try to access it in different ways.

# How to do it...

1.  Create a new file named `reuters.py` and add the following import line in the file. This will specifically allow access to only the `reuters` corpus in our program from the entire NLTK data:

```py
from nltk.corpus import reuters
```

2.  Now we want to check what exactly is available in this corpus. The simplest way to do this is to call the `fileids()` function on the corpus object. Add the following line in your program:

```py
files = reuters.fileids()
print(files)
```

3.  Now run the program and you shall get an output similar to this:

```py
['test/14826', 'test/14828', 'test/14829', 'test/14832', 'test/14833', 'test/14839',
```

These are the lists of files and the relative paths of each of them in the `reuters` corpus.

4.  Now we will access the actual content of any of these files. To do this, we will use the `words()` function on the corpus object as follows, and we will access the `test/16097` file:

```py
words16097 = reuters.words(['test/16097'])
print(words16097)
```

5.  Run the program again and an extra new line of output will appear:

```py
['UGANDA', 'PULLS', 'OUT', 'OF', 'COFFEE', 'MARKET', ...]
```

As you can see, the list of words in the `test/16097` file is shown. This is curtailed though the entire list of words is loaded in the memory object.

6.  Now we want to access a specific number of words (`20`) from the same file, `test/16097`. Yes! We can specify how many words we want to access and store them in a list for use. Append the following two lines in the code:

```py
words20 = reuters.words(['test/16097'])[:20]
print(words20)
```

Run this code and another extra line of output will be appended, which will look like this:

```py
['UGANDA', 'PULLS', 'OUT', 'OF', 'COFFEE', 'MARKET', '-', 'TRADE', 'SOURCES', 'Uganda', "'", 's', 'Coffee', 'Marketing', 'Board', '(', 'CMB', ')', 'has', 'stopped']
```

7.  Moving forward, the `reuters` corpus is not just a list of files but is also hierarchically categorized into 90 topics. Each topic has many files associated with it. What this means is that, when you access any one of the topics, you are actually accessing the set of all files associated with that topic. Let's first output the list of topics by adding the following code:

```py
reutersGenres = reuters.categories()
print(reutersGenres)
```

Run the code and the following line of output will be added to the output console:

```py
['acq', 'alum', 'barley', 'bop', 'carcass', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', ...
```

All 90 categories are displayed.

8.  Finally, we will write four simple lines of code that will not only access two topics but also print out the words in a loosely sentenced fashion as one sentence per line. Add the following code to the Python file:

```py
for w in reuters.words(categories=['bop','cocoa']):
  print(w+' ',end='')
  if(w is '.'):
    print()
```

9.  To explain briefly, we first selected the categories `'bop'` and `'cocoa'` and printed every word from these two categories' files. Every time we encountered a dot (`.`), we inserted a new line. Run the code and something similar to the following will be the output on the console:

```py
['test/14826', 'test/14828', 'test/14829', 'test/14832', 'test/14833', 'test/14839', ...
['UGANDA', 'PULLS', 'OUT', 'OF', 'COFFEE', 'MARKET', ...]
['UGANDA', 'PULLS', 'OUT', 'OF', 'COFFEE', 'MARKET', '-', 'TRADE', 'SOURCES', 'Uganda', "'", 's', 'Coffee', 'Marketing', 'Board', '(', 'CMB', ')', 'has', 'stopped']
['acq', 'alum', 'barley', 'bop', 'carcass', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', ...
SOUTH KOREA MOVES TO SLOW GROWTH OF TRADE SURPLUS South Korea ' s trade surplus is growing too fast and the government has started taking steps to slow it down , Deputy Prime Minister Kim Mahn-je said .
He said at a press conference that the government planned to increase investment , speed up the opening of the local market to foreign imports, and gradually adjust its currency to hold the surplus " at a proper level ." But he said the government would not allow the won to appreciate too much in a short period of time .
South Korea has been under pressure from Washington to revalue the won .
The U .
S .
Wants South Korea to cut its trade surplus with the U .
S ., Which rose to 7 .
4 billion dlrs in 1986 from 4 .
3 billion dlrs in 1985 .
.
.
.
```

# Download an external corpus, load it, and access it

Now that we have learned how to load and access an inbuilt corpus, we will learn how to download and also how to load and access any external corpus. Many of these inbuilt corpora are very good use cases for training purposes, but for solving any real-world problem, you will normally need an external dataset. For this recipe's purpose, we will be using the **Cornell CS Movie** review corpus, which is already labelled for positive and negative reviews and used widely for training sentiment analysis modules.

# Getting ready

First and foremost, you will need to download the dataset from the Internet. Here's the link: [http://www.cs.cornell.edu/people/pabo/movie-review-data/mix20_rand700_tokens_cleaned.zip)](http://www.cs.cornell.edu/people/pabo/movie-review-data/mix20_rand700_tokens_cleaned.zip)). Download the dataset, unzip it, and store the resultant `Reviews` directory at a secure location on your computer.

# How to do it...

1.  Create a new file named `external_corpus.py` and add the following import line to it:

```py
from nltk.corpus import CategorizedPlaintextCorpusReader
```

Since the corpus that we have downloaded is already categorized, we will use `CategorizedPlaintextCorpusReader` to read and load the given corpus. This way, we can be sure that the categories of the corpus are captured, in this case, positive and negative.

2.  Now we will read the corpus. We need to know the absolute path of the `Reviews` folder that we unzipped from the downloaded file from Cornell. Add the following four lines of code:

```py
reader = CategorizedPlaintextCorpusReader(r'/Volumes/Data/NLP-CookBook/Reviews/txt_sentoken', r'.*\.txt', cat_pattern=r'(\w+)/*')
print(reader.categories())
print(reader.fileids())
```

The first line is where you are reading the corpus by calling the `CategorizedPlaintextCorpusReader` constructor. The three arguments from left to right are Absolute Path to the `txt_sentoken` folder on your computer, all sample document names from the `txt_sentoken` folder, and the categories in the given corpus (in our case, `'pos'` and `'neg'`). If you look closely, you'll see that all the three arguments are regular expression patterns. The next two lines will validate whether the corpus is loaded correctly or not, printing the associated categories and filenames of the corpus. Run the program and you should see something similar to the following:

```py
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',....]
[['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of',...]]
```

3.  Now that we've made sure that the corpus is loaded correctly, let's get on with accessing any one of the sample documents from both the categories. For that, let's first create a list, each containing samples of both the categories, `'pos'` and `'neg'`, respectively. Add the following two lines of code:

```py
posFiles = reader.fileids(categories='pos')
negFiles = reader.fileids(categories='neg')
```

The `reader.fileids()` method takes the argument category name. As you can see, what we are trying to do in the preceding two lines of code is straightforward and intuitive.

4.  Now let's select a file randomly from each of the lists of `posFiles` and `negFiles`. To do so, we will need the `randint()` function from the `random` library of Python. Add the following lines of code and we shall elaborate what exactly we did immediately after:

```py
from random import randint
fileP = posFiles[randint(0,len(posFiles)-1)]
fileN = negFiles[randint(0, len(posFiles) - 1)]
print(fileP)
print(fileN)
```

The first line imports the `randint()` function from the `random` library. The next two files select a random file, each from the set of positive and negative category reviews. The last two lines just print the filenames.

5.  Now that we have selected the two files, let's access them and print them on the console sentence by sentence. We will use the same methodology that we used in the first recipe to print a line-by-line output. Append the following lines of code:

```py
for w in reader.words(fileP):
  print(w + ' ', end='')
  if (w is '.'):
    print()
for w in reader.words(fileN):
  print(w + ' ', end='')
  if (w is '.'):
    print()
```

These `for` loops read every file one by one and will print on the console line by line. The output of the complete recipe should look similar to this:

```py
['neg', 'pos']
['neg/cv000_29416.txt', 'neg/cv001_19502.txt', 'neg/cv002_17424.txt', ...]
pos/cv182_7281.txt
neg/cv712_24217.txt
the saint was actually a little better than i expected it to be , in some ways .
in this theatrical remake of the television series the saint...
```

# How it works...

The quintessential ingredient of this recipe is the `CategorizedPlaintextCorpusReader` class of NLTK. Since we already know that the corpus we have downloaded is categorized, we only need provide appropriate arguments when creating the `reader` object. The implementation of the `CategorizedPlaintextCorpusReader` class internally takes care of loading the samples in appropriate buckets (`'pos'` and `'neg'` in this case).

# Counting all the wh words in three different genres in the Brown corpus

The Brown corpus is part of the NLTK data package. It's one of the oldest text corpuses assembled at Brown University. It contains a collection of 500 texts broadly categorized in to 15 different genres/categories such as news, humor, religion, and so on. This corpus is a good use case to showcase the categorized plaintext corpus, which already has topics/concepts assigned to each of the texts (sometimes overlapping); hence, any analysis you do on it can adhere to the attached topic.

# Getting ready

The objective of this recipe is to get you to perform a simple counting task on any given corpus. We will be using `nltk` library's `FreqDist` object for this purpose here, but more elaboration on the power of `FreqDist` will follow in the next recipe. Here, we will just concentrate on the application problem.

# How to do it...

1.  Create a new file named `BrownWH.py` and add the following `import` statements to begin:

```py
import nltk
from nltk.corpus import brown
```

We have imported the `nltk` library and the Brown corpus.

2.  Next up, we will check all the genres in the corpus and will pick any three categories from them to proceed with our task:

```py
print(brown.categories())
```

The `brown.categories()` function call will return the list of all genres in the Brown corpus. When you run this line, you will see the following output:

```py
['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']
```

3.  Now let's pick three `genres`--`fiction`, `humor` and `romance`--from this list as well as the `whwords` that we want to count out from the text of these three `genres`:

```py
genres = ['fiction', 'humor', 'romance']
whwords = ['what', 'which', 'how', 'why', 'when', 'where', 'who']
```

We have created a list containing the three picked `genres` and another list containing the seven `whwords`.

Your list can be longer or shorter depending on what do you consider as `whwords`.

4.  Since we have the `genres` and the words we want to count in lists, we will be extensively using the `for` loop to iterate over them and optimize the number of lines of code. So first, we write a `for` iterator on the `genres` list:

```py
for i in range(0,len(genres)):genre = genres[i]
print()
print("Analysing '"+ genre + "' wh words")
genre_text = brown.words(categories = genre)
```

These four lines of code will only start iterating on the list `genres` and load the entire text of each genre in the `genre_text` variable as a continuous list words.

5.  Next up is a complex little statement where we will use the `nltk` library's `FreqDist` object. For now, let's understand the syntax and the broad-level output we will get from it:

```py
fdist = nltk.FreqDist(genre_text)
```

`FreqDist()` accepts a list of words and returns an object that contains the map word and its respective frequency in the input word list. Here, the `fdist` object will contain the frequency of each of the unique words in the `genre_text` word list.

6.  I'm sure you've already guessed what our next step is going to be. We will simply access the `fdist` object returned by `FreqDist()` and get the count of each of the `wh` words. Let's do it:

```py
for wh in whwords:
print(wh + ':', fdist[wh], end=' ')
```

We are iterating over the `whwords` word list, accessing the `fdist` object with each of the `wh` words as index, getting back the frequency/count of all of them, and printing them out.

After running the complete program, you will get this output:

```py
['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']

Analysing 'fiction' wh words

what: 128 which: 123 how: 54 why: 18 when: 133 where: 76 who: 103

Analysing 'humor' wh words

what: 36 which: 62 how: 18 why: 9 when: 52 where: 15 who: 48

Analysing 'romance' wh words

what: 121 which: 104 how: 60 why: 34 when: 126 where: 54 who: 89
```

# How it works...

On analyzing the output, you can clearly see that we have the word count of all seven `wh` words for the three picked `genres` on our console. By counting the population of `wh` words, you can, to a degree, gauge whether the given text is high on relative clauses or question sentences. Similarly, you may have a populated ontology list of important words that you want to get a word count of to understand the relevance of the given text to your ontology. Counting word populations and analyzing distributions of counts is one of the oldest, simplest, and most popular tricks of the trade to start any kind of textual analysis.

# Explore frequency distribution operations on one of the web and chat text corpus files

Web and chat text corpus is non-formal literature that, as the name implies, contains content from Firefox discussion forums, scripts of movies, wine reviews, personal advertisements, and overheard conversations. Our objective here in this recipe is to understand the use of frequency distribution and its features/functions.

# Getting ready

In keeping with the objective of this recipe, we will run the frequency distribution on the personal advertising file inside `nltk.corpus.webtext`. Following that, we will explore the various functionalities of the `nltk.FreqDist` object such as the count of distinct words, 10 most common words, maximum-frequency words, frequency distribution plot, and tabulation.

# How to do it...

1.  Create a new file named `webtext.py` and add the following three lines to it:

```py
import nltk
from nltk.corpus import webtext
print(webtext.fileids())
```

We just imported the required libraries and the `webtext` corpus; along with that, we also printed the constituent file's names. Run the program and you shall see the following output:

```py
['firefox.txt', 'grail.txt', 'overheard.txt', 'pirates.txt', 'singles.txt', 'wine.txt']
```

2.  Now we will select the file that contains personal advertisement data and and run frequency distribution on it. Add the following three lines for it:

```py
fileid = 'singles.txt'
wbt_words = webtext.words(fileid)
fdist = nltk.FreqDist(wbt_words)
```

`singles.txt` contains our target data; so, we loaded the words from that file in `wbt_words` and ran frequency distribution on it to get the `FreqDist` object `fdist`.

3.  Add the following lines, which will show the most commonly appearing word (with the `fdist.max()` function) and the count of that word (with the `fdist[fdist.max()`] operation):

```py
print('Count of the maximum appearing token "',fdist.max(),'" : ', fdist[fdist.max()])
```

4.  The following line will show us the count of distinct words in the bag of our frequency distribution using the  `fdist.N()` function. Add the line in your code:

```py
print('Total Number of distinct tokens in the bag : ', fdist.N())
```

5.  Now let's find out the 10 most common words in the selected corpus bag. The function `fdist.most_common()` will do this for us. Add the following two lines in the code:

```py
print('Following are the most common 10 words in the bag')
 print(fdist.most_common(10))
```

6.  Let us tabulate the entire frequency distribution using the `fdist.tabulate()` function. Add these lines in the code:

```py
print('Frequency Distribution on Personal Advertisements')
 print(fdist.tabulate())
```

7.  Now we will plot the graph of the frequency distribution with `cumulative` frequencies using the `fdist.plot()` function:

```py
fdist.plot(cumulative=True)
```

Let's run the program and see the output; we will discuss the same in the following section:

```py
['firefox.txt', 'grail.txt', 'overheard.txt', 'pirates.txt', 'singles.txt', 'wine.txt']

Count of the maximum appearing token " , " : 539

Total Number of distinct tokens in the bag : 4867

Following are the most common 10 words in the bag

[(',', 539), ('.', 353), ('/', 110), ('for', 99), ('and', 74), ('to', 

4), ('lady', 68), ('-', 66), ('seeks', 60), ('a', 52)]

Frequency Distribution on Personal Advertisements

, . / for and to lady .........

539 353 110 99 74 74 .........

None
```

You will also see the following graph pop up:

>![](img/d21e36af-4aae-4c1c-97a9-fe2f637fefe2.png)

# How it works...

Upon analyzing the output, we realize that all of it is very intuitive. But what is peculiar is that most of it is not making sense. The token with maximum frequency count is `,`. And when you look at the `10` most common tokens, again you can't make out much about the target dataset. The reason is that there is no preprocessing done on the corpus. In the third chapter, we will learn one of the most fundamental preprocessing steps called stop words treatment and will also see the difference it makes.

# Take an ambiguous word and explore all its senses using WordNet

From this recipe onwards, we will turn our attention to WordNet. As you can read in the title, we are going to explore what word sense is. To give an overview, English is a very ambiguous language. Almost every other word has a different meaning in different contexts. For example, let's take the simplest of words, *bat* which you will learn as part of the first 10 English words in a language course almost anywhere on the planet. The first meaning is a club used for hitting the ball in various sports such as cricket, baseball, tennis, squash, and so on.

Now a *bat* can also mean a nocturnal mammal that flies at nights. The *Bat* is also Batman's preferred and most advanced transportation vehicle according to DC comics. These are all noun variants; let's consider verb possibilities. *Bat* can also mean a slight wink (bat an eyelid). Consequently, it can also mean beating someone to pulp in a fight or a competition. We believe that's enough of an introduction; with this, let's move on to the actual recipe.

# Getting ready

Keeping the objective of the recipe in mind we have to choose a word for which we would be exploring its various senses as understood by WordNet. And yes, NLTK comes equipped with WordNet; you need not worry about installing any further libraries. So let's choose another simple word, *CHAIR*, as our sample for the purpose of this recipe.

# How to do it...

1.  Create a new file named `ambiguity.py` and add the following lines of code to start with:

```py
from nltk.corpus import wordnet as wn
chair = 'chair'
```

Here we imported the required NLTK corpus reader `wordnet` as the `wn` object. We can import it just like any another corpus readers we have used so far. In preparation for the next steps, we have created our string variable containing the word `chair`.

2.  Now is the most important step. Let's add two lines and I will elaborate what we are doing:

```py
chair_synsets = wn.synsets(chair)
print('Synsets/Senses of Chair :', chair_synsets, '\n\n')
```

The first line, though it looks simple, is actually the API interface that is accessing the internal WordNet database and fetching all the senses associated with the word `chair`. WordNet calls each of these senses  `synsets`. The next line simply asks the interpreter to print what it has fetched. Run this much and you should get an output like:

```py
Synsets/Senses of Chair : [Synset('chair.n.01'), Synset('professorship.n.01'), Synset('president.n.04'), Synset('electric_chair.n.01'), Synset('chair.n.05'), Synset('chair.v.01'), Synset('moderate.v.01')]
```

As you can see, the list contains seven `Synsets`, which means seven different senses of the word `Chair` exist in the WordNet database.

3.  We will add the following `for` loop, which will iterate over the list of `synsets` we have obtained and perform certain operations:

```py
for synset in chair_synsets:
  print(synset, ': ')
  print('Definition: ', synset.definition())
  print('Lemmas/Synonymous words: ', synset.lemma_names())
  print('Example: ', synset.examples(), '\n')
```

We are iterating over the list of `synsets` and printing the definition of each sense, associated lemmas/synonymous words, and example usage of each of the senses in a sentence. One typical iteration will print something similar to this:

```py
Synset('chair.v.01') :

Definition: act or preside as chair, as of an academic department in a university

Lemmas/Synonymous words: ['chair', 'chairman']

Example: ['She chaired the department for many years']
```

The first line is the name of `Synset`, the second line is the definition of this sense/`Synset`, the third line contains `Lemmas` associated with this `Synset`, and the fourth line is an example sentence.

We will obtain this output:

```py
Synsets/Senses of Chair : [Synset('chair.n.01'), Synset('professorship.n.01'), Synset('president.n.04'), Synset('electric_chair.n.01'), Synset('chair.n.05'), Synset('chair.v.01'), Synset('moderate.v.01')]

Synset('chair.n.01') :

Definition: a seat for one person, with a support for the back

Lemmas/Synonymous words: ['chair']

Example: ['he put his coat over the back of the chair and sat down']

Synset('professorship.n.01') :

Definition: the position of professor

Lemmas/Synonymous words: ['professorship', 'chair']

Example: ['he was awarded an endowed chair in economics']

Synset('president.n.04') :

Definition: the officer who presides at the meetings of an organization

Lemmas/Synonymous words: ['president', 'chairman', 'chairwoman', 

chair', 'chairperson']

Example: ['address your remarks to the chairperson']

Synset('electric_chair.n.01') :

Definition: an instrument of execution by electrocution; resembles an ordinary seat for one person

Lemmas/Synonymous words: ['electric_chair', 'chair', 'death_chair', 'hot_seat']

Example: ['the murderer was sentenced to die in the chair']

Synset('chair.n.05') :

Definition: a particular seat in an orchestra

Lemmas/Synonymous words: ['chair']

Example: ['he is second chair violin']

Synset('chair.v.01') :

Definition: act or preside as chair, as of an academic department in a university

Lemmas/Synonymous words: ['chair', 'chairman']

Example: ['She chaired the department for many years']

Synset('moderate.v.01') :

Definition: preside over

Lemmas/Synonymous words: ['moderate', 'chair', 'lead']

Example: ['John moderated the discussion']
```

# How it works...

As you can see, definitions, Lemmas, and example sentences of all seven senses of the word `chair` are seen in the output. Straightforward API interfaces are available for each of the operations as elaborated in the preceding code sample. Now, let's talk a little bit about how WordNet arrives at such conclusions. WordNet is a database of words that stores all information about them in a hierarchical manner. If we take a look at the current example Write about `synsets` and hierarchical nature of WordNet storage. The following diagram will explain it in more detail.

# Pick two distinct synsets and explore the concepts of hyponyms and hypernyms using WordNet

A hyponym is a word of a more specific meaning than a more generic word such as *bat,* which we explored in the introduction section of our previous recipe. What we mean by *more specific* is, for example, cricket bat, baseball bat, carnivorous bat, squash racket, and so on. These are more specific in terms of communicating what exactly we are trying to mean.

As opposed to a hyponym, a hypernym is a more general form or word of the same concept. For our example, *bat* is a more generic word and it could mean club, stick, artifact, mammal, animal, or organism. We can go as generic as the physical entity, living thing, or object and still be considered as a hypernym of the word *bat*.

# Getting ready

For the purpose of exploring the concepts of hyponym and hypernym, we have decided to select the synsets `bed.n.01` (first word sense of bed) and `woman.n.01` (second word sense of woman). Now we will explain the usage and meaning of the hypernym and hyponym APIs in the actual recipe section.

# How to do it...

1.  Create a new file named `HypoNHypernyms.py` and add following three lines:

```py
from nltk.corpus import wordnet as wn
woman = wn.synset(woman.n.02')
bed = wn.synset('bed.n.01')
```

We've imported the libraries and initialized the two synsets that we will use in later processing.

2.  Add the following two lines:

```py
print(woman.hypernyms())
woman_paths = woman.hypernym_paths()
```

It's a simple call to the `hypernyms()` API function on the woman `Synset`; it will return the set of synsets that are direct parents of the same. However, the `hypernym_paths()` function is a little tricky. It will return a list of sets. Each set contains the path from the root node to the woman `Synset`. When you run these two statements, you will see the two direct parents of the `Synset` woman as follows in the console:

```py
[Synset('adult.n.01'), Synset('female.n.02')]
```

Woman belongs to the adult and female categories in the hierarchical structure of the WordNet database.

3.  Now we will try to print the paths from root node to the `woman.n.01` node. To do so, add the following lines of code and nested `for` loop:

```py
for idx, path in enumerate(woman_paths):
  print('\n\nHypernym Path :', idx + 1)
for synset in path:
  print(synset.name(), ', ', end='')
```

As explained, the returned object is a list of sets ordered in such a way that it follows the path from the root to the `woman.n.01` node exactly as stored in the WordNet hierarchy. When you run, here's an example `Path`:

```py
Hypernym Path : 1

entity.n.01 , physical_entity.n.01 , causal_agent.n.01 , person.n.01 , adult.n.01 , woman.n.01
```

4.  Now let's work with `hyponyms`. Add the following two lines, which will fetch the `hyponyms` for the synset `bed.n.01` and print them to the console:

```py
types_of_beds = bed.hyponyms()
print('\n\nTypes of beds(Hyponyms): ', types_of_beds)
```

As explained, run them and you will see the following 20 synsets as output:

```py
Types of beds(Hyponyms): [Synset('berth.n.03'), Synset('built-in_bed.n.01'), Synset('bunk.n.03'), Synset('bunk_bed.n.01'), Synset('cot.n.03'), Synset('couch.n.03'), Synset('deathbed.n.02'), Synset('double_bed.n.01'), Synset('four-poster.n.01'), Synset('hammock.n.02'), Synset('marriage_bed.n.01'), Synset('murphy_bed.n.01'), Synset('plank-bed.n.01'), Synset('platform_bed.n.01'), Synset('sickbed.n.01'), Synset('single_bed.n.01'), Synset('sleigh_bed.n.01'), Synset('trundle_bed.n.01'), Synset('twin_bed.n.01'), Synset('water_bed.n.01')]
```

These are `Hyponyms` or more specific terms for the word sense `bed.n.01` within WordNet.

5.  Now let's print the actual words or `lemmas` that will make more sense to humans. Add the following line of code:

```py
print(sorted(set(lemma.name() for synset in types_of_beds for lemma in synset.lemmas())))
```

This line of code is pretty similar to what we did in the hypernym example nested `for` loop written in four lines, which is clubbed in a single line here (in other words, we're just showing off our skills with Python here). It will print the 26 `lemmas` that are very meaningful and specific words. Now let's look at the final output:

```py
Output: [Synset('adult.n.01'), Synset('female.n.02')]
Hypernym Path : 1
entity.n.01 , physical_entity.n.01 , causal_agent.n.01 , person.n.01 , adult.n.01 , woman.n.01 ,
Hypernym Path : 2
entity.n.01 , physical_entity.n.01 , object.n.01 , whole.n.02 , living_thing.n.01 , organism.n.01 , person.n.01 , adult.n.01 , woman.n.01 ,
Hypernym Path : 3
entity.n.01 , physical_entity.n.01 , causal_agent.n.01 , person.n.01 , female.n.02 , woman.n.01 ,
Hypernym Path : 4
entity.n.01 , physical_entity.n.01 , object.n.01 , whole.n.02 , living_thing.n.01 , organism.n.01 , person.n.01 , female.n.02 , woman.n.01 ,

Types of beds(Hyponyms): [Synset('berth.n.03'), Synset('built-in_bed.n.01'), Synset('bunk.n.03'), Synset('bunk_bed.n.01'), Synset('cot.n.03'), Synset('couch.n.03'), Synset('deathbed.n.02'), Synset('double_bed.n.01'), Synset('four-poster.n.01'), Synset('hammock.n.02'), Synset('marriage_bed.n.01'), Synset('murphy_bed.n.01'), Synset('plank-bed.n.01'), Synset('platform_bed.n.01'), Synset('sickbed.n.01'), Synset('single_bed.n.01'), Synset('sleigh_bed.n.01'), Synset('trundle_bed.n.01'), Synset('twin_bed.n.01'), Synset('water_bed.n.01')]

['Murphy_bed', 'berth', 'built-in_bed', 'built_in_bed', 'bunk', 'bunk_bed', 'camp_bed', 'cot', 'couch', 'deathbed', 'double_bed', 'four-poster', 'hammock', 'marriage_bed', 'plank-bed', 'platform_bed', 'sack', 'sickbed', 'single_bed', 'sleigh_bed', 'truckle', 'truckle_bed', 'trundle', 'trundle_bed', 'twin_bed', 'water_bed']
```

# How it works...

As you can see, `woman.n.01` has two hypernyms, namely adult and female, but it follows four different routes in the hierarchy of WordNet database from the root node `entity` to `woman` as shown in the output.

Similarly, the Synset `bed.n.01` has 20 hyponyms; they are more specific and less ambiguous (for nothing is unambiguous in English). Generally the hyponyms correspond to leaf nodes or nodes very much closer to the leaves in the hierarchy as they are the least ambiguous ones.

# Compute the average polysemy of nouns, verbs, adjectives, and adverbs according to WordNet

First, let's understand what polysemy is. Polysemy means many possible meanings of a word or a phrase. As we have already seen, English is an ambiguous language and more than one meaning usually exists for most of the words in the hierarchy. Now, turning back our attention to the problem statement, we must calculate the average polysemy based on specific linguistic properties of all words in WordNet. As we'll see, this recipe is different from previous recipes. It's not just an API concept discovery but we are going to discover a linguistic concept here (I'm all emotional to finally get a chance to do so in this chapter.

# Getting ready

I have decided to write the program to compute the polysemy of any one of the POS types of words and will leave it to you guys to modify the program to do so for the other three. I mean we shouldn't just spoon-feed everything, right? Not to worry! I will provide enough hints in the recipe itself to make it easier for you (for those who think it's already not very intuitive). Let's get on with the actual recipe then; we will compute the average polysemy of nouns alone.

# How to do it...

1.  Create a new file named `polysemy.py` and add these two initialization lines:

```py
from nltk.corpus import wordnet as wn
type = 'n'
```

We have initialized the POS type of words we are interested in and, of course, imported the required libraries. To be more descriptive, `n` corresponds to nouns.

2.  This is the most important line of code of this recipe:

```py
synsets = wn.all_synsets(type)
```

This API returns all `synsets` of type `n` that is a noun present in the WordNet database, full coverage. Similarly, if you change the POS type to a verb, adverb, or adjective, the API will return all words of the corresponding type (hint #1).

3.  Now we will consolidate all `lemmas` in each of the `synset` into a single mega list that we can process further. Add the following code to do that:

```py
lemmas = []
for synset in synsets:
  for lemma in synset.lemmas():
    lemmas.append(lemma.name())
```

This piece of code is pretty intuitive; we have a nested `for` loop that iterates over the list of `synsets` and the `lemmas` in each `synset` and adds them up in our mega list lemmas.

4.  Although we have all `lemmas` in the mega list, there is a problem. There are some duplicates as it's a list. Let's remove the duplicates and take the count of distinct `lemmas`:

```py
lemmas = set(lemmas)
```

Converting a list into a set will automatically deduplicate (yes, it's a valid English word, I invented it) the list.

5.  Now, the second most important step in the recipe. We count the senses of each `lemma` in the WordNet database:

```py
count = 0
for lemma in lemmas:
  count = count + len(wn.synsets(lemma, type))
```

Most of the code is intuitive; let's focus on the the API `wn.synsets(lemma, type)`. This API takes as input a word/lemma (as the first argument) and the POS type it belongs to and returns all the senses (`synsets`) belonging to the `lemma` word. Note that depending on what you provide as the POS type, it will return senses of the word of only the given POS type (hint #2).

6.  We have all the counts we need to compute the average polysemy. Let's just do it and print it on the console:

```py
print('Total distinct lemmas: ', len(lemmas))
print('Total senses :',count)
print('Average Polysemy of ', type,': ' , count/len(lemmas))
```

This prints the total distinct lemmas, the count of senses, and the average polysemy of POS type `n` or nouns:

```py
Output: Total distinct lemmas: 119034
Total senses : 152763
Average Polysemy of n : 1.2833560159282222
```

# How it works...

There is nothing much to say in this section, so I will instead give you some more information on how to go about computing the polysemy of the rest of the types. As you saw, *Noun -> 'n'*. Similarly, *Verbs -> 'v'*, *Adverbs -> 'r'*, and *Adjective -> 'a'* (hint # 3).

Now, I hope I have given you enough hints to get on with writing an NLP program of your own and not be dependent on the feed of the recipes.

