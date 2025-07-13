# Information Extraction and Text Classification

In this chapter, we will cover the following recipes:

*   Using inbuilt NERs
*   Creating, inversing, and using dictionaries
*   Creating your own NEs
*   Choosing the feature set
*   Segmenting sentences using classification
*   Classifying documents
*   Writing a POS tagger with context

# Introduction

Information retrieval is a vast area and has many challenges. In previous chapters, we understood regular expressions, grammars, **Parts-of-Speech** (**POS**) tagging, and chunking. The natural step after this process is to identify the Interested Entities in a given piece of text. To be clear, when we are processing large amounts of data, we are really interested in finding out whether any famous personalities, places, products, and so on are mentioned. These things are called **named entitie****s** in NLP. We will understand more about these with examples in the following recipes. Also, we will see how we can leverage the clues that are present in the input text to categorize large amounts of text, and many more examples will be explained. Stay tuned!

# Understanding named entities

So far, we have seen how to parse the text, identify parts of speech, and extract chunks from the text. The next thing that we need to look into is finding **proper nouns**, which are also called named entities.

Named entities help us understand more about what is being referred to in a given text so that we can further classify the data. Since named entities comprise more than one word, it is sometimes difficult to find these from the text.
Let's take up the following examples to understand what named entities are:

| **Sentence** | **Named entities** |
| Hampi is on the South Bank of Tungabhadra river | Hampi, Tungabhadra River |
| Paris is famous for Fashion | Paris |
| Burj Khalifa is one of the Skyscrapers in Dubai | Burj Khalifa , Dubai |
| Jeff Weiner is the CEO of LinkedIn | Jeff Weiner, LinkedIn |

Let's take a closer look at these and try to understand:

1.  Even though *South Bank* refers to a direction, it does not qualify as a named entity because we cannot uniquely identify the object from that.
2.  Even though *Fashion* is a noun, we cannot completely qualify it as named entity.
3.  *Skyscraper* is a noun, but there can be many possibilities for Skyscrapers.
4.  *CEO* is a role here; there are many possible persons who can hold this title. So, this also cannot be a named entity.

To further understand, let's just look at these NEs from a categories perspective:

| **Category** | **Examples of named entities** |
| `TIMEZONE` | Asia/Kolkata, IST, UTC |
| `LOCATION` | Mumbai, Kolkata, Egypt |
| `RIVERS` | Ganga, Yamuna, Nile |
| `COSMETICS` | Maybelline Deep Coral Lipstick, LOreal Excellence Creme Hair Color |
| `CURRENCY` | 100 bitcoins, 1.25 INR |
| `DATE` | 17-Aug-2017, 19-Feb-2016 |
| `TIME` | 10:10 AM |
| `PERSON` | Satya Nadella, Jeff Weiner, Bill Gates |

# Using inbuilt NERs

Python NLTK has built-in support for **Named Entity Recognition** (**NER**). In order to use this feature, first we need to recollect what we have done so far:

1.  Break a large document into sentences.
2.  Break the sentence into words (or tokens).
3.  Identify the parts of speech in the sentence.
4.  Extract chunks of consecutive words (non-overlapping) from the sentence.
5.  Assign IOB tags to these words based on the chunking patterns.

The next logical step would be to further extend the algorithms to find out the named entities as a sixth step. So, we will basically be using data that is preprocessed until step 5 as part of this example.

We will be using `treebank` data to understand the NER process. Remember, the data is already pre-tagged in IOB format. Without the training process, none of the algorithms that we are seeing here are going to work. (So, there is no magic!)

In order to understand the importance of the training process, let's take up an example. Say, there is a need for the Archaeological department to figure out which of the famous places in India are being tweeted and mentioned in social networking websites in the Kannada Language.

Assuming that they have already got the data somewhere and it is in terabytes  or even in petabytes, how do they find out all these names? This is where we need to take a sample dataset from the original input and do the  training process to further use this trained data set to extract the named entities in Kannada.

# Getting ready

You should have Python installed, along with the `nltk` library.

# How to do it...

1.  Open Atom editor (or you favorite programming editor).
2.  Create a new file called `NER.py`.
3.  Type the following source code:

![](img/0abb176a-d572-4e81-aa7d-8c8dd2d9fa5d.png)

4.  Save the file.
5.  Run the program using the Python interpreter.

6.  You will see the following output:

![](img/6018fa18-cf7f-452b-8d0a-e3612aa361cd.png)

# How it works...

The code looks so simple, right? However, all the algorithms are implemented in the `nltk` library. So, let's dig into how this simple program gives what we are looking for. This instruction imports the `nltk` library into the program:

```py
import nltk
```

These three instructions define a new function called `sampleNE()`. We are importing the first tagged sentence from the `treebank` corpus and then passing it to the `nltk.ne_chunk()` function to extract the named entities. The output from this program includes all the named entities with their proper category:

```py
def sampleNE():
  sent = nltk.corpus.treebank.tagged_sents()[0]
  print(nltk.ne_chunk(sent))
```

These three instructions define a new function called `sampleNE2()`. We are importing the first tagged sentence from the `treebank` corpus and then passing it to the `nltk.ne_chunk()` function to extract the named entities. The output from this program includes all the named entities without any proper category. This is helpful if the training dataset is not accurate enough to tag the named entities with the proper category such as person, organization, location, and so on:

```py
def sampleNE2():
  sent = nltk.corpus.treebank.tagged_sents()[0]
  print(nltk.ne_chunk(sent, binary=True))
```

These three instructions will call the two sample functions that we have defined before and print the results on the screen.

```py
if __name__ == '__main__':
  sampleNE()
  sampleNE2()
```

# Creating, inversing, and using dictionaries

Python, as a general-purpose programming language, has support for many built-in data structures. Of those, one of the most powerful data structures are dictionaries. Before we jump into what dictionaries are, let's try to understand what these data structures are used for. Data structures, in short, help programmers to store, retrieve, and traverse through data that is stored in these structures. Each data structure has its own sets of behaviors and performance benefits that programmers should understand before selecting them for a given task at hand.
Let's get back to dictionaries. The basic use case of dictionaries can be explained with a simple example:

```py
All the flights got delayed due to bad weather
```

We can use POS identification on the preceding sentence. But if someone were to ask what POS `flights` is in this sentence, we should have an efficient way to look for this word. This is where dictionaries come into play. They can be thought of as **one-to-one** mappings between data of interest. Again this one-to-one is at the highest level of abstraction of the data unit that we are talking about. If you are an expert programmer in Python, you know how to do **many-to-many** also. In this simple example, we need something like this:

```py
flights -> Noun
Weather -> Noun
```

Now let's answer a different question. Is it possible to print the list of all the words in the sentence that are nouns? Yes, for this too, we will learn how to use a Python dictionary.

# Getting ready

You should have Python installed, along with the `nltk` library, in order to run this example.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `Dictionary.py`.

3.  Type the following source code:

![](img/48346cdf-5656-4b23-bacc-c095c5296b82.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/8633a272-e323-4220-ae47-b90ccb3e4f2a.png)

# How it works...

Now, let's understand more about dictionaries by going through the instructions we have written so far. We are importing the `nltk` library into the program:

```py
import nltk
```

We are defining a new class called `LearningDictionary`:

```py
class LearningDictionary():
```

We are creating a constructor for `LearningDictionary` that takes `sentence` text as an argument:

```py
def __init__(self, sentence):
```

This instruction breaks the sentence into words using the `nltk.word_tokenize()` function and saves the result in the class member `words`:

```py
self.words = nltk.word_tokenize(sentence)
```

This instruction identifies the POS for `words` and saves the result in the class member tagged:

```py
self.tagged = nltk.pos_tag(self.words)
```

This instruction invokes the `buildDictionary()` function that is defined in the class:

```py
self.buildDictionary()
```

This instruction invokes the `buildReverseDictionary()` function that is defined in the class:

```py
 self.buildReverseDictionary()
```

This instruction defines a new class member function called `buildDictionary()`*:*

```py
 def buildDictionary(self):
```

This instruction initializes a empty `dictionary` variable in the class. These two instructions iterate over all the tagged `pos` list elements and then assign each `word` to the `dictionary` as key and the POS as value of the key:

```py
self.dictionary = {}
for (word, pos) in self.tagged:
  self.dictionary[word] = pos
```

This instruction defines another class member function called `buildReverseDictionary()`:

```py
def buildReverseDictionary(self):
```

This instruction initializes an empty `dictionary` to a class member, `rdictionary`:

```py
self.rdictionary = {}
```

This instruction iterates over all the `dictionary` keys and puts the key of `dictionary` into a local variable called `key`:

```py
 for key in self.dictionary.keys():
```

This instruction extracts the `value` (POS) of the given `key` (word) and stores it in a local variable called `value`:

```py
value = self.dictionary[key]
```

These four instructions check whether a given `key` (word) is already in the reverse dictionary variable (`rdictionary`). If it is, then we append the currently found word to the list. If the word is not found, then we create a new list of size one with the current word as the member:

```py
if value not in self.rdictionary:
  self.rdictionary[value] = [key]
else:
  self.rdictionary[value].append(key)
```

This function returns `Yes` or `No` depending on whether a given word is found in the `dictionary`:

```py
def isWordPresent(self, word):
  return 'Yes' if word in self.dictionary else 'No'
```

This function returns the POS for the given word by looking into `dictionary`. If the value is not found, a special value of `None` is returned:

```py
def getPOSForWord(self, word):
  return self.dictionary[word] if word in self.dictionary else None
```

These two instructions define a function that returns all the words in the sentence with a given POS by looking into `rdictionary` (reverse dictionary). If the POS is not found, a special value of `None` is returned:

```py
def getWordsForPOS(self, pos):
  return self.rdictionary[pos] if pos in self.rdictionary else None
```

We define a variable called `sentence`, which stores the string that we are interested in parsing:

```py
sentence = "All the flights got delayed due to bad weather"
```

Initialize the `LearningDictionary()` class with `sentence` as a parameter. Once the class object is created, it is assigned to the learning variable:

```py
learning = LearningDictionary(sentence)
```

We create a list of `words` that we are interested in knowing the POS of. If you see carefully, we have included a few words that are not in the sentence:

```py
words = ["chair", "flights", "delayed", "pencil", "weather"]
```

We create a list of `pos` for which we are interested in seeing the words that belong to these POS classifications:

```py
pos = ["NN", "VBS", "NNS"]
```

These instructions iterate over all the `words`, take one `word` at a time, check whether the `word` is in the dictionary by calling the `isWordPresent()` function of the object, and then print its status. If the `word` is present in the dictionary, then we print the POS for the word:

```py
for word in words:
  status = learning.isWordPresent(word)
  print("Is '{}' present in dictionary ? : '{}'".format(word, status))
  if status is True:
    print("\tPOS For '{}' is '{}'".format(word, learning.getPOSForWord(word)))
```

In these instructions, we iterate over all the `pos`. We take one word at a time and then print the words that are in this POS using the `getWordsForPOS()` function.

```py
for pword in pos:
  print("POS '{}' has '{}' words".format(pword, learning.getWordsForPOS(pword)))
```

# Choosing the feature set

Features are one of the most powerful components of `nltk` library. They represent clues within the language for easy tagging of the data that we are dealing with. In Python terminology, features are expressed as dictionaries, where the keys are the labels and the values are the clues extracted from input data.
Let's say we are dealing with some transport department data and we are interested in finding out whether a given vehicle number belongs to the Government of Karnataka or not. Right now we have no clue about the data we are dealing with. So how can we tag the given numbers accurately?

Let's try to learn how the vehicle numbers give some clues about what they mean:

| **Vehicle number** | **Clues about the pattern** |
| **KA-[0-9]{2} [0-9]{2}** | Normal vechicle number |
| **KA-[0-9]{2}-F** | KSRTC, BMTC vehicles |
| **KA-[0-9]{2}-G** | Government vehicles |

Using these clues (features), let's try to come up with a simple program that can tell us the classification of a given input number.

# Getting ready

You should have Python installed, along with the `nltk` library.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `Features.py`.

3.  Type the following source code:

![](img/045dfc44-f4cf-45eb-a7d2-69fc0792a43c.png)

4.  Save the file.
5.  Run the program using the Python interpreter.

6.  You will see the following output:

![](img/cb969140-d0cc-4018-9883-f9a2aaf9c635.png)

# How it works...

Now, let's see what our program does. These two instructions import the `nltk` and `random` libraries into the current program:

```py
import nltk
import random
```

We are defining a list of Python tuples, where the first element in the tuple is the vehicle number and the second element is the predefined label that is applied to the number.

These instructions define that all the numbers are classified into three labels—`rtc`, `gov`, and `oth`:

```py
sampledata = [
  ('KA-01-F 1034 A', 'rtc'),
  ('KA-02-F 1030 B', 'rtc'),
  ('KA-03-FA 1200 C', 'rtc'),
  ('KA-01-G 0001 A', 'gov'),
  ('KA-02-G 1004 A', 'gov'),
  ('KA-03-G 0204 A', 'gov'),
  ('KA-04-G 9230 A', 'gov'),
  ('KA-27 1290', 'oth')
]
```

This instruction shuffles all of the data in the `sampledata` list to make sure that the algorithm is not biased by the order of elements in the input sequence:

```py
random.shuffle(sampledata)
```

These are the test vehicle numbers for which we are interested in finding the category:

```py
testdata = [
  'KA-01-G 0109',
  'KA-02-F 9020 AC',
  'KA-02-FA 0801',
  'KA-01 9129'
]
```

This instruction defines a new function called `learnSimpleFeatures`:

```py
def learnSimpleFeatures():
```

These instructions define a new function, `vehicleNumberFeature()`, which takes the vehicle number and returns the seventh character in the that number. The return type is `dictionary`:

```py
def vehicleNumberFeature(vnumber):
  return {'vehicle_class': vnumber[6]}
```

This instruction creates a list of feature tuples, where the first member in the tuple is feature dictionary and the second member in tuple is the label of the data. After this instruction, the input vehicle numbers in `sampledata` are no longer visible. This is one of the key things to remember:

```py
featuresets = [(vehicleNumberFeature(vn), cls) for (vn, cls) in sampledata]
```

This instruction trains `NaiveBayesClassifier` with the feature dictionary and the labels that are applied to `featuresets`. The result is available in the classifier object, which we will use further:

```py
classifier = nltk.NaiveBayesClassifier.train(featuresets)
```

These instructions iterate over the test data and then print the label of the data from the classification done using `vehicleNumberFeature`. Observe the output carefully. You will see that the feature extraction function that we have written does not perform well in labeling the numbers correctly:

```py
for num in testdata:
  feature = vehicleNumberFeature(num)
  print("(simple) %s is of type %s" %(num, classifier.classify(feature)))
```

This instruction defines a new function called `learnFeatures`:

```py
def learnFeatures():
```

These instructions define a new function called `vehicleNumberFeature` that returns the feature dictionary with two keys. One key, `vehicle_class`, returns the character at position `6` in the string, and `vehicle_prev` has the character at position `5`. These kinds of clues are very important to make sure we eliminate bad labeling of data:

```py
def vehicleNumberFeature(vnumber):
  return {
    'vehicle_class': vnumber[6],
    'vehicle_prev': vnumber[5]
  }
```

This instruction creates a list of `featuresets` and input labels by iterating over of all the input trained data. As before, the original input vehicle numbers are no longer present here:

```py
featuresets = [(vehicleNumberFeature(vn), cls) for (vn, cls) in sampledata]
```

This instruction creates a `NaiveBayesClassifier.train()` function on `featuresets` and returns the object for future use:

```py
classifier = nltk.NaiveBayesClassifier.train(featuresets)
```

These instructions loop through the `testdata` and print the classification of the input vehicle number based on the trained dataset. Here, if you observe carefully, the false-positive is not there:

```py
for num in testdata:
  feature = vehicleNumberFeature(num)
  print("(dual) %s is of type %s" %(num, classifier.classify(feature)))
```

Invoke both the functions and print the results on the screen.

```py
learnSimpleFeatures()
learnFeatures()
```

If we observe carefully, we realize that the first function's results have one false positive, where it cannot identify the `gov` vehicle. This is where the second function performs well, as it has more features that improve accuracy.

# Segmenting sentences using classification

A natural language that supports question marks (?), full stops (.), and exclamations (!) poses a challenge to us in identifying whether a statement has ended or it still continues after the punctuation characters.

This is one of the classic problems to solve.

In order to solve this problem, let's find out the features (or clues) that we can leverage to come up with a classifier and then use the classifier to extract sentences in large text.
If we encounter a punctuation mark like `.` then it ends a sentence If we encounter a punctuation mark like `.` and the next word's first letter is a capital letter, then it ends a sentence.

Let's try to write a simple classifier using these two features to mark sentences.

# Getting ready

You should have Python installed along with `nltk` library.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new File called `Segmentation.py`.

3.  Type the following source code:

![](img/f6c31463-5cf4-4682-8e0f-18aa270db410.png) 

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/c170092c-1016-4885-bd4f-f386b152bf19.png)

# How it works...

This instruction imports the `nltk` library into the program:

```py
import nltk
```

This function defines a modified feature extractor that returns a tuple containing the dictionary of the features and `True` or `False` to tell whether this feature indicates a sentence boundary or not:

```py
def featureExtractor(words, i):
    return ({'current-word': words[i], 'next-is-upper': words[i+1][0].isupper()}, words[i+1][0].isupper())
```

This function takes a `sentence` as input and returns a list of `featuresets` that is a list of tuples, with the feature dictionary and `True` or `False`:

```py
def getFeaturesets(sentence):
  words = nltk.word_tokenize(sentence)
  featuresets = [featureExtractor(words, i) for i in range(1, len(words) - 1) if words[i] == '.']
  return featuresets
```

This function takes the input text, breaks it into words, and then traverses through each word in the list. Once it encounters a full stop, it calls `classifier` to conclude whether it has encountered a sentence end. If the `classifier` returns `True`, then the sentence is found and we move on to the next word in the input. The process is repeated for all words in the input:

```py
def segmentTextAndPrintSentences(data):
  words = nltk.word_tokenize(data)
  for i in range(0, len(words) - 1):
    if words[i] == '.':
      if classifier.classify(featureExtractor(words, i)[0]) == True:
        print(".")
      else:
        print(words[i], end='')
    else:
      print("{} ".format(words[i]), end='')
    print(words[-1])
```

These instructions define a few variables for training and evaluation of our classifier:

```py
# copied the text from https://en.wikipedia.org/wiki/India
traindata = "India, officially the Republic of India (Bhārat Gaṇarājya),[e] is a country in South Asia. it is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world. It is bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast. It shares land borders with Pakistan to the west;[f] China, Nepal, and Bhutan to the northeast; and Myanmar (Burma) and Bangladesh to the east. In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives. India's Andaman and Nicobar Islands share a maritime border with Thailand and Indonesia."

testdata = "The Indian subcontinent was home to the urban Indus Valley Civilisation of the 3rd millennium BCE. In the following millennium, the oldest scriptures associated with Hinduism began to be composed. Social stratification, based on caste, emerged in the first millennium BCE, and Buddhism and Jainism arose. Early political consolidations took place under the Maurya and Gupta empires; the later peninsular Middle Kingdoms influenced cultures as far as southeast Asia. In the medieval era, Judaism, Zoroastrianism, Christianity, and Islam arrived, and Sikhism emerged, all adding to the region's diverse culture. Much of the north fell to the Delhi sultanate; the south was united under the Vijayanagara Empire. The economy expanded in the 17th century in the Mughal Empire. In the mid-18th century, the subcontinent came under British East India Company rule, and in the mid-19th under British crown rule. A nationalist movement emerged in the late 19th century, which later, under Mahatma Gandhi, was noted for nonviolent resistance and led to India's independence in 1947."
```

Extract all the features from the `traindata` variable and store it in `traindataset`:

```py
traindataset = getFeaturesets(traindata)
```

Train the `NaiveBayesClassifier` on `traindataset` to get  `classifier` as object:

```py
classifier = nltk.NaiveBayesClassifier.train(traindataset)
```

Invoke the function on `testdata` and print all the found sentences as output on the screen:

```py
segmentTextAndPrintSentences(testdata)
```

# Classifying documents

In this recipe, we will learn how to write a classifier that can be used to classify documents. In our case, we will classify **rich site summary** (**RSS**) feeds. The list of categories is known ahead of time, which is important for the classification task.

In this information age, there are vast amounts of text available. Its humanly impossible for us to properly categorize all information for further consumption. This is where categorization algorithms help us to properly categorize the newer sets of documents that are being produced based on the training given on sample data.

# Getting ready

You should have Python installed, along with the `nltk` and `feedparser` libraries.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `DocumentClassify.py`.

3.  Type the following source code:

![](img/8df78831-67d3-4c71-bb2e-22278a21cc58.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see this output:

![](img/1cb21192-bb6d-41d2-be3d-7391822aac2e.png)

# How it works...

Let's see how this document classification works. Importing three libraries into the program:

```py
import nltk
import random
import feedparser
```

This instruction defines a new dictionary with two RSS feeds pointing to Yahoo! sports. They are pre-categorized. The reason we have selected these RSS feeds is that data is readily available and categorized for our example:

```py
urls = {
  'mlb': 'https://sports.yahoo.com/mlb/rss.xml',
  'nfl': 'https://sports.yahoo.com/nfl/rss.xml',
}
```

Initializing the empty dictionary variable `feedmap` to keep the list of RSS feeds in memory until the program terminates:

```py
feedmap = {}
```

Getting the list of `stopwords` in English and storing it in the `stopwords` variable:

```py
stopwords = nltk.corpus.stopwords.words('english')
```

This function, `featureExtractor()`, takes list of words and then adds them to the dictionary, where each key is the word and the value is `True`. The dictionary is returned, which are the features for the given input `words`:

```py
def featureExtractor(words):
  features = {}
  for word in words:
    if word not in stopwords:
      features["word({})".format(word)] = True
    return features
```

Empty list to store all the correctly labeled `sentences`:

```py
sentences = []
```

Iterate over all the `keys()` of the dictionary called `urls` and store the key in a variable called `category`:

```py
for category in urls.keys():
```

Download one feed and store the result in the `feedmap[category]` variable using the `parse()` function from the `feedparser` module:

```py
feedmap[category] = feedparser.parse(urls[category])
```

Display the `url` that is being downloaded on the screen, using Python's built-in the `print` function:

```py
print("downloading {}".format(urls[category]))
```

Iterate over all the RSS entries and store the current entry in a variable called `entry` variable:

```py
for entry in feedmap[category]['entries']:
```

Take the `summary` (news text) of the RSS feed item into the `data` variable:

```py
data = entry['summary']
```

We brea `summary` into `words` based on space so that we can pass these to `nltk` for feature extraction:

```py
words = data.split()
```

Store all `words` in the current RSS feed item, along with `category` it belongs to, in a tuple:

```py
sentences.append((category, words))
```

Extract all the features of  `sentences` and store them in the variable `featuresets`. Later, do `shuffle()` on this array so that all the elements in the list are randomized for the algorithm:

```py
featuresets = [(featureExtractor(words), category) for category, words in sentences]
random.shuffle(featuresets)
```

Create two datasets, one `trainset` and the other `testset`, for our analysis:

```py
total = len(featuresets)
off = int(total/2)
trainset = featuresets[off:]
testset = featuresets[:off]
```

Create a `classifier` using the `trainset` data by using the `NaiveBayesClassifier` module's `train()` function:

```py
classifier = nltk.NaiveBayesClassifier.train(trainset)
```

Print the accuracy of  `classifier` using `testset`:

```py
print(nltk.classify.accuracy(classifier, testset))
```

Print the informative features about this data using the built-in function of `classifier`:

```py
classifier.show_most_informative_features(5)
```

Take four sample entries from the `nfl` RSS item. Try to tag the document based on `title` (remember, we have classified them based on `summary`):

```py
for (i, entry) in enumerate(feedmap['nfl']['entries']):
  if i < 4:
    features = featureExtractor(entry['title'].split())
    category = classifier.classify(features)
    print('{} -> {}'.format(category, entry['title']))
```

# Writing a POS tagger with context

In previous recipes, we have written regular-expression-based POS taggers that leverage word suffixes such as *ed*, *ing*, and so on to check whether the word is of a given POS or not. In the English language, the same word can play a dual role depending on the context in which it is used.
For example, the word `address` can be both noun and verb depending on the context:

```py
"What is your address when you're in Bangalore?"
"the president's address on the state of the economy."
```

Let's try to write a program that leverages the feature extraction concept to find the POS of the words in the sentence.

# Getting ready

You should have Python installed, along with `nltk`.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `ContextTagger.py`.

3.  Type the following source code:

![](img/1e05adaf-bf14-4c51-b890-3dfd893cf92f.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/8fe4c10f-51e2-4850-8334-c218929e0b99.png)

# How it works...

Let's see how the current program works. This instruction imports the `nltk` libarary into the program:

```py
import nltk
```

Some sample strings that indicate the dual behavior of the words, `address`, `laugh`:

```py

sentences = [
  "What is your address when you're in Bangalore?",
  "the president's address on the state of the economy.",
  "He addressed his remarks to the lawyers in the audience.",
  "In order to address an assembly, we should be ready",
  "He laughed inwardly at the scene.",
  "After all the advance publicity, the prizefight turned out to be a laugh.",
  "We can learn to laugh a little at even our most serious foibles."
]
```

This function takes `sentence` strings and returns a list of lists, where the inner lists contain the words along with their POS tags:

```py

def getSentenceWords():
  sentwords = []
  for sentence in sentences:
    words = nltk.pos_tag(nltk.word_tokenize(sentence))
    sentwords.append(words)
    return sentwords

```

In order to set up a baseline and see how bad the tagging can be, this function explains how  `UnigramTagger` can be used to print the POS of the words just by looking at the current word. We are feeding the sample text to it as learning. This `tagger` performs very badly when compared to the built-in tagger that `nltk` comes with. But this is just for our understanding:

```py

def noContextTagger():
  tagger = nltk.UnigramTagger(getSentenceWords())
  print(tagger.tag('the little remarks towards assembly are laughable'.split()))
```

Defining a new function called `withContextTagger()`:

```py

def withContextTagger():
```

This function does feature extraction on a given set of words and returns a dictionary of the last three characters of the current word and previous word information:

```py
def wordFeatures(words, wordPosInSentence):
  # extract all the ing forms etc
  endFeatures = {
    'last(1)': words[wordPosInSentence][-1],
    'last(2)': words[wordPosInSentence][-2:],
    'last(3)': words[wordPosInSentence][-3:],
  }
  # use previous word to determine if the current word is verb or noun
  if wordPosInSentence > 1:
    endFeatures['prev'] = words[wordPosInSentence - 1]
  else:
    endFeatures['prev'] = '|NONE|'
    return endFeatures
```

We are building a `featuredata` list. It contains tuples of `featurelist` and `tag` members, which we will use to classify using `NaiveBayesClassifier`:

```py
allsentences = getSentenceWords()
featureddata = []
for sentence in allsentences:
  untaggedSentence = nltk.tag.untag(sentence)
  featuredsentence = [(wordFeatures(untaggedSentence, index), tag) for index, (word, tag) in enumerate(sentence)]
  featureddata.extend(featuredsentence)
```

We take 50% for training and 50% of the feature extracted data to test our classifier:

```py
breakup = int(len(featureddata) * 0.5)
traindata = featureddata[breakup:]
testdata = featureddata[:breakup]
```

This instruction creates `classifier` using the training data:

```py
classifier = nltk.NaiveBayesClassifier.train(traindata)
```

This instruction prints the accuracy of the classifier using `testdata`:

```py
print("Accuracy of the classifier : {}".format(nltk.classify.accuracy(classifier, testdata)))
```

These two functions print the results of two preceding functions' computations.

```py
noContextTagger()
withContextTagger()
```

