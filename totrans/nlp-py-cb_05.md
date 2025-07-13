# POS Tagging and Grammars

In this chapter, we will cover the following recipes:

*   Exploring the in-built tagger
*   Writing your own tagger
*   Training your own tagger
*   Learning to write your own grammar
*   Writing a probabilistic context-free grammar--CFG
*   Writing a recursive CFG

# Introduction

This chapter primarily focuses on learning the following subjects using Python NLTK:

*   Taggers
*   CFG

Tagging is the process of classifying the words in a given sentence using **parts of speech** (**POS**). Software that helps achieve this is called **tagger**. NLTK has support for a variety of taggers. We will go through the following taggers as part of this chapter:

*   In-built tagger
*   Default tagger
*   Regular expression tagger
*   Lookup tagger

CFG describes a set of rules that can be applied to text in a formal language specification to generate newer sets of text.

CFG in a language comprises the following things:

*   Starting token
*   A set of tokens that are terminals (ending symbols)
*   A set of tokens that are non-terminals (non-ending symbols)
*   Rules or productions that define rewrite rules that help transform non-terminals to either terminals or non-terminals

# Exploring the in-built tagger

In the following recipe, we use the Python NLTK library to understand more about the POS tagging features in a given text.

We will make use of the following technologies from the Python NLTK library:

*   Punkt English tokenizer
*   Averaged perception tagger

The datasets for these taggers can be downloaded from your NLTK distribution by invoking `nltk.download()` from the Python prompt.

# Getting ready

You should have a working Python (Python 3.6 is preferred) installed in your system along with the NLTK library and all its collections for optimal experience.

# How to do it...

1.  Open atom editor (or favorite programming editor).
2.  Create a new file called `Exploring.py`.

3.  Type the following source code:

![](img/60ff55a2-66f9-42da-8a22-035a143c1dc6.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/06bf1f75-3720-4b26-925b-fb807cebef24.png)

# How it works...

Now, let's go through the program that we have just written and dig into the details:

```py
import nltk
```

This is the first instruction in our program, which instructs the Python interpreter to load the module from disk to memory and make the NLTK library available for use in the program:

```py
simpleSentence = "Bangalore is the capital of Karnataka."
```

In this instruction, we are creating a variable called `simpleSentence` and assigning a hard coded string to it:

```py
wordsInSentence = nltk.word_tokenize(simpleSentence)
```

In this instruction, we are invoking the NLTK built-in tokenizer function `word_tokenize()`; it breaks a given sentence into words and returns a Python `list` datatype. Once the result is computed by the function, we assign it to a variable called `wordsInSentence` using the `=` (equal to) operator:

```py
print(wordsInSentence)
```

In this instruction, we are calling the Python built-in `print()` function, which displays the given data structure on the screen. In our case, we are displaying the list of all words that are tokenized. See the output carefully; we are displaying a Python `list` data structure on screen, which consists of all the strings separated by commas, and all the list elements are enclosed in square brackets:

```py
partsOfSpeechTags = nltk.pos_tag(wordsInSentence)
```

In this instruction we are invoking the NLTK built-in tagger `pos_tag()`, which takes a list of words in the `wordsInSentence` variable and identifies the POS. Once the identification is complete, a list of tuples. Each tuple has the tokenized word and the POS identifier:

```py
print(partsOfSpeechTags)
```

In this instruction, we are invoking the Python built-in `print()` function, which prints the given parameter to the screen. In our case, we can see a list of tuples, where each tuple consists of the original word and POS identifier.

# Writing your own tagger

In the following recipe, we will explore the NLTK library by writing our own taggers. The following types of taggers will be written:

*   Default tagger
*   Regular expression tagger
*   Lookup tagger

# Getting ready

You should have a working Python (Python 3.6 is preferred) installed in your system along with the NLTK library and all its collections for optimal experience.

You should also have `python-crfsuite` installed to run this recipe.

# How to do it...

1.  Open your atom editor (or favorite programming editor).
2.  Create a new file called `OwnTagger.py`.
3.  Type the following source code:

![](img/7ead53b2-d77f-44e1-b0d6-50863789a0ba.png)

4.  Save the File.
5.  Run the program using the Python interpreter.

6.  You will see the following output.

![](img/eb492d87-304b-4d3e-ad8b-4587cb0af1ed.png)

# How it works...

Now, let's go through the program that we have just written to understand more:

```py
import nltk
```

This is the first instruction in our program; it instructs the Python interpreter to load the module from disk to memory and make the NLTK library available for use in the program:

```py
def learnDefaultTagger(simpleSentence):
  wordsInSentence = nltk.word_tokenize(simpleSentence)
  tagger = nltk.DefaultTagger("NN")
  posEnabledTags = tagger.tag(wordsInSentence)
  print(posEnabledTags)
```

All of these instructions are defining a new Python function that takes a string as input and prints the words in this sentence along with the default tag on screen. Let's further understand this function to see what it's trying to do:

```py
def learnDefaultTagger(simpleSentence):
```

In this instruction, we are defining a new Python function called `learnDefaultTagger`; it takes a parameter named `simpleSentence`:

```py
wordsInSentence = nltk.word_tokenize(simpleSentence)
```

In this instruction, we are calling the `word_tokenize` function from the NLTK library. We are passing `simpleSentence` as the first parameter to this function. Once the data is computed by this function, the return value is stored in the `wordsInSentence` variable. Which are list of words:

```py
tagger = nltk.DefaultTagger("NN")
```

In this instruction, we are creating an object of the `DefaultTagger()` class from the Python `nltk` library with `NN` as the argument passed to it. This will initialize the tagger and assign the instance to the `tagger` variable:

```py
posEnabledTags = tagger.tag(wordsInSentence)
```

In this instruction, we are calling the `tag()` function of the `tagger` object, which takes the tokenized words from the `wordsInSentence` variable and returns the list of tagged words. This is saved in `posEnabledTags`. Remember that all the words in the sentence will be tagged as `NN` as that's what the tagger is supposed to do. This is like a very basic level of tagging without knowing anything about POS:

```py
print(posEnabledTags)
```

Here we are calling Python's built-in `print()` function to inspect the contents of the `posEnabledTags` variable. We can see that all the words in the sentence will be tagged with `NN:`

```py
def learnRETagger(simpleSentence):
  customPatterns = [
    (r'.*ing$', 'ADJECTIVE'),
    (r'.*ly$', 'ADVERB'),
    (r'.*ion$', 'NOUN'),
    (r'(.*ate|.*en|is)$', 'VERB'),
    (r'^an$', 'INDEFINITE-ARTICLE'),
    (r'^(with|on|at)$', 'PREPOSITION'),
    (r'^\-?[0-9]+(\.[0-9]+)$', 'NUMBER'),
    (r'.*$', None),
  ]
  tagger = nltk.RegexpTagger(customPatterns)
  wordsInSentence = nltk.word_tokenize(simpleSentence)
  posEnabledTags = tagger.tag(wordsInSentence)
  print(posEnabledTags)
```

These are the instructions to create a new function called `learnRETagger()`, which takes a string as input and prints the list of all tokens in the string with properly identified tags using the regular expression tagger as output.

Let's try to understand one instruction at a time:

```py
def learnRETagger(simpleSentence):
```

We are defining a new Python function named `*learnRETagger*` to take a parameter called *`simpleSentence`.*

In order to understand the next instruction, we should learn more about Python lists, tuples, and regular expressions:

*   A Python list is a data structure that is an ordered set of elements

*   A Python tuple is a immutable (read-only) data structure that is an ordered set of elements

*   Python regular expressions are strings that begin with the letter `r` and follow the standard PCRE notation:

```py
customPatterns = [
  (r'.*ing$', 'ADJECTIVE'),
  (r'.*ly$', 'ADVERB'),
  (r'.*ion$', 'NOUN'),
  (r'(.*ate|.*en|is)$', 'VERB'),
  (r'^an$', 'INDEFINITE-ARTICLE'),
  (r'^(with|on|at)$', 'PREPOSITION'),
  (r'^\-?[0-9]+(\.[0-9]+)$', 'NUMBER'),
  (r'.*$', None),
]
```

Even though this looks big, this is a single instruction that does many things:

*   Creating a variable called `customPatterns`
*   Defining a new Python list datatype with `[`
*   Adding eight elements to this list
*   Each element in this list is a tuple that has two items in it
*   The first item in the tuple is a regular expression
*   The second item in the tuple is a string

Now, translating the preceding instruction into a human-readable form, we have added eight regular expressions to tag the words in a sentence to be any of `ADJECTIVE`, `ADVERB`, `NOUN`, `VERB`, `INDEFINITE-ARTICLE`, `PREPOSITION`, `NUMBER`, or `None` type.

We do this by identifying certain patterns in English words identifiable as a given POS.

In the preceding example, these are the clues we are using to tag the POS of English words:

*   Words that end with `ing` can be called `ADJECTIVE`, for example, `running`
*   Words that end with `ly` can be called `ADVERB`, for example, `willingly`
*   Words that end with `ion` can be called `NOUN`, for example, `intimation`
*   Words that end with `ate` or `en` can be called `VERB`, for example, `terminate`, `darken`, or `lighten`
*   Words that end with `an` can be called `INDEFINITE-ARTICLE`
*   Words such as `with`, `on`, or `at` are `PREPOSITION`
*   Words that are like, `-123.0`, `984` can be called `NUMBER`
*   We are tagging everything else as `None`, which is a built-in Python datatype used to represent nothing

```py
tagger = nltk.RegexpTagger(customPatterns)
```

In this instruction, we are creating an instance of the NLTK built-in regular expression tagger `RegexpTagger`. We are passing the list of tuples in the `customPatterns` variable as the first parameter to the class to initialize the object. This object can be referenced in future with the variable named `tagger`:

```py
wordsInSentence = nltk.word_tokenize(simpleSentence)
```

Following the general process, we first try to tokenize the string in `simpleSentence` using the NLTK built-in `word_tokenize()` function and store the list of tokens in the `wordsInSentence` variable:

```py
posEnabledTags = tagger.tag(wordsInSentence)
```

Now we are invoking the regular expression tagger's `tag()` function to tag all the words that are in the `wordsInSentence` variable. The result of this tagging process is stored in the `posEnabledTags` variable:

```py
print(posEnabledTags)
```

We are calling the Python built-in `print()` function to display the contents of the `posEnabledTags` data structure on screen:

```py
def learnLookupTagger(simpleSentence):
  mapping = {
    '.': '.', 'place': 'NN', 'on': 'IN',
    'earth': 'NN', 'Mysore' : 'NNP', 'is': 'VBZ',
    'an': 'DT', 'amazing': 'JJ'
  }
  tagger = nltk.UnigramTagger(model=mapping)
  wordsInSentence = nltk.word_tokenize(simpleSentence)
  posEnabledTags = tagger.tag(wordsInSentence)
  print(posEnabledTags)
```

Let's take a closer look:

```py
def learnLookupTagger(simpleSentence):
```

We are defining a new function, `learnLookupTagger`, which takes a string as parameter into the `simpleSentence` variable:

```py
tagger = nltk.UnigramTagger(model=mapping)
```

In this instruction, we are calling `UnigramTagger` from the `nltk` library. This is a lookup tagger that takes the Python dictionary we have created and assigned to the `mapping` variable. Once the object is created, it's available in the `tagger` variable for future use:

```py
wordsInSentence = nltk.word_tokenize(simpleSentence)
```

Here, we are tokenizing the sentence using the NLTK built-in `word_tokenize()` function and capturing the result in the `wordsInSentence` variable:

```py
posEnabledTags = tagger.tag(wordsInSentence)
```

Once the sentence is tokenized, we call the `tag()` function of the tagger by passing the list of tokens in the `wordsInSentence` variable. The result of this computation is assigned to the `posEnabledTags` variable:

```py
print(posEnabledTags)
```

In this instruction, we are printing the data structure in `posEnabledTags` on the screen for further inspection:

```py
testSentence = "Mysore is an amazing place on earth. I have visited Mysore 10 times."
```

We are creating a variable called `testSentence` and assigning a simple English sentence to it:

```py
learnDefaultTagger(testSentence)
```

We call the `learnDefaultTagger` function created in this recipe by passing the `testSentence` as the first argument to it. Once this function execution completes, we will see the sentence POS tagged:

```py
learnRETagger(testSentence)
```

In this expression, we are invoking the `learnRETagger()` function with the same test sentence in the `testSentence` variable. The output from this function is a list of tags that are tagged as per the regular expressions that we have defined ourselves:

```py
learnLookupTagger(testSentence)
```

The output from this function `learnLookupTagger` is list of all tags from the sentence `testSentence` that are tagged using the lookup dictionary that we have created.

# Training your own tagger

In this recipe, we will learn how to train our own tagger and save the trained model to disk so that we can use it later for further computations.

# Getting ready

You should have a working Python (Python 3.6 is preferred) installed in your system, along with the NLTK library and all its collections for optimal experience.

# How to do it...

1.  Open your atom editor (or favorite programming editor).
2.  Create a new file called `Train3.py`.

3.  Type the following source code:

![](img/7fdeb930-8718-49f3-af6e-f3b66f77beee.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/e2cf88f7-b8f8-4e8e-b538-ca05208ac5f3.png)

# How it works...

Let's understand how the program works:

```py
import nltk
import pickle
```

In these two instructions, we are loading the `nltk` and `pickle` modules into the program. The `pickle` module implements powerful serialization and de-serialization algorithms to handle very complex Python objects:

```py
def sampleData():
  return [
    "Bangalore is the capital of Karnataka.",
    "Steve Jobs was the CEO of Apple.",
    "iPhone was Invented by Apple.",
    "Books can be purchased in Market.",
  ]
```

In these instructions, we are defining a function called `sampleData()` that returns a Python list. Basically, we are returning four sample strings:

```py
def buildDictionary():
  dictionary = {}
  for sent in sampleData():
    partsOfSpeechTags = nltk.pos_tag(nltk.word_tokenize(sent))
    for tag in partsOfSpeechTags:
      value = tag[0]
      pos = tag[1]
      dictionary[value] = pos
    return dictionary
```

We now define a function called `buildDictionary()`; it reads one string at a time from the list generated by the `sampleData()` function. Each string is tokenized using the `nltk.word_tokenize()` function. The resultant tokens are added to a Python dictionary, where the dictionary key is the word in the sentence and the value is POS. Once a dictionary is computed, it's returned to the caller:

```py
def saveMyTagger(tagger, fileName):
  fileHandle = open(fileName, "wb")
  pickle.dump(tagger, fileHandle)
  fileHandle.close()
```

In these instructions, we are defining a function called `saveMyTagger()` that takes two parameters: 

*   `tagger`: An object to the POS tagger
*   `fileName`: This contains the name of the file to store the `tagger` object in

We first open the file in **write binary** (**wb**) mode. Then, using `pickle` module's `dump()` method, we store the entire tagger in the file and call the `close()` function on `fileHandle`:

```py
def saveMyTraining(fileName):
  tagger = nltk.UnigramTagger(model=buildDictionary())
  saveMyTagger(tagger, fileName)
```

In these instructions, we are defining a new function called `saveMyTraining`; it takes a single argument called `fileName`.

We are building an `nltk.UnigramTagger()` object with the model as output from the `buildDictionary()` function (which itself is built from the sample set of strings that we have defined). Once the `tagger` object is created, we call the `saveMyTagger()` function to save it to disk:

```py
def loadMyTagger(fileName):
  return pickle.load(open(fileName, "rb"))
```

Here, we are defining a new function, `loadMyTagger()`, which takes `fileName` as a single argument. This function reads the file from disk and passes it to the `pickle.load()` function which unserializes the tagger from disk and returns a reference to it:

```py
sentence = 'Iphone is purchased by Steve Jobs in Bangalore Market'
fileName = "myTagger.pickle"
```

In these two instructions, we are defining two variables, `sentence` and `fileName`, which contain a sample string that we want to analyze and the file path at which we want to store the POS tagger respectively:

```py
saveMyTraining(fileName)
```

This is the instruction that actually calls the function `saveMyTraining()` with `myTagger.pickle` as argument. So, we are basically storing the trained tagger in this file:

```py
myTagger = loadMyTagger(fileName)
```

In this instruction, we take the `myTagger.pickle` as argument of the `loadMyTagger()` function, which loads the tagger from disk, deserializes it, and creates an object, which further gets assigned to the `myTagger` variable:

```py
print(myTagger.tag(nltk.word_tokenize(sentence)))
```

In this instruction, we are calling the `tag()` function of the tagger that we have just loaded from disk. We use it to tokenize the sample string that we have created.

Once the processing is done, the output is displayed on the screen.

# Learning to write your own grammar

In automata theory, CFG consists of the following things:

*   A starting symbol/ token
*   A set of symbols/ tokens that are terminals
*   A set of symbols/ tokens that are non-terminals
*   A rule (or production) that defines the start symbol/token and the possible end symbols/tokens

The symbol/token can be anything that is specific to the language that we consider.

For example:

*   In the case of the English language, *a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z* are symbols/tokens/alphabets.
*   In the case of the decimal numbering system *0, 1, 2, 3, 4, 5, 6, 7, 8, 9 *are symbols/tokens/alphabets.

Generally, rules (or productions) are written in **Backus-Naur form** (**BNF**) notation.

# Getting ready

You should have a working Python (Python 3.6 is preferred) installed on your system, along with the NLTK library.

#  How to do it...

1.  Open your atom editor (or your favorite programming editor).
2.  Create a new file called `Grammar.py`.
3.  Type the following source code:

![](img/46df3271-f331-4d11-b97e-60d2883bbf79.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/a1901496-60f9-4ce1-99b9-19fa547116b2.png)

# How it works...

Now, let's go through the program that we have just written and dig into the details:

```py
import nltk
```

We are importing the `nltk` library into the current program:

```py
import string
```

This instruction imports the `string` module into the current program:

```py
from nltk.parse.generate import generate
```

This instruction imports the `generate` function from the `nltk.parse.generate` module, which helps in generating strings from the CFG that we are going to create:

```py
productions = [
  "ROOT -> WORD",
  "WORD -> ' '",
  "WORD -> NUMBER LETTER",
  "WORD -> LETTER NUMBER",
]
```

We are defining a new grammar here. The grammar can contain the following production rules:

*   The starting symbol is `ROOT`
*   The `ROOT` symbol can produce `WORD` symbol
*   The `WORD` symbol can produce `' '` (empty space); this is a dead end production rule
*   The `WORD` symbol can produce `NUMBER` symbol followed by `LETTER` symbol
*   The `WORD` symbol can produce `LETTER` symbol followed by `NUMBER` symbol

These instructions further extend the production rules.

```py
digits = list(string.digits)
for digit in digits[:4]:
  productions.append("NUMBER -> '{w}'".format(w=digit))
```

*   `NUMBER` can produce terminal alphabets `0`, `1`, `2`, or `3`:

These instructions further extend the production rules.

```py
letters = "' | '".join(list(string.ascii_lowercase)[:4])
productions.append("LETTER -> '{w}'".format(w=letters))
```

*   `LETTER` can produce lowercase alphabets `a`, `b`, `c`, or `d`.

Let's try to understand what this grammar is for. This grammar represents the language wherein there are words such as `0a`, `1a`, `2a`, `a1`, `a3`, and so on.

All the production rules that we have stored so far in the list variable called `productions` are converted to a string:

```py
grammarString = "\n".join(productions)
```

We are creating a new grammar object using the `nltk.CFG.fromstring()` method, which takes the `grammarString` variable that we have just created:

```py
grammar = nltk.CFG.fromstring(grammarString)
```

These instructions print the first five auto generated words that are present in this language, which is defined with the grammar:

```py
for sentence in generate(grammar, n=5, depth=5):
  palindrome = "".join(sentence).replace(" ", "")
  print("Generated Word: {}, Size : {}".format(palindrome, len(palindrome)))
```

# Writing a probabilistic CFG

Probabilistic CFG is a special type of CFG in which the sum of all the probabilities for the non-terminal tokens (left-hand side) should be equal to one.

Let's write a simple example to understand more.

# Getting ready

You should have a working Python (Python 3.6 is preferred) installed on your system, along with the NLTK library.

# How to do it...

1.  Open your atom editor (or your favorite programming editor).
2.  Create a new file called `PCFG.py`.
3.  Type the following source code:

![](img/4e881933-b935-4459-8f08-a303c6d7e4a0.png)

4.  Save the file.
5.  Run the program using the Python interpreter.

6.  You will see the following output:

![](img/1720a2bb-4c3f-4c2b-9eb7-73164094b88d.png)

#  How it works...

Now, let's go through the program that we have just written and dig into the details:

```py
import nltk
```

This instruction imports the `nltk` module into our program:

```py
from nltk.parse.generate import generate
```

This instruction imports the `generate` function from the `nltk.parse.genearate` module:

```py
productions = [
  "ROOT -> WORD [1.0]",
  "WORD -> P1 [0.25]",
  "WORD -> P1 P2 [0.25]",
  "WORD -> P1 P2 P3 [0.25]",
  "WORD -> P1 P2 P3 P4 [0.25]",
  "P1 -> 'A' [1.0]",
  "P2 -> 'B' [0.5]",
  "P2 -> 'C' [0.5]",
  "P3 -> 'D' [0.3]",
  "P3 -> 'E' [0.3]",
  "P3 -> 'F' [0.4]",
  "P4 -> 'G' [0.9]",
  "P4 -> 'H' [0.1]",
]
```

Here, we are defining the grammar for our language, which goes like this:

| **Description** | **Content** |
| Starting symbol | `ROOT` |
| Non-terminals | `WORD`, `P1`, `P2`, `P3`, `P4` |
| Terminals | `'A'`, `'B'`, `'C'`, `'D'`, `'E'`, `'F'`, `'G'`, `'H'` |

Once we have identified the tokens in the grammar, let's see what the production rules look like:

*   There is a `ROOT` symbol, which is the starting symbol for this grammar
*   There is a `WORD` symbol that has a probability of `1.0`
*   There is a `WORD` symbol that can produce `P1` with a probability of `0.25`
*   There is a `WORD` symbol that can produce `P1 P2` with a probability of `0.25`
*   There is a `WORD` symbol that can produce `P1 P2 P3` with a probability of `0.25`
*   There is a `WORD` symbol that can produce `P1 P2 P3 P4` with a probability of `0.25`
*   The `P1` symbol can produce symbol `'A'` with a `1.0` probability
*   The `P2` symbol can produce symbol `'B'` with a `0.5` probability
*   The `P2` symbol can produce symbol `'C'` with a `0.5` probability
*   The `P3` symbol can produce symbol `'D'` with a `0.3` probability
*   The `P3` symbol can produce symbol `'E'` with a `0.3` probability
*   The `P3` symbol can produce symbol `'F'` with a `0.4` probability
*   The `P4` symbol can produce symbol `'G'` with a `0.9` probability
*   The `P4` symbol can produce symbol `'H'` with a `0.1` probability

If you observe carefully, the sum of all the probabilities of the non-terminal symbols is equal to `1.0`. This is a mandatory requirement for the PCFG.

We are joining the list of all the production rules into a string called the `grammarString` variable:

```py
grammarString = "\n".join(productions)
```

This instruction creates a `grammar` object using the `nltk.PCFG.fromstring` method and taking the `grammarString` as input:

```py
grammar = nltk.PCFG.fromstring(grammarString)
```

This instruction uses the Python built-in `print()` function to display the contents of the `grammar` object on screen. This will summarize the total number of tokens and production rules we have in the grammar that we have just created:

```py
print(grammar)
```

We are printing 10 strings from this grammar using the NLTK built-in function `generate` and then displaying them on screen:

```py
for sentence in generate(grammar, n=10, depth=5):
  palindrome = "".join(sentence).replace(" ", "")
  print("String : {}, Size : {}".format(palindrome, len(palindrome)))
```

# Writing a recursive CFG

Recursive CFGs are a special types of CFG where the Tokens on the left-hand side are present on the right-hand side of a production rule.

Palindromes are the best examples of recursive CFG. We can always write a recursive CFG for palindromes in a given language.

To understand more, let's consider a language system with alphabets 0 and 1; so palindromes can be expressed as follows:

*   11
*   1001
*   010010

No matter in whatever direction we read these alphabets (left to right or right to left), we always get the same value. This is the special feature of palindromes.

In this recipe, we will write grammar to represent these palindromes and generate a few palindromes using the NLTK built-in string generation libraries.

Let's write a simple example to understand more.

# Getting ready

You should have a working Python (Python 3.6 is preferred) installed on your system, along with the NLTK library.

# How to do it...

1.  Open your atom editor (or your favorite programming editor).
2.  Create a new file called `RecursiveCFG.py`.
3.  Type the following source code:

![](img/7adccfc0-5c94-41b2-9caf-5a8695e28f15.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/3237ca32-26ba-44bc-81ff-305fa771527a.png)

# How it works...

Now, let's go through the program that we have just written and dig into the details. We are importing the `nltk` library into our program for future use:

```py
import nltk
```

We are also importing the `string` library into our program for future use:

```py
import string
```

We are importing the `generate` function from the `nltk.parse.generate` module:

```py
from nltk.parse.generate import generate
```

We have created a new list data structure called `productions`, where there are two elements. Both the elements are strings that represent the two productions in our CFG:

```py
productions = [
  "ROOT -> WORD",
  "WORD -> ' '"
]
```

We are retrieving the list of decimal digits as a list in the `alphabets` variable:

```py
alphabets = list(string.digits)
```

Using the digits 0 to 9, we add more productions to our list. These are the production rules that define palindromes:

```py
for alphabet in alphabets:
  productions.append("WORD -> '{w}' WORD '{w}'".format(w=alphabet))
```

Once all the rules are generated, we concatenate them as strings to a variable, `grammarString`:

```py
grammarString = "\n".join(productions)
```

In this instruction, we are creating a new `grammar` object by passing the newly constructed `grammarString` to the NLTK built-in `nltk.CFG.fromstring` function:

```py
grammar = nltk.CFG.fromstring(grammarString)
```

In this instruction, we print the grammar that we have just created by calling the Python built-in `print()` function:

```py
print(grammar)
```

We are generating five palindromes using the `generate` function of the NLTK library and printing the same on the screen:

```py
for sentence in generate(grammar, n=5, depth=5):
  palindrome = "".join(sentence).replace(" ", "")
  print("Palindrome : {}, Size : {}".format(palindrome, len(palindrome)))
```

