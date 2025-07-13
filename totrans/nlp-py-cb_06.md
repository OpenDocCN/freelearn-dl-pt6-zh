# Chunking, Sentence Parse, and Dependencies

In this chapter, we will perform the following recipes:

*   Using the built-in chunker
*   Writing your own simple chunker
*   Training a chunker
*   Parsing recursive descent
*   Parsing shift reduce
*   Parsing dependency grammar and projective dependency
*   Parsing a chart

# Introduction

We have learned so far that the Python NLTK can be used to do **part-of-speech** (**POS**) recognition in a given piece of text. But sometimes we are interested in finding more details about the text that we are dealing with. For example, I might be interested in finding the names of some famous personalities, places, and so on in a given text. We can maintain a very big dictionary of all these names. But in the simplest form, we can use a POS analysis to identify these patterns very easily.

Chunking is the process of extracting short phrases from text. We will leverage POS tagging algorithms to do chunking. Remember that the tokens (words) produced by chunking do not overlap.

# Using the built-in chunker

In this recipe, we will learn how to use the in-built chunker. These are the features that will be used from NLTK as part of this process:

*   Punkt tokenizer (default)
*   Averaged perception tagger (default)
*   Maxent NE chunker (default)

# Getting ready

You should have Python installed along with the `nltk` library. Prior understanding of POS tagging as explained in [Chapter 5](d81001da-02ad-4499-a128-913770e0833e.xhtml), *POS Tagging and Grammars* is good to have.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `Chunker.py`.
3.  Type the following source code:

![](img/46ace391-59f8-4f54-aa9c-1749d3465536.png)

4.  Save the file.
5.  Run the program using the Python interpreter.

6.  You will see the following output:

![](img/f0b45d44-d2cd-4a5b-bfbb-4d9dbc099f35.png)

# How it works...

Let's try to understand how the program works. This instruction imports the `nltk` module into the program:

```py
import nltk
```

This is the data that we are going to analyze as part of this recipe. We are adding this string to a variable called `text`:

```py
text = "Lalbagh Botanical Gardens is a well known botanical garden in Bengaluru, India."
```

This instruction is going to break the given text into multiple sentences. The result is a list of sentences stored in the `sentences` variable:

```py
sentences = nltk.sent_tokenize(text)
```

In this instruction, we are looping through all the sentences that we have extracted. Each sentence is stored in the sentence variable:

```py
for sentence in sentences:
```

This instruction breaks the sentence into non-overlapping words. The result is stored in a variable called `words`:

```py
words = nltk.word_tokenize(sentence)
```

In this instruction, we do POS analysis using the default tagger that is available with NLTK. Once the identification is done, the result is stored in a variable called `tags`:

```py
tags = nltk.pos_tag(words)
```

In this instruction, we call the `nltk.ne_chunk()` function, which does the chunking part for us. The result is stored in a variable called chunks. The result is actually tree-structured data that contains the paths of the tree:

```py
chunks = nltk.ne_chunk(tags)
```

This prints the chunks that are identified in the given input string. Chunks are grouped in brackets, '`(`' and '`)`', to easily distinguish them from other words that are in the input text.

```py
print(chunks)
```

# Writing your own simple chunker

In this recipe, we will write our own Regex chunker. Since we are going to use regular expressions to write this chunker, we need to understand a few differences in the way we write regular expressions for chunking.

In [Chapter 4](bdfe8ef1-c7dd-42ff-895d-84c90c5ccfb3.xhtml), *Regular Expressions*, we understood regular expressions and how to write them. For example, a regular expression of the form *[a-z, A-Z]+* matches all words in a sentence that is written in English.

We already understand that by using NLTK, we can identify the POS in their short form (tags such as `V`, `NN`, `NNP`, and so on). Can we write regular expressions using these POS?

The answer is yes. You have guessed it correctly. We can leverage POS-based regular expression writing. Since we are using POS tags to write these regular expressions, they are called tag patterns.

Just like the way we write the native alphabets (a-z) of a given natural language to match various patterns, we can also leverage POS to match words (any combinations from dictionary) according to the NLTK matched POS.

These tag patterns are one of the most powerful features of NLTK because they give us the flexibility to match the words in a sentence just by POS-based regular expressions.

In order to learn more about these, let's dig further:

```py
"Ravi is the CEO of a Company. He is very powerful public speaker also."
```

Once we identify the POS, this is how the result looks:

```py
[('Ravi', 'NNP'), ('is', 'VBZ'), ('the', 'DT'), ('CEO', 'NNP'), ('of', 'IN'), ('a', 'DT'), ('Company', 'NNP'), ('.', '.')]
[('He', 'PRP'), ('is', 'VBZ'), ('very', 'RB'), ('powerful', 'JJ'), ('public', 'JJ'), ('speaker', 'NN'), ('also', 'RB'), ('.', '.')]
```

Later, we can use this information to extract the noun phrases.

Let's pay close attention to the preceding POS output. We can make the following observations:

*   Chunks are one or more continuous `NNP`
*   Chunks are `NNP` followed by a `DT`
*   Chunks are `NP` followed by one more `JJ`

By using these three simple observations, let's write a regular expression using POS, which is called as tag phrase in the BNF form:

```py
NP -> <PRP>
NP -> <DT>*<NNP>
NP -> <JJ>*<NN>
NP -> <NNP>+
```

We are interested in extracting the following chunks from the input text:

*   `Ravi`
*   `the CEO`
*   `a company`
*   `powerful public speaker`

Let's write a simple Python program that gets the job done.

# Getting ready

You should have Python installed, along with the `nltk` library. A fair understanding of regular expressions is good to have.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `SimpleChunker.py`.
3.  Type the following source code:

![](img/17d11112-d843-46df-98d1-f3123da35353.png)

4.  Save the file.
5.  Run the program using the Python interpreter.

6.  You will see the following output:

![](img/6edc4d4c-4540-4bf8-8338-d0e8f592a15d.png)

# How it works...

Now, let's understand how the program works:

This instruction imports the `nltk` library into the current program:

```py
import nltk
```

We are declaring the `text` variable with the sentences that we want to process:

```py
text = "Ravi is the CEO of a Company. He is very powerful public speaker also."
```

In this instruction, we are writing regular expressions, which are written using POS; so they are specially called tag patterns. These tag patterns are not a randomly created ones. They are carefully crafted from the preceding example.

```py
grammar = '\n'.join([
  'NP: {<DT>*<NNP>}',
  'NP: {<JJ>*<NN>}',
  'NP: {<NNP>+}',
])
```

Let's understand these tag patterns:

*   `NP` is followed by one or more `<DT>` and then an `<NNP>`
*   `NP` is followed by one or more `<JJ>` and then an `<NN>`
*   `NP` is one more `<NNP>`

The more text we process, the more rules like this we can discover. These are specific to the language we process. So, this is a practice we should do in order to become more powerful at information extraction:

```py
sentences = nltk.sent_tokenize(text)
```

First we break the input text into sentences by using the `nltk.sent_tokenize()` function:

```py
for sentence in sentences:
```

This instruction iterates through a list of all sentences and assigns one sentence to the `sentence` variable:

```py
words = nltk.word_tokenize(sentence)
```

This instruction breaks the sentence into tokens using the `nltk.word_tokenize()` function and puts the result into the `words` variable:

```py
tags = nltk.pos_tag(words)
```

This instruction does the POS identification on the words variable (which has a list of words) and puts the result in the `tags` variable (which has each word correctly tagged with its respective POS tag):

```py
chunkparser = nltk.RegexpParser(grammar)
```

This instruction invokes the `nltk.RegexpParser` on the grammar that we have created before. The object is available in the `chunkparser` variable:

```py
result = chunkparser.parse(tags)
```

We parse the tags using the object and the result is stored in the `result` variable:

```py
print(result)
```

Now, we display the identified chunks on screen using the `print()` function. The output is a tree structure with words and their associated POS.

# Training a chunker

In this recipe, will learn the training process, training our own chunker, and evaluating it.
Before we go into training, we need to understand the type of data we are dealing with. Once we have a fair understanding of the data, we must train it according to the pieces of information we need to extract. One particular way of training the data is to use IOB tagging for the chunks that we extract from the given text.

Naturally, we find different words in a sentence. From these words, we can find POS. Later, when chunking the text, we need to further tag the words according to where they are present in the text.

Take the following example:

```py
"Bill Gates announces Satya Nadella as new CEO of Microsoft"
```

Once we've done POS tagging and hunking of the data, we will see an output similar to this one:

```py
Bill NNP B-PERSON
Gates NNP I-PERSON
announces NNS O
Satya NNP B-PERSON
Nadella NNP I-PERSON
as IN O
new JJ O
CEO NNP B-ROLE
of IN O
Microsoft NNP B-COMPANY
```

This is called the IOB format, where each line consists of three tokens separated by spaces.

| **Column** | **Description** |
| First column in IOB |  The actual word in the input sentence |
| Second column in IOB | The POS for the word |
| Third column in IOB | Chunk identifier with I (inside chunk), O (outside chunk), B (beginning word of the chunk), and the appropriate suffix to indicate the category of the word |

Let's see this in a diagram:

![](img/4b640277-64c6-4281-bcd4-f8df8a5825c8.png)

Once we have the training data in IOB format, we can further use it to extend the reach of our chunker by applying it to other datasets. Training is very expensive if we want to do it from scratch or want to identify new types of keywords from the text.

Let's try to write a simple chunker using the `regexparser` and see what types of results it gives.

# Getting ready

You should have Python installed, along with the `nltk` library.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `TrainingChunker.py`.
3.  Type the following source code:

![](img/395ef854-eaff-4e85-ad25-0101fcd8f20b.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see this output:

![](img/e8f410a5-afca-465e-abb6-b2ba89496bc2.png)

# How it works...

This instruction imports the `nltk` module into the current program:

```py
import nltk
```

This instruction imports the `conll2000` corpus into the current program:

```py
from nltk.corpus import conll2000
```

This instruction imports the `treebank` corpus into the current program:

```py
from nltk.corpus import treebank_chunk
```

We are defining a new function, `mySimpleChunker()`. We are also defining a simple tag pattern that extracts all the words that have POS of `NNP` (proper nouns). This grammar is used for our chunker to extract the named entities:

```py
def mySimpleChunker():
  grammar = 'NP: {<NNP>+}'
  return nltk.RegexpParser(grammar)
```

This is a simple chunker; it doesn't extract anything from the given text. Useful to see if the algorithm works correctly:

```py
def test_nothing(data):
  cp = nltk.RegexpParser("")
  print(cp.evaluate(data))
```

This function uses `mySimpleChunker()` on the test data and evaluates the accuracy of the data with respect to already tagged input data:

```py
def test_mysimplechunker(data):
  schunker = mySimpleChunker()
  print(schunker.evaluate(data))
```

We create a list of two datasets, one from `conll2000` and another from `treebank`:

```py
datasets = [
  conll2000.chunked_sents('test.txt', chunk_types=['NP']),
  treebank_chunk.chunked_sents()
]
```

We iterate over the two datasets and call `test_nothing()` and `test_mysimplechunker()` on the first 50-IOB tagged sentences to see what the accuracy of the chunker looks like.

```py
for dataset in datasets:
  test_nothing(dataset[:50])
  test_mysimplechunker(dataset[:50])
```

# Parsing recursive descent

Recursive descent parsers belong to the family of parsers that read the input from left to right and build the parse tree in a top-down fashion and traversing nodes in a pre-order fashion. Since the grammar itself is expressed using CFG methodology, the parsing is recursive in nature. This kind of parsing technique is used to build compilers for parsing instructions of programming languages.

In this recipe, we will explore how we can use the RD parser that comes with the NLTK library.

# Getting ready

You should have Python installed, along with the `nltk` library.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `ParsingRD.py`.
3.  Type the following source code:

![](img/b0d8fa0e-dd08-4019-9c64-7f2d13c8f1cb.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/9ef87056-47cd-4a8a-8492-83731320a330.png)

This graph is the output of the second sentence in the input as parsed by the RD parser:

![](img/b237930a-c158-4924-a936-f5d0eb839775.png)

# How it works...

Let's see how the program works. In this instruction, we are importing the `nltk` library:

```py
import nltk
```

In these instructions, we are defining a new function, `SRParserExample`; it takes a `grammar` object and `textlist` as parameters:

```py
def RDParserExample(grammar, textlist):
```

We are creating a new parser object by calling `RecursiveDescentParser` from the `nltk.parse` library. We pass grammar to this class for initialization:

```py
parser = nltk.parse.RecursiveDescentParser(grammar)
```

In these instructions, we are iterating over the list of sentences in the `textlist` variable. Each text item is tokenized using the `nltk.word_tokenize()` function and then the resultant words are passed to the `parser.parse()` function. Once the parse is complete, we display the result on the screen and also show the parse tree:

```py
for text in textlist:
  sentence = nltk.word_tokenize(text)
  for tree in parser.parse(sentence):
    print(tree)
    tree.draw()
```

We create a new `CFG` object using `grammar`:

```py
grammar = nltk.CFG.fromstring("""
S -> NP VP
NP -> NNP VBZ
VP -> IN NNP | DT NN IN NNP
NNP -> 'Tajmahal' | 'Agra' | 'Bangalore' | 'Karnataka'
VBZ -> 'is'
IN -> 'in' | 'of'
DT -> 'the'
NN -> 'capital'
""")
```

These are the two sample sentences we use to understand the parser:

```py
text = [
  "Tajmahal is in Agra",
  "Bangalore is the capital of Karnataka",
]
```

We call `RDParserExample` using the `grammar` object and the list of sample sentences.

```py
RDParserExample(grammar, text)
```

# Parsing shift-reduce

In this recipe, we will learn to use and understand shift-reduce parsing.

Shift-reduce parsers are special types of parsers that parse the input text from left to right on a single line sentences and top to bottom on multiline sentences.

For every alphabet/token in the input text, this is how parsing happens:

*   Read the first token from the input text and push it to the stack (shift operation)
*   Read the complete parse tree on the stack and see which production rule can be applied, by reading the production rule from right to left (reduce operation)
*   This process is repeated until we run out of production rules, when we accept that parsing has failed
*   This process is repeated until all of the input is consumed; we say parsing has succeeded

In the following examples, we see that only one input text is going to be parsed successfully and the other cannot be parsed.

# Getting ready

You should have Python installed, along with the `nltk` library. An understanding of writing grammars is needed.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `ParsingSR.py`.

3.  Type the following source code:

![](img/baa9e588-2950-481f-8e4d-d91c2077c7b6.png)

4.  Save the file.
5.  Run the program using the Python interpreter.

6.  You will see the following output:

![](img/9ad57c15-333e-450e-88cc-17aed8d160f9.png)

# How it works...

Let's see how the program works. In this instruction we are importing the `nltk` library:

```py
import nltk
```

In these instructions, we are defining a new function, `SRParserExample`; it takes a `grammar` object and `textlist` as parameters:

```py
def SRParserExample(grammar, textlist):
```

We are creating a new parser object by calling `ShiftReduceParser` from the `nltk.parse` library. We pass `grammar` to this class for initialization:

```py
parser = nltk.parse.ShiftReduceParser(grammar)
```

In these instructions, we are iterating over the list of sentences in the `textlist` variable. Each text item is tokenized using the `nltk.word_tokenize()` function and then the resultant words are passed to the `parser.parse()` function. Once the parse is complete, we display the result on the screen and also show the parse tree:

```py
for text in textlist:
  sentence = nltk.word_tokenize(text)
  for tree in parser.parse(sentence):
    print(tree)
    tree.draw()
```

These are the two sample sentences we are using to understand the shift-reduce parser:

```py
text = [
  "Tajmahal is in Agra",
  "Bangalore is the capital of Karnataka",
]
```

We create a new `CFG` object using the `grammar`:

```py
grammar = nltk.CFG.fromstring("""
S -> NP VP
NP -> NNP VBZ
VP -> IN NNP | DT NN IN NNP
NNP -> 'Tajmahal' | 'Agra' | 'Bangalore' | 'Karnataka'
VBZ -> 'is'
IN -> 'in' | 'of'
DT -> 'the'
NN -> 'capital'
""")
```

We call the `SRParserExample` using the `grammar` object and the list of sample sentences.

```py
SRParserExample(grammar, text)
```

# Parsing dependency grammar and projective dependency

In this recipe, we will learn how to parse dependency grammar and use it with the projective dependency parser.

Dependency grammars are based on the concept that sometimes there are direct relationships between words that form a sentence. The example in this recipe shows this clearly.

# Getting ready

You should have Python installed, along with the `nltk` library.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `ParsingDG.py`.
3.  Type the following source code:

![](img/180a1044-b8a0-4c35-b11d-011d90bd07fb.png)

4.  Save the file.
5.  Run the program using the Python interpreter.

6.  You will see the following output:

![](img/e2b13330-f4fb-4219-adc7-5561dce8dce2.png)

# How it works...

Let's see how the program works. This instruction imports the `nltk` library into the program:

```py
import nltk
```

This instruction creates a `grammar` object using the `nltk.grammar.DependencyGrammar` class. We are adding the following productions to the grammar:

```py
grammar = nltk.grammar.DependencyGrammar.fromstring("""
'savings' -> 'small'
'yield' -> 'savings'
'gains' -> 'large'
'yield' -> 'gains'
""")
```

 Let's understand more about these productions:

*   `small` related to `savings`
*   `savings` related to `yield`
*   `large` related to `gains`
*   `gains` related to `yield`

This is the sample sentence on which we are going to run the parser. It is stored in a variable called `sentence`:

```py
sentence = 'small savings yield large gains'
```

This instruction is creating a new `nltk.parse.ProjectiveDependencyParser` object using the `grammar` we have just defined:

```py
dp = nltk.parse.ProjectiveDependencyParser(grammar)
```

We are doing many things in this for loop:

```py
for t in sorted(dp.parse(sentence.split())):
  print(t)
  t.draw()
```

Preceding for loop does:

*   We are breaking the words in the sentence
*   All the list of words are fed to the `dp` object as input
*   The result from the parsed output is sorted using the `sorted()` built-in function
*   Iterate over all the tree paths and display them on screen as well as render the result in a beautiful tree form

# Parsing a chart

Chart parsers are special types of parsers which are suitable for natural languages as they have ambiguous grammars. They use dynamic programming to generate the desired results.

The good thing about dynamic programming is that, it breaks the given problem into subproblems and stores the result in a shared location, which can be further used by algorithm wherever similar subproblem occurs elsewhere. This greatly reduces the need to re-compute the same thing over and over again.

In this recipe, we will learn the chart parsing features that are provided by the NLTK library.

# Getting ready

You should have Python installed, along with the `nltk` library. An understanding of grammars is good to have.

# How to do it...

1.  Open Atom editor (or your favorite programming editor).
2.  Create a new file called `ParsingChart.py`.
3.  Type the following source code:

![](img/6b02dae4-ee2e-4108-9654-eaff227ef980.png)

4.  Save the file.
5.  Run the program using the Python interpreter.
6.  You will see the following output:

![](img/08b17e6d-bc4e-4db0-a801-e74fe4695ff6.png)

# How it works...

Let's see how the program works. This instruction imports the `CFG` module into the program:

```py
from nltk.grammar import CFG
```

This instruction imports the `ChartParser` and `BU_LC_STRATEGY` features into the program:

```py
from nltk.parse.chart import ChartParser, BU_LC_STRATEGY
```

We are creating a sample grammar for the example that we are going to use. All the producers are expressed in the BNF form:

```py
grammar = CFG.fromstring("""
S -> T1 T4
T1 -> NNP VBZ
T2 -> DT NN
T3 -> IN NNP
T4 -> T3 | T2 T3
NNP -> 'Tajmahal' | 'Agra' | 'Bangalore' | 'Karnataka'
VBZ -> 'is'
IN -> 'in' | 'of'
DT -> 'the'
NN -> 'capital'
""")
```

The grammar consists of:

*   A starting token, `S`, which produces `T1 T4`
*   Non-terminal tokens `T1`, `T2`, `T3`, and `T4`, which further produce `NNP VBZ`, `DT NN`, `IN NNP`, `T2`, or `T2 T3` respectively
*   Terminal tokens, which are words from the English dictionary

A new chart parser object is created using the grammar object `BU_LC_STRATEGY`, and we have set `trace` to `True` so that we can see how the parsing happens on the screen:

```py
cp = ChartParser(grammar, BU_LC_STRATEGY, trace=True)
```

We are going to process this sample string in this program; it is stored in a variable called `sentence`:

```py
sentence = "Bangalore is the capital of Karnataka"
```

This instruction creates a list of words from the example sentence:

```py
tokens = sentence.split()
```

This instruction takes the list of words as input and then starts the parsing. The result of the parsing is made available in the `chart` object:

```py
chart = cp.chart_parse(tokens)
```

We are acquiring all the parse trees that are available in the chart into the `parses` variable:

```py
parses = list(chart.parses(grammar.start()))
```

This instruction prints the total number of edges in the current `chart` object:

```py
print("Total Edges :", len(chart.edges()))
```

This instruction prints all the parse trees on the screen:

```py
for tree in parses: print(tree)
```

This instruction shows a nice tree view of the chart on a GUI widget.

```py
tree.draw()
```

