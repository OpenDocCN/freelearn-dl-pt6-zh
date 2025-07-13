# Raw Text, Sourcing, and Normalization

In this chapter, we will be covering the following topics:

*   The importance of string operations
*   Getting deeper with string operations
*   Reading a PDF file in Python
*   Reading Word documents in Python
*   Taking  PDF, DOCX, and plain text files and creating a user-defined corpus from them
*   Reading contents from an RSS feed
*   HTML parsing using BeautifulSoup

# Introduction

In the previous chapter, we looked at NLTK inbuilt corpora. The corpora are very well organized and normalized for usage, but that will not always be the case when you work on your industry problems. Let alone normalization and organization, we may not even get the data we need in a uniform format. The goal of this chapter is to introduce some Python libraries that will help you extract data from binary formats: PDF and Word DOCX files. We will also look at libraries that can fetch data from web feeds such as RSS and a library that will help you parse HTML and extract the raw text out of the documents. We will also learn to extract raw text from heterogeneous sources, normalize it, and create a user-defined corpus from it.

In this chapter, you will learn seven different recipes. As the name of the chapter suggests, we will be learning to source data from PDF files, Word documents, and the Web. PDFs and Word documents are binary, and over the Web, you will get data in the form of HTML. For this reason, we will also perform normalization and raw text conversion tasks on this data.

# The importance of string operations

As an NLP expert, you are going to work on a lot of textual content. And when you are working with text, you must know string operations. We are going to start with a couple of short and crisp recipes that will help you understand the `str` class and operations with it in Python.

# Getting ready…

For this recipe, you will just need the Python interpreter and a text editor, nothing more. We will see `join`, `split`, `addition`, and `multiplication` operators and indices.

# How to do it…

1.  Create a new Python file named `StringOps1.py`.
2.  Define two objects:

```py
namesList = ['Tuffy','Ali','Nysha','Tim' ]
sentence = 'My dog sleeps on sofa'
```

The first object, `nameList`, is a list of `str` objects containing some names as implied, and the second object, `sentence`, is a sentence that is an `str` object.

3.  First, we will see the join functionality and what it does:

```py
names = ';'.join(namesList)
print(type(names), ':', names)
```

The `join()` function can be called on any `string` object. It accepts a list of `str` objects as argument and concatenates all the star objects into a single `str` object, with the calling string object's contents as the joining delimiter. It returns that object. Run these two lines and your output should look like:

```py
<class 'str'> : Tuffy;Ali;Nysha;Tim
```

4.  Next, we will check out the `split` method:

```py
wordList = sentence.split(' ')
print((type(wordList)), ':', wordList)
```

The `split` function called on a string will split its contents into multiple `str` objects, create a list of the same, and return that list. The function accepts a single `str` argument, which is used as the splitting criterion. Run the code and you will see the following output:

```py
<class 'list'> : ['My', 'dog', 'sleeps', 'on', 'sofa']
```

5.  The arithmetic operators `+` and `*` can also be used with strings. Add the following lines and see the output:

```py
additionExample = 'ganehsa' + 'ganesha' + 'ganesha'
multiplicationExample = 'ganesha' * 2
print('Text Additions :', additionExample)
print('Text Multiplication :', multiplicationExample)
```

This time we will first see the output and then discuss how it works:

```py
Text Additions: ganehsaganeshaganesha
Text Multiplication: ganeshaganesha
```

The `+` operator is known as concatenation. It produces a new string, concatenating the strings into a single `str` object. Using the `*` operator, we can multiply the strings too, as shown previously in the output. Also, please note that these operations don't add anything extra, such as insert a space between the strings.

6.  Let's look at the indices of the characters in the strings. Add the following lines of code:

```py
str = 'Python NLTK'
print(str[1])
print(str[-3])
```

First, we declare a new `string` object. Then we access the second character (`y`) in the string, which just shows that it is straightforward. Now comes the tricky part; Python allows you to use negative indexes when accessing any list object; `-1` means the last member, `-2` is the second last, and so on. For example, in the preceding `str` object, index `7` and `-4` are the same character, `N`:

```py
Output: <class 'str'> : Tuffy;Ali;Nysha;Tim
<class 'list'> : ['My', 'dog', 'sleeps', 'on', 'sofa']
Text Additions : ganehsaganeshaganesha
Text Multiplication : ganeshaganesha
y
L
```

# How it works…

We created a list of strings from a string and a string from a list of strings using the `split()` and `join()` functions, respectively. Then we saw the use of some arithmetic operators with strings. Please note that we can't use the "`-`"(negation) and the "`/`"(division) operators with strings. In the end, we saw how to access individual characters in any string, in which peculiarly, we can use negative index numbers while accessing strings.

This recipe is pretty simple and straightforward, in that the objective was to introduce some common and uncommon string operations that Python allows. Up next, we will continue where we left off and do some more string operations.

# Getting deeper with string operations

Moving ahead from the previous recipe, we will see substrings, string replacements, and how to access all the characters of a string.

Let's get started.

# How to do it…

1.  Create a new Python file named `StringOps2.py` and define the following string object `str`:

```py
str = 'NLTK Dolly Python'
```

2.  Let's access the substring that ends at the fourth character from the `str` object.

```py
print('Substring ends at:',str[:4])
```

As we know the index starts at zero, this will return the substring containing characters from zero to three. When you run, the output will be:

```py
Substring ends at: NLTK
```

3.  Now we will access the substring that starts at a certain point until the end in object `str`:

```py
print('Substring starts from:',str[11:] )
```

This tells the interpreter to return a substring of object `str` from index `11` to the end. When you run this, the following output will be visible:

```py
Substring starts from: Python
```

4.  Let's access the `Dolly` substring from the `str` object. Add the following line:

```py
print('Substring :',str[5:10])
```

The preceding syntax returns characters from index `5` to `10`, excluding the 10th character. The output is:

```py
Substring : Dolly
```

5.  Now, it's time for a fancy trick. We have already seen how negative indices work for string operations. Let's try the following and see how it works:

```py
print('Substring fancy:', str[-12:-7])
Run and check the output, it will be –
Substring fancy: Dolly
```

Exactly similar to the previous step! Go ahead and do the back calculations: `-1` as the last character, `-2` as the last but one, and so and so forth. Thus, you will get the index values.

6.  Let's check the `in` operator with `if`:

```py
if 'NLTK' in str:
  print('found NLTK')
```

Run the preceding code and check the output; it will be:

```py
found NLTK
```

As elaborate as it looks, the `in` operator simply checks whether the left-hand side string is a substring of the right-hand side string.

7.  We will use the simple `replace` function on an `str` object:

```py
replaced = str.replace('Dolly', 'Dorothy')
print('Replaced String:', replaced)
```

The `replace` function simply takes two arguments. The first is the substring that needs to be replaced and the second is the new substring that will come in place of it. It returns a new `string` object and doesn't modify the object it was called upon. Run and see the following output:

```py
Replaced String: NLTK Dorothy Python
```

8.  Last but not least, we will iterate over the `replaced` object and access every character:

```py
print('Accessing each character:')
for s in replaced:
  print(s)
```

This will print each character from the replaced object on a new line. Let's see the final output:

```py
Output: Substring ends at: NLTK
Substring starts from: Python
Substring : Dolly
Substring fancy: Dolly
found NLTK
Replaced String: NLTK Dorothy Python
Accessing each character:
N
L
T
K
D
o
r
o
t
h
y
P
y
t
h
o
n
```

# How it works…

A `string` object is nothing but a list of characters. As we saw in the first step we can access every character from the string using the `for` syntax for accessing a list. The character `:` inside square brackets for any list denotes that we want a piece of the list; `:` followed by a number means we want the sublist starting at zero and ending at the index minus 1\. Similarly, a number followed by `a :` means we want a sublist from the given number to the end.

This ends our brief journey of exploring string operations with Python. After this, we will move on to files, online resources, HTML, and more.

# Reading a PDF file in Python

We start off with a small recipe for accessing PDF files from Python. For this, you need to install the `PyPDF2` library.

# Getting ready

We assume you have `pip` installed. Then, to install the `PyPDF2` library with `pip` on Python 2 and 3, you only need to run the following command from the command line:

```py
pip install pypdf2
```

If you successfully install the library, we are ready to go ahead. Along with that, I also that request you to download some test documents that we will be using during this chapter from this link: [https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0](https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0).

# How to do it…

1.  Create a file named `pdf.py` and add the following import line to it:

```py
from PyPDF2 import PdfFileReader
```

It imports the `PdfFlleReader` class from the lib `PyPDF2`.

2.  Add this Python function in the file that is supposed to read the file and return the full text from the PDF file:

```py
def getTextPDF(pdfFileName, password = '')
```

This function accepts two arguments, the path to the PDF file you want to read and the password (if any) for the PDF file. As you can see, the `password` parameter is optional.

3.  Now let's define the function. Add the following lines under the function:

```py
pdf_file = open(pdfFileName, 'rb')
read_pdf = PdfFileReader(pdf_file)
```

The first line opens the file in read and backwards seek mode. The first line is essentially the Python open file command/function that will only open a file that is non-text in binary mode. The second line will pass this opened file to the `PdfFileReader` class, which will consume the PDF document.

4.  The next step is to decrypt password-protected files, if any:

```py
if password != '':
  read_pdf.decrypt(password)
```

If a password is provided with the function call, then we will try to decrypt the file using the same.

5.  Now we will read the text from the file:

```py
text = []
for i in range(0,read_pdf.getNumPages()-1):
  text.append(read_pdf.getPage(i).extractText())
```

We create a list of strings and append text from each page to that list of strings.

6.  Return the final output:

```py
return '\n'.join(text)
```

We return the single `string` object by joining the contents of all the string objects inside the list with a new line.

7.  Create another file named `TestPDFs.py` in the same folder as `pdf.py`, and add the following import statement:

```py
import pdf
```

8.  Now we'll just print out the text from a couple of documents, one password protected and one plain:

```py
pdfFile = 'sample-one-line.pdf'
pdfFileEncrypted = 'sample-one-line.protected.pdf'
print('PDF 1: \n',pdf.getTextPDF(pdfFile))
print('PDF 2: \n',pdf.getTextPDF(pdfFileEncrypted,'tuffy'))
```

**Output**: The first six steps of the recipe only create a Python function and no output will be generated on the console. The seventh and eighth steps will output the following:

```py
This is a sample PDF document I am using to demonstrate in the tutorial.

This is a sample PDF document

password protected.
```

# How it works…

`PyPDF2` is a pure Python library that we use to extract content from PDFs. The library has many more functionalities to crop pages, superimpose images for digital signatures, create new PDF files, and much more. However, your purpose as an NLP engineer or in any text analytics task would only be to read the contents. In step *2*, it's important to open the file in backwards seek mode since the `PyPDF2` module tries to read files from the end when loading the file contents. Also, if any PDF file is password protected and you do not decrypt it before accessing its contents, the Python interpreter will throw a `PdfReadError`.

# Reading Word documents in Python

In this recipe, we will see how to load and read Word/DOCX documents. The libraries available for reading DOCX word documents are more comprehensive, in that we can also see paragraph boundaries, text styles, and do what are called runs. We will see all of this as it can prove vital in your text analytics tasks.

If you do not have access to Microsoft Word, you can always use open source versions of Liber Office and Open Office to create and edit `.docx` files.

# Getting ready…

Assuming you already have `pip` installed on your machine, we will use pip to install a module named `python-docx`. Do not confuse this with another library named `docx`, which is a different module altogether. We will be importing the `docx` object from the `python-docx` library. The following command, when fired on your command line, will install the library:

```py
pip install python-docx
```

After having successfully installed the library, we are ready to go ahead. We will be using a test document in this recipe, and if you have already downloaded all the documents from the link shared in the first recipe in this chapter, you should have the relevant document. If not, then please download the `sample-one-line.docx` document from [https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0](https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0).

Now we are good to go.

# How to do it…

1.  Create a new Python file named `word.py` and add the following `import` line:

```py
import docx
```

Simply import the `docx` object of the `python-docx` module.

2.  Define the function `getTextWord`:

```py
def getTextWord(wordFileName):
```

The function accepts one `string` parameter, `wordFileName`, which should contain the absolute path to the Word file you are interested in reading.

3.  Initialize the `doc` object:

```py
doc = docx.Document(wordFileName)
```

The `doc` object is now loaded with the word file you want to read.

4.  We will read the text from the document loaded inside the `doc` object. Add the following lines for that:

```py
fullText = []
for para in doc.paragraphs:
  fullText.append(para.text)
```

First, we initialized a string array, `fullText`. The `for` loop reads the text from the document paragraph by paragraph and goes on appending to the list `fullText`.

5.  Now we will join all the fragments/paras in a single string object and return it as the final output of the function:

```py
return '\n'.join(fullText)
```

We joined all the constituents of the `fullText` array with the delimited `\n` and returned the resultant object. Save the file and exit.

6.  Create another file, name it `TestDocX.py`, and add the following import statements:

```py
import docx
import word
```

Simply import the `docx` library and the `word.py` that we wrote in the first five steps.

7.  Now we will read a DOCX document and print the full contents using the API we wrote on `word.py.` Write down the following two lines:

```py
docName = 'sample-one-line.docx'
print('Document in full :\n',word.getTextWord(docName))
```

Initialize the document path in the first line, and then, using the API print out the full document. When you run this part, you should get an output that looks something similar to:

```py
Document in full :
```

This is a sample PDF document with some text in bold, some in italic, and some underlined. We are also embedding a title shown as follows:

```py
This is my TITLE.
This is my third paragraph.
```

8.  As already discussed, Word/DOCX documents are a much richer source of information and the libraries will give us much more than text. Now let us look at the paragraph information. Add the following four lines of code:

```py
doc = docx.Document(docName)
print('Number of paragraphs :',len(doc.paragraphs))
print('Paragraph 2:',doc.paragraphs[1].text)
print('Paragraph 2 style:',doc.paragraphs[1].style)
```

The second line in the previous snippet gives us the number of paragraphs in the given document. The third line returns only the second paragraph from the document and the fourth line will analyze the style of the second paragraph, which is `Title` in this case. When you run, the output for these four lines will be:

```py
Number of paragraphs : 3
Paragraph 2: This is my TITLE.
Paragraph 2 style: _ParagraphStyle('Title') id: 4374023248
```

It is quite self-explanatory.

9.  Next, we will see what a run is. Add the following lines:

```py
print('Paragraph 1:',doc.paragraphs[0].text)
print('Number of runs in paragraph 1:',len(doc.paragraphs[0].runs))
for idx, run in enumerate(doc.paragraphs[0].runs):
  print('Run %s : %s' %(idx,run.text))
```

Here, we are first returning the first paragraph; next we are returning the number of runs in the paragraph. Later we are printing out every run.

10.  And now to identify the styling of each run, write the following lines of code:

```py
print('is Run 0 underlined:',doc.paragraphs[0].runs[5].underline)
print('is Run 2 bold:',doc.paragraphs[0].runs[1].bold)
print('is Run 7 italic:',doc.paragraphs[0].runs[3].italic)
```

Each line in the previous snippet is checking for underline, bold, and italic styling respectively. In the following section, we will see the final output:

```py
Output: Document in full :
This is a sample PDF document with some text in BOLD, some in ITALIC and some underlined. We are also embedding a Title down below.
This is my TITLE.
This is my third paragraph.
Number of paragraphs : 3
Paragraph 2: This is my TITLE.
Paragraph 2 style: _ParagraphStyle('Title') id: 4374023248
Paragraph 1: This is a sample PDF document with some text in BOLD, some in ITALIC and some underlined. We're also embedding a Title down below.
Number of runs in paragraph 1: 8
Run 0 : This is a sample PDF document with
Run 1 : some text in BOLD
Run 2 : ,
Run 3 : some in ITALIC
Run 4 :  and
Run 5 : some underlined.
Run 6 :  We are also embedding a Title down below
Run 7 : .
is Run 0 underlined: True
is Run 2 bold: True
is Run 7 italic: True
```

# How it works…

First, we wrote a function in the `word.py` file that will read any given DOCX file and return to us the full contents in a `string` object. The preceding output text you see is fairly self-explanatory though some things I would like to elaborate are `Paragraph` and `Run` lines. The structure of a `.docx` document is represented by three data types in the `python-docx` library. At the highest level is the `Document` object. Inside each document, we have multiple paragraphs.

Every time we see a new line or a carriage return, it signifies the start of a new paragraph. Every paragraph contains multiple `Runs` , which denotes a change in word styling. By styling, we mean the possibilities of different fonts, sizes, colors, and other styling elements such as bold, italic, underline, and so on. Each time any of these elements vary, a new run is started.

# Taking PDF, DOCX, and plain text files and creating a user-defined corpus from them

For this recipe, we are not going to use anything new in terms of libraries or concepts. We are reinvoking the concept of corpus from the first chapter. Just that we are now going to create our own corpus here instead of using what we got from the Internet.

# Getting ready

In terms of getting ready, we are going to use a few files from the Dropbox folder introduced in the first recipe of this chapter. If you've downloaded all the files from the folder, you should be good. If not, please download the following files from [https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0](https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0):

*   `sample_feed.txt`
*   `sample-pdf.pdf`
*   `sample-one-line.docx`

If you haven't followed the order of this chapter, you will have to go back and look at the first two recipes in this chapter. We are going to reuse two modules we wrote in the previous two recipes, `word.py` and `pdf.py`. This recipe is more about an application of what we did in the first two recipes and the corpus from the first chapter than introducing a new concept. Let's get on with the actual code.

# How to do it…

1.  Create a new Python file named `createCorpus.py` and add the following import lines to start off:

```py
import os
import word, pdf
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
```

We have imported the `os` library for use with file operations, the `word` and `pdf` modules we wrote in the first two recipes of this chapter, and the `PlaintextCorpusReader`, which is our final objective of this recipe.

2.  Now let's write a little function that will take as input the path of a plain text file and return the full text as a `string` object. Add the following lines:

```py
def getText(txtFileName):
  file = open(txtFileName, 'r')
  return file.read()
```

The first line defines the function and input parameter. The second line opens the given file in reading mode (the second parameter of the open function `r` denotes read mode). The third line reads the content of the file and returns it into a `string` object, all at once in a single statement.

3.  We will create the new `corpus` folder now on the disk/filesystem. Add the following three lines:

```py
newCorpusDir = 'mycorpus/'
if not os.path.isdir(newCorpusDir):
  os.mkdir(newCorpusDir)
```

The first line is a simple `string` object with the name of the new folder. The second line checks whether a directory/folder of the same name already exists on the disk. The third line instructs the `os.mkdir()` function to create the directory on the disk with the specified name. As the outcome, a new directory with the name `mycorpus` would be created in the working directory where your Python file is placed.

4.  Now we will read the three files one by one. Starting with the plain text file, add the following line:

```py
txt1 = getText('sample_feed.txt')
```

Calling the `getText()` function written earlier, it will read the `sample_feed.txt` file and return the output in the `txt1` string object.

5.  Now we will read the PDF file. Add the following line:

```py
txt2 = pdf.getTextPDF('sample-pdf.pdf')
```

Using the `pdf.py` module's `getTextPDF()` function, we are retrieving the contents of the `sample-pdf.pdf` file into the `txt2` string object.

6.  Finally, we will read the DOCX file by adding the following line:

```py
txt3 = word.getTextWord('sample-one-line.docx')
```

Using the `word.py` module's `getTextWord()` function, we are retrieving the contents of the `sample-one-line.docx` file into the `txt3` string object.

7.  The next step is to write the contents of these three string objects on the disk, in files. Write the following lines of code for that:

```py
files = [txt1,txt2,txt3]
for idx, f in enumerate(files):
  with open(newCorpusDir+str(idx)+'.txt', 'w') as fout:
    fout.write(f)
```

*   **First line**: Creates an array from the string objects so as to use it in the upcoming for loop
*   **Second line**: A `for` loop with index on the files array
*   **Third line**: This opens a new file in write mode (the `w` option in the open function call)
*   **Fourth line**: Writes the contents of the string object in the file

8.  Now we will create a `PlainTextCorpus` object from the `mycorpus` directory, where we have stored our files:

```py
newCorpus = PlaintextCorpusReader(newCorpusDir, '.*')
```

A simple one-line instruction but internally it does a lot of text processing, identifying paragraphs, sentences, words, and much more. The two parameters are the path to the corpus directory and the pattern of the filenames to consider (here we have asked the corpus reader to consider all files in the directory). We have created a user-defined corpus. As simple as that!

9.  Let us see whether the our `PlainTextCorpusReader` is loaded correctly. Add the following lines of code to test it:

```py
print(newCorpus.words())
print(newCorpus.sents(newCorpus.fileids()[1]))
print(newCorpus.paras(newCorpus.fileids()[0]))
```

The first line will print the array containing all the words in the corpus (curtailed). The second line will print the sentences in file `1.txt`. The third line will print the paragraphs in file `0.txt`:

```py
Output: ['Five', 'months', '.', 'That', "'", 's', 'how', ...]
[['A', 'generic', 'NLP'], ['(', 'Natural', 'Language', 'Processing', ')', 'toolset'], ...]
[[['Five', 'months', '.']], [['That', "'", 's', 'how', 'long', 'it', "'", 's', 'been', 'since', 'Mass', 'Effect', ':', 'Andromeda', 'launched', ',', 'and', 'that', "'", 's', 'how', 'long', 'it', 'took', 'BioWare', 'Montreal', 'to', 'admit', 'that', 'nothing', 'more', 'can', 'be', 'done', 'with', 'the', 'ailing', 'game', "'", 's', 'story', 'mode', '.'], ['Technically', ',', 'it', 'wasn', "'", 't', 'even', 'a', 'full', 'five', 'months', ',', 'as', 'Andromeda', 'launched', 'on', 'March', '21', '.']], ...]
```

# How it works…

The output is fairly straightforward and as explained in the last step of the recipe. What is peculiar is the characteristics of each of objects on show. The first line is the list of all words in the new corpus; it doesn't have anything to do with higher level structures like sentences/paragrpahs/files and so on. The second line is the list of all sentences in the file `1.txt`, of which each sentence is a list of words inside each of the sentences. The third line is a list of paragraphs, of which each paragraph object is in turn a list of sentences, of which each sentence is in turn a list of words in that sentence, all from the file `0.txt`. As you can see, a lot of structure is maintained in paragraphs and sentences.

# Read contents from an RSS feed

A **Rich Site Summary** (**RSS**) feed is a computer-readable format in which regularly changing content on the Internet is delivered. Most of the websites that provide information in this format give updates, for example, news articles, online publishing and so on. It gives the listeners access to the updated feed at regular intervals in a standardized format.

# Getting ready

The objective of this recipe is to read such an RSS feed and access content of one of the posts from that feed. For this purpose, we will be using the RSS feed of Mashable. Mashable is a digital media website, in short a tech and social media blog listing. The URL of the website's RS feed is [http://feeds.mashable.com/Mashable](http://feeds.mashable.com/Mashable).

Also, we need the `feedparser` library to be able to read an RSS feed. To install this library on your computer, simply open the terminal and run the following command:

```py
pip install feedparser
```

Armed with this module and the useful information, we can begin to write our first RSS feed reader in Python.

# How to do it…

1.  Create a new file named `rssReader.py` and add the following import:

```py
import feedparser
```

2.  Now we will load the Mashable feed into our memory. Add the following line:

```py
myFeed = feedparser.parse("http://feeds.mashable.com/Mashable")
```

The `myFeed` object contains the first page of the RSS feed of Mashable. The feed will be downloaded and parsed to fill all the appropriate fields by the `feedparser`. Each post will be part of the entry list in to the `myFeed` object.

3.  Let's check the title and count the number of posts in the current feed:

```py
print('Feed Title :', myFeed['feed']['title'])
print('Number of posts :', len(myFeed.entries))
```

In the first line, we are fetching the feed title from the `myFeed` object, and in the second line, we are counting the length of the `entries` object inside the `myFeed` object. The `entries` object is nothing but a list of all the posts from the parsed feed as mentioned previously. When you run, the output is something similar to:

```py
Feed Title: Mashable
Number of posts : 30
```

`Title` will always be Mashable, and at the time of writing this chapter, the Mashable folks were putting a maximum of 30 posts in the feed at a time.

4.  Now we will fetch the very first `post` from the entries list and print it's title on the console:

```py
post = myFeed.entries[0]
print('Post Title :',post.title)
```

In the first line, we are physically accessing the zeroth element in the entries list and loading it in the `post` object. The second line prints the title of that post. Upon running, you should get an output similar to the following:

```py
Post Title: The moon literally blocked the sun on Twitter
```

I say something similar and not exactly the same as the feed keeps updating itself.

5.  Now we will access the raw HTML content of the post and print it on the console:

```py
content = post.content[0].value
print('Raw content :\n',content)
```

First we access the content object from the post and the actual value of the same. And then we print it on the console:

```py
Output: Feed Title: Mashable
Number of posts : 30
Post Title: The moon literally blocked the sun on Twitter
Raw content :
<img alt="" src="img/https%3A%2F%2Fblueprint-api-production.s3.amazonaws.com%2Fuploads%2Fcard%2Fimage%2F569570%2F0ca3e1bf-a4a2-4af4-85f0-1bbc8587014a.jpg" /><div style="float: right; width: 50px;"><a href="http://twitter.com/share?via=Mashable&text=The+moon+literally+blocked+the+sun+on+Twitter&url=http%3A%2F%2Fmashable.com%2F2017%2F08%2F21%2Fmoon-blocks-sun-eclipse-2017-twitter%2F%3Futm_campaign%3DMash-Prod-RSS-Feedburner-All-Partial%26utm_cid%3DMash-Prod-RSS-Feedburner-All-Partial" style="margin: 10px;">
<p>The national space agency threw shade the best way it knows how: by blocking the sun. Yep, you read that right. </p>
<div><div><blockquote>
<p>HA HA HA I've blocked the Sun! Make way for the Moon<a href="https://twitter.com/hashtag/SolarEclipse2017?src=hash">#SolarEclipse2017</a> <a href="https://t.co/nZCoqBlSTe">pic.twitter.com/nZCoqBlSTe</a></p>
<p>— NASA Moon (@NASAMoon) <a href="https://twitter.com/NASAMoon/status/899681358737539073">August 21, 2017</a></p>
</blockquote></div></div>
```

# How it works…

Most of the RSS feeds you will get on the Internet will follow a chronological order, with the latest post on top. Hence, the post we accessed in the recipe will be always be the most recent post the feed is offering. The feed itself is ever-changing. So every time you run the program, the format of the output will the remain same, but the content of the post on the console may differ depending upon how fast the feed updates. Also, here we are directly displaying the raw HTML on the console and not the clean content. Up next, we are going to look at parsing HTML and getting only the information we need from a page. Again, a further addendum to this recipe could be to read any feed of your choice, store all the posts from the feed on disk, and create a plain text corpus using it. Needless to say, you can take inspiration from the previous and the next recipes.

# HTML parsing using BeautifulSoup

Most of the times when you have to deal with data on the Web, it will be in the form of HTML pages. For this purpose, we thought it is necessary to introduce you to HTML parsing in Python. There are many Python modules available to do this, but in this recipe, we will see how to parse HTML using the library `BoutifulSoup4`.

# Getting ready

The package `BeautifulSoup4` will work for Python 2 and Python 3\. We will have to download and install this package on our interpreter before we can start using it. In tune with what we have been doing throughout, we will use the pip install utility for it. Run the following command from the command line:

```py
pip install beautifulsoup4
```

Along with this module, you will also need the `sample-html.html` file from the chapter's Dropbox location. In case you haven't downloaded the files already, here's the link again:

[https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0](https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0)

# How to do it…

1.  Assuming you have already installed the required package, start with the following import statement:

```py
from bs4 import BeautifulSoup
```

We have imported the `BeautifulSoup` class from the module `bs4`, which we will be using to parse the HTML.

2.  Let's load the HTML file into the `BeautifulSoup` object:

```py
html_doc = open('sample-html.html', 'r').read()
soup = BeautifulSoup(html_doc, 'html.parser')
```

In the first line, we load the `sample-html.html` file's content into the `str` object `html_doc`. Next we create a `BeautifulSoup` object, passing to it the contents of our HTML file as the first argument and `html.parser` as the second argument. We instruct it to parse the document using the `html` parser. This will load the document into the `soup` object, parsed and ready to use.

3.  The first, simplest, and most useful task on this `soup` object will be to strip all the HTML tags and get the text content. Add the following lines of code:

```py
print('\n\nFull text HTML Stripped:')
print(soup.get_text())
```

The `get_text()` method called on the `soup` object will fetch us the HTML stripped content of the file. If you run the code written so far, you will get this output:

```py
Full text HTML Stripped:
Sample Web Page

Main heading
This is a very simple HTML document
Improve your image by including an image.
Add a link to your favorite Web site.
This is a new sentence without a paragraph break, in bold italics.
This is purely the contents of our sample HTML document without any of the HTML tags.
```

4.  Sometimes, it's not enough to have pure HTML stripped content. You may also need specific tag contents. Let's access one of the tags:

```py
print('Accessing the <title> tag :', end=' ')
print(soup.title)
```

The `soup.title` will return the first title tag it encounters in the file. Output of these lines will look like:

```py
Accessing the <title> tag : <title>Sample Web Page</title>
```

5.  Let us get only the HTML stripped text from a tag now. We will grab the text of the `<h1>` tag with the following piece of code:

```py
print('Accessing the text of <H1> tag :', end=' ')
print(soup.h1.string)
```

The command `soup.h1.string` will return the text surrounded by the first `<h1>` tag encountered. The output of this line will be:

```py
Accessing the text of <H1> tag : Main heading
```

6.  Now we will access attributes of a tag. In this case, we will access the `alt` attribute of the `img` tag; add the following lines of code:

```py
print('Accessing property of <img> tag :', end=' ')
print(soup.img['alt'])
```

Look carefully; the syntax to access attributes of a tag is different than accessing the text. When you run this piece of code, you will get this output:

```py
Accessing property of <img> tag : A Great HTML Resource
```

7.  Finally, there can be multiple occurrences of any type of tag in an HTML file. Simply using the `.` syntax will only fetch you the first instance. To fetch all instances, we use the `find_all()` functionality, shown as follows:

```py
print('\nAccessing all occurences of the <p> tag :')
for p in soup.find_all('p'):
  print(p.string)
```

The `find_all()` function called on a `BeautifulSoup` object will take as an argument the name of the tag, search through the entire HTML tree, and return all occurrences of that tag as a list. We are accessing that list in the `for` loop and printing the content/text of all the `<p>` tags in the given `BeautifulSoup` object:

```py
Output: Full text HTML Stripped:

Sample Web Page

Main heading
This is a very simple HTML document
Improve your image by including an image.

Add a link to your favorite Web site.
 This is a new sentence without a paragraph break, in bold italics.

Accessing the <title> tag : <title>Sample Web Page</title>
Accessing the text of <H1> tag : Main heading
Accessing property of <img> tag : A Great HTML Resource

Accessing all occurences of the <p> tag :
This is a very simple HTML document
Improve your image by including an image.
None
```

# How it works…

BeautifulSoup 4 is a very handy library used to parse any HTML and XML content. It supports Python's inbuilt HTML parser, but you can also use other third-party parsers with it, for example, the `lxml` parser and the pure-Python `html5lib` parser. In this recipe, we used the Python inbuilt HTML parser. The output generated is pretty much self-explanatory, and of course, the assumption is that you do know what HTML is and how to write simple HTML.

