# Regular Expressions

In this chapter, we will cover the following recipes:

*   Regular expression – learning to use *, +, and ?
*   Regular expression – learning to use $ and ^, and the non-start and non-end of a word
*   Searching multiple literal strings and substring occurrence
*   Learning to create date regex and a set of characters or ranges of character
*   Finding all five character words and making abbreviations in some sentences
*   Learning to write your own regex tokenizer
*   Learning to write your own regex stemmer

# Introduction

In the previous chapter, we saw what preprocessing tasks you would want to perform on your raw data. This chapter, immediately after, provides an excellent opportunity to introduce regular expressions. Regular expressions are one of the most simple and basic, yet most important and powerful, tools that you will learn. More commonly known as regex, they are used to match patterns in text. We will learn exactly how powerful this is in this chapter.

We do not claim that you will be an expert in writing regular expressions after this chapter and that is perhaps not the goal of this book or this chapter. The aim of this chapter is to introduce you to the concept of pattern matching as a way to do text analysis and for this, there is no better tool to start with than regex. By the time you finish the recipes, you shall feel fairly confident of performing any text match, text split, text search, or text extraction operation.

Let's look at the aforementioned recipes in detail.

# Regular expression – learning to use *, +, and ?

We start off with a recipe that will elaborate the use of the , `+`, and `?` operators in regular expressions. These short-hand operators are more commonly known as wild cards, but I prefer to call them zero or more (`*`) one or more (`+`), and zero or one (`?`) for distinction. These names are much more intuitive if you think about them.

# Getting ready

The regular expressions library is a part of the Python package and no additional packages need to be installed.

# How to do it…

1.  Create a file named `regex1.py` and add the following `import` line to it:

```py
import re
```

This imports the `re` object, which allows processing and implementation of regular expressions.

2.  Add the following Python function in the file that is supposed to apply the given patterns for matching:

```py
def text_match(text, patterns):
```

This function accepts two arguments; `text` is the input text on which the `patterns` will be applied for match.

3.  Now, let's define the function. Add the following lines under the function:

```py
if re.search(patterns,  text):
  return 'Found a match!'
else:
  return('Not matched!')
```

The `re.search()` method applies the given pattern to the `text` object and returns true or false depending on the outcome after applying the method. That is the end of our function.

4.  Let's apply the wild card patterns one by one. We start with zero or one:

```py
    print(text_match("ac", "ab?"))
    print(text_match("abc", "ab?"))
    print(text_match("abbc", "ab?"))
```

5.  Let's look at this pattern `ab?`. What this means is a followed by zero or one `b`. Let's see what the output will be when we execute these three lines:

```py
Found a match!
Found a match!
Found a match!
```

Now, all of them found a match. These patterns are trying to match a part of the input and not the entire input; hence, they find a match with all three inputs.

6.  On to the next one, zero or more! Add the following three lines:

```py
print(text_match("ac", "ab*"))
print(text_match("abc", "ab*"))
print(text_match("abbc", "ab*"))
```

7.  The same set of inputs but a different string. The pattern says, `a` followed by zero or more `b`. Let's see the output of these three lines:

```py
Found a match!
Found a match!
Found a match!
```

As you can see, all the texts find a match. As rule of thumb, whatever matches zero or one wild card will also match zero or more. The `?` wildcard is a subset of `*`.

8.  Now, the one or more wild card. Add the following lines:

```py
print(text_match("ac", "ab+"))
print(text_match("abc", "ab+"))
print(text_match("abbc", "ab+"))
```

9.  The same input! Just that the pattern contains the `+` one or more wild card. Let's see the output:

```py
Not matched!
Found a match!
Found a match!
```

As you can see, the first input string couldn't find the match. The rest did as expected.

10.  Now, being more specific in the number of repetitions, add the following line:

```py
print(text_match("abbc", "ab{2}"))
```

The pattern says `a` followed by exactly two `b`. Needless to say, the pattern will find a match in the input text.

11.  Time for a range of repetitions! Add the following line:

```py
print(text_match("aabbbbc", "ab{3,5}?"))
```

This will also be a match as we have as a substring `a` followed by four `b`.

The output of the program won't really make much sense in full. We have already `ana.ysed` the output of each and every step; hence, we won't be printing it down here again.

# How it works…

The `re.search()` function is a function that will only apply the given pattern as a test and will return true or false as the result of the test. It won't return the matching value. For that, there are other re functions that we shall learn in later recipes.

# Regular expression – learning to use $ and ^, and the non-start and non-end of a word

The starts with (^) and ends with ($) operators are indicators used to match the given patterns at the start or end of an input text.

# Getting ready

We could have reused the `text_match()` function from the previous recipe, but instead of importing an external file, we shall rewrite it. Let's look at the recipe implementation.

# How to do it…

1.  Create a file named `regex2.py` and add the following `import` line to it:

```py
import re
```

2.  Add this Python function in the file that is supposed to apply the given patterns for matching:

```py
def text_match(text, patterns):
  if re.search(patterns,  text):
    return 'Found a match!'
  else:
    return('Not matched!')
```

This function accepts two arguments; `text` is the input text on which `patterns` will be applied for matching and will return whether the match was found or not. The function is exactly what we wrote in the previous recipe.

3.  Let's apply the following pattern. We start with a simple starts with ends with:

```py
print("Pattern to test starts and ends with")
print(text_match("abbc", "^a.*c$"))
```

4.  Let's look at this pattern, `^a.*c$`. This means: start with `a`, followed by zero or more of any characters, and end with `c`. Let's see the output when we execute these three lines:

```py
Pattern to test starts and ends with

Found a match!
```

It found a match for the input text, of course. What we introduced here is a new `.` wildcard. The dot matches any character except a newline in default mode; that is, when you say `.*`, it means zero or more occurrences of any character.

5.  On to the next one, to find a pattern that looks for an input text that begins with a word. Add the following two lines:

```py
print("Begin with a word")
print(text_match("Tuffy eats pie, Loki eats peas!", "^\w+"))
```

6.  `\w` stands for any alphanumeric character and underscore. The pattern says: start with (`^`) any alphanumeric character (`\w`) and one or more occurrences of it (`+`). The output:

```py
Begin with a word

Found a match!
```

As expected, the pattern finds a match.

7.  Next, we check for an ends with a word and optional punctuation. Add the following lines:

```py
print("End with a word and optional punctuation")
print(text_match("Tuffy eats pie, Loki eats peas!", "\w+\S*?$"))
```

8.  The pattern means one or more occurrences of `\w`, followed by zero or more occurrences of `\S`, and that should be falling towards the end of the input text. To understand `\S` (capital `S`), we must first understand `\s`, which is all whitespace characters. `\S` is the reverse or the anti-set of `\s`, which when followed by `\w` translates to looking for a punctuation:

```py
End with a word and optional punctuation

Found a match!
```

We found the match with peas! at the end of the input text.

9.  Next, find a word that contains a specific character. Add the following lines:

```py
print("Finding a word which contains character, not start or end of the word")
print(text_match("Tuffy eats pie, Loki eats peas!", "\Bu\B"))
```

For decoding this pattern, `\B` is a anti-set or reverse of `\b`. The `\b` matches an empty string at the beginning or end of a word, and we have already seen what a word is. Hence, `\B` will match inside the word and it will match any word in our input string that contains character `u`:

```py
Finding a word which contains character, not start or end of the word

Found a match!
```

We find the match in the first word, `Tuffy`.

Here's the output of the program in full. We have already seen it in detail, so I will not go into it again:

```py
Pattern to test starts and ends with

Found a match!

Begin with a word

Found a match!

End with a word and optional punctuation

Found a match!

Finding a word which contains character, not start or end of the word

Found a match!
```

# How it works…

Along with starts with and ends with, we also learned the wild card character `.` and some other special sequences such as, `\w`, `\s`, `\b`, and so on. 

# Searching multiple literal strings and substring occurrences

In this recipe, we shall run some iterative functions with regular expressions. More specifically, we shall run multiple patterns on an input string with a `for` loop and we shall also run a single pattern for multiple matches on the input. Let's directly see how to do it.

# Getting ready

Open your PyCharm editor or any other Python editor that you use, and you are ready to go.

# How to do it…

1.  Create a file named `regex3.py` and add the following `import` line to it:

```py
import re
```

2.  Add the following two Python lines to declare and define our patterns and the input text:

```py
patterns = [ 'Tuffy', 'Pie', 'Loki' ]
text = 'Tuffy eats pie, Loki eats peas!'
```

3.  Let us write our first for loop. Add these lines:

```py
for pattern in patterns:
  print('Searching for "%s" in "%s" -&gt;' % (pattern, text),)
  if re.search(pattern,  text):
    print('Found!')
  else:
    print('Not Found!')
```

This is a simple for loop, iterating on the list of patterns one by one and calling the search function of `re`. Run this piece and you shall find a match for two of the three words in the input string. Also, do note that these patterns are case sensitive; the capitalized word `Tuffy`! We will discuss the output in the output section.

4.  On to the next one, to search a substring and find its location too. Let's define the pattern and the input text first:

```py
text = 'Diwali is a festival of lights, Holi is a festival of colors!'
pattern = 'festival'
```

The preceding two lines define the input text and the pattern to search for respectively.

5.  Now, the `for` loop that will iterate over the input text and fetch all occurrences of the given pattern:

```py
for match in re.finditer(pattern, text):
  s = match.start()
  e = match.end()
  print('Found "%s" at %d:%d' % (text[s:e], s, e))
```

6.  The `finditer` function takes as input the pattern and the input text on which to apply that pattern. On the returned list, we shall iterate. For every object, we will call the `start` and `end` methods to know the exact location where we found a match for the pattern. We will discuss the output of this block here. The output of this little block will look like:

```py
Found "festival" at 12:20

Found "festival" at 42:50
```

Two lines of output! Which suggests that we found the pattern at two places in the input. The first was at position `12:20` and the second was at `42:50` as displayed in the output text lines.

Here's the output of the program in full. We have already seen some parts in detail but we will go through it again:

```py
Searching for "Tuffy" in "Tuffy eats pie, Loki eats peas!" -&gt;

Found!

Searching for "Pie" in "Tuffy eats pie, Loki eats peas!" -&gt;

Not Found!

Searching for "Loki" in "Tuffy eats pie, Loki eats peas!" -&gt;

Found!

Found "festival" at 12:20

Found "festival" at 42:50
```

The output is quite intuitive, or at least the first six lines are. We searched for the word `Tuffy` and it was found. The word `Pie` wasn't found (the `re.search()` function is case sensitive) and then the word `Loki` was found. The last two lines we've already discussed, in the sixth step. We didn't just search the string but also pointed out the index where we found them in the given input.

# How it works...

Let's discuss some more things about the `re.search()` function we have used quite heavily so far. As you can see in the preceding output, the word `pie` is part of the input text but we search for the capitalized word `Pie` and we can't seem to locate it. If you add a flag in the search function call `re.IGNORECASE`, only then will it be a case-insensitive search. The syntax will be `re.search(pattern, string, flags=re.IGNORECASE)`.

Now, the `re.finditer()` function. The syntax of the function is `re.finditer(pattern, string, flags=0)`. It returns an iterator containing `MatchObject` instances over all the non-overlapping matches found the in the input string.

# Learning to create date regex and a set of characters or ranges of character

In this recipe, we shall first run a simple date regex. Along with that, we will learn the significance of the () groups. Since that's too less to include in a recipe, we shall also throw in some more things like the squared brackets [], which indicate a set (we will see in detail what a set is).

# How to do it...

1.  Create a file named `regex4.py` and add the following `import` line to it:

```py
import re
```

2.  Let's declare a `url` object and write a simple date finder regular expression to start:

```py
url= "http://www.telegraph.co.uk/formula-1/2017/10/28/mexican-grand-prix-2017-time-does-start-tv-channel-odds-lewis1/"

date_regex = '/(\d{4})/(\d{1,2})/(\d{1,2})/'
```

The `url` is a simple string object. The `date_regex` is also a simple string object but it contains a regex that will match a date with format *YYYY/DD/MM* or *YYYY/MM/DD* type of dates. `\d` denotes digits starting from 0 to 9\. We've already learned the notation {}.

3.  Let's apply `date_regex` to `url` and see the output. Add the following line:

```py
print("Date found in the URL :", re.findall(date_regex, url))
```

4.  A new `re` function, `re.findall(pattern, input, flags=0)`, which again accepts the pattern, the input text, and optionally flags (we learned case sensitive flag in the previous recipe). Let's see the output:

```py
Date found in the URL : [('2017', '10', '28')]
```

So, we've found the date 28 October 2017 in the given input string object.

5.  Now comes the next part, where we will learn about the set of characters notation `[]`. Add the following function in the code:

```py
def is_allowed_specific_char(string):
  charRe = re.compile(r'[^a-zA-Z0-9.]')
  string = charRe.search(string)
  return not bool(string)
```

The purpose here is to check whether the input string contains a specific set of characters or others. Here, we are going with a slightly different approach; first, we `re.compile` the pattern, which returns a `RegexObject`. Then, we call the `search` method of `RegexObject` on the already compiled pattern. If a match is found, the `search` method returns a `MatchObject`, and `None` otherwise. Now, turning our attention to the set notation `[]`. The pattern enclosed inside the squared brackets means: not (`^`) the range of characters `a-z`, `A-Z`, `0-9`, or `.`. Effectively, this is an OR operation of all things enclosed by the squared brackets.

6.  Now the test for the pattern. Let's call the function on two different types of inputs, one that matches and one that doesn't:

```py
print(is_allowed_specific_char("ABCDEFabcdef123450."))
print(is_allowed_specific_char("*&%@#!}{"))
```

7.  The first set of characters contains all of the allowed list of characters, whereas the second set contains all of the disallowed set of characters. As expected, the output of these two lines will be:

```py
True

False
```

The pattern will iterate through each and every character of the input string and see if there is any disallowed character, and it will flag it out. You can try adding any of the disallowed set of characters in the first call of `is_allwoed_specific_char()` and check for yourself.

Here's the output of the program in full. We have already seen it in detail, so we shall not go through it again:

```py
Date found in the URL : [('2017', '10', '28')]

True

False
```

# How it works...

Let's first discuss what a group is. A group in any regular expression is what is enclosed inside the brackets `()` inside the pattern declaration. If you see the output of the date match, you will see a set notation, inside which you have three string objects: `[('2017', '10', '28')]`. Now, look at the pattern declared carefully, `/(\d{4})/(\d{1,2})/(\d{1,2})/`. All the three components of the date are marked inside the group notation `()`, and hence all three are identified separately.

Now, the `re.findall()` method will find all the matches in the given input. This means that if there were more dates inside the give input text, the output would've looked like `[('2017', '10', '28'), ('2015', '05', '12')]`.

The `[]` notation that is set essentially means: match either of the characters enclosed inside the set notation. If any single match is found, the pattern is true.

# Find all five-character words and make abbreviations in some sentences

We have covered all the important notations that I wanted to cover with examples in the previous recipes. Now, going forward, we will look at a few small recipes that are geared more towards accomplishing a certain task using regular expressions than explaining any notations. Needless to say, we will still learn some more notations.

# How to do it…

1.  Create a file named `regex_assignment1.py` and add the following `import` line to it:

```py
import re
```

2.  Add the following two Python lines to define the input string and apply the substitution pattern for abbreviation:

```py
street = '21 Ramkrishna Road'
print(re.sub('Road', 'Rd', street))
```

3.  First, we are going to do the abbreviation, for which we use the `re.sub()` method. The pattern to look for is `Road`, the string to replace it with `Rd`, and the input is the string object `street`. Let's look at the output:

```py
21 Ramkrishna Rd
```

Clearly, it works as expected.

4.  Now, let us find all five-character words inside any given sentence. Add these two lines of code for that:

```py
text = 'Diwali is a festival of light, Holi is a festival of color!'
print(re.findall(r"\b\w{5}\b", text))
```

5.  Declare a string object `text` and put the sentence side it. Next, create a pattern and apply it using the `re.findall()` function. We are using the `\b` boundary set to identify the boundary between words and the `{}` notation to make sure we are only shortlisting five-character words. Run this and you shall see the list of words matched as expected:

```py
['light', 'color']
```

Here's the output of the program in full. We have already seen it in detail, so we will not go through it again:

```py
21 Ramkrishna Rd

['light', 'color']
```

# How it works...

By now, I assume you have a good understanding of the regular expression notations and syntax. Hence, the explanations given when we wrote the recipe are quite enough. Instead, let us look at something more interesting. Look at the `findall()` method; you will see a notation like `r&lt;pattern&gt;`. This is called the raw string notation; it helps keep the regular expression sane looking. If you don't do it, you will have to provide an escape sequence to all the backslashes in your regular expression. For example, patterns `r"\b\w{5}\b"` and `"\\b\\w{5}\\b"` do the exact same job functionality wise.

# Learning to write your own regex tokenizer

We already know the concepts of tokens, tokenizers, and why we need them from the previous chapter. We have also seen how to use the inbuilt tokenizers of the NLTK module. In this recipe, we will write our own tokenizer; it will evolve to mimic the behavior of `nltk.word_tokenize()`.

# Getting ready

If you have your Python interpreter and editor ready, you are as ready as you can ever be.

# How to do it...

1.  Create a file named `regex_tokenizer.py` and add the following `import` line to it:

```py
import re
```

2.  Let's define our raw sentence to tokenize and the first pattern:

```py
raw = "I am big! It's the pictures that got small."

print(re.split(r' +', raw))
```

3.  This pattern will perform the same as the space tokenizer we saw in previous chapter. Let's look at the output:

```py
['I', 'am', 'big!', "It's", 'the', 'pictures', 'that', 'got', 'small.']
```

As we can see, our little pattern works exactly as expected.

4.  Now, this is not enough, is it? We want to split the tokens on anything non-word and not the `' '` characters alone. Let's try the following pattern:

```py
print(re.split(r'\W+', raw))
```

5.  We are splitting on all non-word characters, that is, `\W`. Let's see the output:

```py
['I', 'am', 'big', 'It', 's', 'the', 'pictures', 'that', 'got', 'small', '']
```

We did split out on all the non-word characters (`' '`, `,`, `!`, and so on), but we seem to have removed them from the result altogether. Looks like we need to do something more and different.

6.  Split doesn't seem to be doing the job; let's try a different `re` function, `re.findall()`. Add the following line:

```py
print(re.findall(r'\w+|\S\w*', raw))
```

7.  Let's run and see the output:

```py
['I', 'am', 'big', '!', 'It', "'s", 'the', 'pictures', 'that', 'got', 'small', '.']
```

Looks like we hit the jackpot.

Here's the output of the program in full. We have already discussed it; let's print it out:

```py
['I', 'am', 'big!', "It's", 'the', 'pictures', 'that', 'got', 'small.']

['I', 'am', 'big', 'It', 's', 'the', 'pictures', 'that', 'got', 'small', '']

['I', 'am', 'big', '!', 'It', "'s", 'the', 'pictures', 'that', 'got', 'small', '.']
```

As you can see, we have gradually improved upon our pattern and approach to achieve the best possible outcome in the end.

# How it works...

We started with a simple `re.split` on space characters and improvised it using the non-word character. Finally, we changed our approach; instead of trying to split, we went about matching what we wanted by using `re.findall`, which did the job.

# Learning to write your own regex stemmer

We already know the concept of stems/lemmas, stemmer, and why we need them from the previous chapter. We have seen how to use the inbuilt porter stemmer and Lancaster stemmer of the NLTK module. In this recipe, we will write our own regular expression stemmer that will get rid of the trailing unwanted suffixes to find the correct stems.

# Getting ready

As we did in previous stemmer and lemmatizer recipes, we will need to tokenize the text before we apply the stemmer. That's exactly what we are going to do. We will reuse the final tokenizer pattern from the last recipe. If you haven't checked out the previous recipe, please do so and you are ready set to start this one.

# How to do it…

1.  Create a file named `regex_tokenizer.py` and add the following `import` line to it:

```py
import re
```

2.  We will write a function that will do the job of stemming for us. Let's first declare the syntax of the function in this step and we will define it in the next step:

```py
def stem(word):
```

This function shall accept a string object as parameter and is supposed to return a string object as the outcome. Word in stem out!

3.  Let's define the `stem()` function:

```py
splits = re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', word)
stem = splits[0][0]
return stem
```

We are applying the `re.findall()` function to the input word to return two groups as output. First is the stem and then it's any possible suffix. We return the first group as our result from the function call.

4.  Let's define our input sentence and tokenize it. Add the following lines:

```py
raw = "Keep your friends close, but your enemies closer."
tokens = re.findall(r'\w+|\S\w*', raw)
print(tokens)
```

5.  Let's run and see the output:

```py
['Keep', 'your', 'friends', 'close', ',', 'but', 'your', 'enemies', 'closer', '.']
```

Looks like we got our tokens to do stemming.

6.  Let's apply our `stem()` method to the list of tokens we just generated. Add the following `for` loop:

```py
for t in tokens:
  print("'"+stem(t)+"'")
```

We are just looping over all tokens and printing the returned stem one by one. We will see the output in the upcoming output section and discuss it there.

Let's see the output of the entire code:

```py
['Keep', 'your', 'friends', 'close', ',', 'but', 'your', 'enemies', 'closer', '.']

'Keep'

'your'

'friend'

'close'

','

'but'

'your'

'enem'

'closer'

'.'
```

Our stemmer seems to be doing a pretty decent job. However, I reckon I have passed an easy-looking sentence for the stemmer.

# How it works…

Again, we are using the `re.findall()` function to get the desired output, though you might want to look closely at the first group's regex pattern. We are using a non-greedy wildcard match (`.*?`); otherwise, it will greedily gobble up the entire word and there will be no suffixes identified. Also, the start and end of the input are mandatory to match the entire input word and split it.

