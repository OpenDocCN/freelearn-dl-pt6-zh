

# Feature Engineering and NLP Algorithms

Feature engineering is the most important part of developing NLP applications. Features are the input parameters for **machine learning** (**ML**) algorithms. These ML algorithms generate output based on the input features. Feature engineering is a kind of art and skill because it generates the best possible features, and choosing the best algorithm to develop NLP application requires a lot of effort and understanding about feature engineering as well as NLP and ML algorithms. In [Chapter 2](837b4260-d60f-474a-a208-68a1eaa8e1bb.xhtml), *Practical Understanding of Corpus and Dataset,* we saw how data is gathered and what the different formats of data or corpus are. In [Chapter 3](f65e61fc-1d20-434f-b606-f36cf401fc41.xhtml), *Understanding Structure of Sentences,* we touched on some of the basic but important aspects of NLP and linguistics. We will use these concepts to derive features in this chapter. In [Chapter 4](59dc896d-b0de-44c9-9cbe-18c63e510897.xhtml), *Preprocessing,* we looked preprocessing techniques. Now, we will work on the corpus that we preprocessed and will generate features from that corpus.

Refer to *Figure 5.1*, which will help you understand all the stages that we have covered so far, as well as all the focus points of this chapter:

![](img/5e23efc2-689b-4d33-a782-6761259fe6bf.png)

Figure 5.1: An overview of the features generation process

You can refer to *Figure 1.4* in [Chapter 1](238411c1-88cf-4377-a6c6-7451e0f48daa.xhtml), *Introduction*. We covered the first four stages in the preceding three chapters.

In this chapter, we will mostly focus on the practical aspect of NLP applications. We will cover the following topics:

*   What is feature engineering?
*   Understanding basic features of NLP
*   Basic statistical features of NLP

As well as this, we will explore topics such as how various tools or libraries are developed to generate features, what the various libraries that we can use are, and how you can tweak open source libraries or open source tools as and when needed.

We will also look at the challenges for each concept. Here, we will not develop tools from scratch as it is out of the scope of this book, but we will walk you through the procedure and algorithms that are used to develop the tools. So if you want to try and build customized tools, this will help you, and will give you an idea of how to approach those kind of problem statements.

# Understanding feature engineering

Before jumping into the feature generation techniques, we need to understand feature engineering and its purpose.

# What is feature engineering?

Feature engineering is the process of generating or deriving features (attributes or an individual measurable property of a phenomenon) from raw data or corpus that will help us develop NLP applications or solve NLP-related problems.

A feature can be defined as a piece of information or measurable property that is useful when building NLP applications or predicting the output of NLP applications.

We will use ML techniques to process the natural language and develop models that will give us the final output. This model is called the **machine learning model** (**ML model**). We will feed features for machine learning algorithms as input and to generate the machine learning model. After this, we will use the generated machine learning model to produce an appropriate output for an NLP application.

If you're wondering what information can be a feature, then the answer is that any attribute can be a feature as long as it is useful in order to generate a good ML model that will produce the output for NLP applications accurately and efficiently. Here, your input features are totally dependent on your dataset and the NLP application.

Features are derived using domain knowledge for NLP applications. This is the reason we have explored the basic linguistics aspect of natural language, so that we can use these concepts in feature engineering.

# What is the purpose of feature engineering?

In this section, we will look at the major features that will help us to understand feature engineering:

*   We have raw data in natural language that the computer can't understand, and algorithms don't have the ability to accept the raw natural language and generate the expected output for an NLP application. Features play an important role when you are developing NLP applications using machine learning techniques.
*   We need to generate the attributes that are representative for our corpus as well as those attributes that can be understood by machine learning algorithms. ML algorithms can understand only the language of feature for communication, and coming up with appropriate attributes or features is a big deal. This is the whole purpose of feature engineering.
*   Once we have generated the feature, we then need to feed them to the machine learning algorithm as input, and after processing these input features, we will get the ML model. This ML model will be used to predict or generate the output for new features. The ML models, accuracy and efficiency is majorly dependent on features, which is why we say that features engineering is a kind of art and skill.

# Challenges

The following are the challenges involved in feature engineering:

*   Coming up with good features is difficult and sometimes complex.
*   After generating features, we need to decide which features we should select this selection of features also plays a major role when we perform machine learning techniques on top of that. The process of selecting appropriate feature is called **feature selection**.
*   Sometimes, during the feature selection, we need to eliminate some of the less important features, and this elimination of features is also a critical part of the feature engineering.
*   Manual feature engineering is time-consuming.
*   Feature engineering requires domain expertise or, at least, basic knowledge about domains.

# Basic feature of NLP

Apart from the challenges, NLP applications heavily rely on feature that are manually crafted based on various NLP concepts. From this point onwards, we will explore the basic features that are available in the NLP world. Let's dive in!

# Parsers and parsing

By parsing sentences, you can derive some of the most important features that can be helpful for almost every NLP application.

We will explore the concept of parser and parsing. Later, we will understand **context-free grammar** (**CFG**) and **probabilistic context-free grammar** (**PCFG**). We will see how statistical parsers are developed. If you want to make your own parser, then we will explain the procedure to do so, or if you want to tweak the existing parser, then what steps you should follow. We will also do practical work using the available parser tools. We will look at the challenges later in this same section.

# Understanding the basics of parsers

Here, I'm going to explain parser in terms of the NLP domain. The parser concept is also present in other computer science domains, but let's focus on the NLP domain and start understanding parser and what it can do for us.

In the NLP domain, a parser is the program or, more specifically, tool that takes natural language in the form of a sentence or sequence of tokens. It breaks the input stream into smaller chunks. This will help us understand the syntactic role of each element present in the stream and the basic syntax-level meaning of the sentence. In NLP, a parser actually analyzes the sentences using the rules of context-free grammar or probabilistic context-free grammar. We have seen CFG in [Chapter 3](f65e61fc-1d20-434f-b606-f36cf401fc41.xhtml), *Understanding Structure of Sentences*.

A parser usually generates output in the form of a parser tree or abstract syntax tree. Let's see some of the example parser trees here. There are certain grammar rules that parsers use to generate the parse tree with single words or lexicon items.

See the grammar rules in *Figure 5.2*:

![](img/3ea61a64-940d-4a7e-9c11-0d302aec1cbc.png)

Figure 5.2: Grammar rules for parser

Let's discuss the symbols first:

*   **S** stands for sentence
*   **NP** stands for noun phrase
*   **VP** stands for verb phrase
*   **V** stands for verb
*   **N** stand for noun
*   **ART** stands for article a, an, or the

See a parse tree generated using the grammar rules in *Figure 5.3*:

![](img/6e4b587a-30e3-459f-be47-c84cc79ef02b.png)

Figure 5.3: A parse tree as per the grammar rules defined in Figure 5.2

Here in F*igure 5.3,* we converted our sentence into the parse tree format, and as you can see, each word of the sentence is expressed by the symbols of the grammar that we already defined in F*igure 5.2.*

There are two major types of parsers. We are not going to get into the technicality of each type of parser here because it's more about the compiler designing aspect. Instead, we will explore the different types of parsers, so you can get some clarity on which type of parser we generally use in NLP. Refer to *Figure 5.4*:

![](img/d32ee1e7-1041-486a-a143-46e2dced8b8e.png)

Figure 5.4: Types of parser

We will also look at the differences between top-down parsers and bottom-up parsers in the next section, as the difference is related to the process; this will be followed by each of the parsers so that we understand the difference.

Let's jump into the concept of parsing.

# Understanding the concept of parsing

First of all, let's discuss what parsing is. Let's define the term parsing. Parsing is a formal analysis or a process that uses a sentence or the stream of tokens, and with the help of defined formal grammar rules, we can understand the sentence structure and meaning. So, parsing uses each of the words in the sentence and determines its structure using a constituent structure. What is a consistent structure? A constituent structure is based on the observation of which words combine with other words to form a sensible sentence unit. So, in the English language, the subject mostly comes first in the sentence; the sentence **He is Tom** makes sense to us, whereas the sentence **is Tom he** doesn't make sense. By parsing, we actually check as well as try to obtain a sensible constituent structure. These are the following points that will explain what parser and parsing does for us:

*   The parser tool performs the process of parsing as per the grammar rules and generates a parse tree. This parse tree structure is used to verify the syntactical structure of the sentence. If a parse tree of the sentence follows the grammar rules as well as generates a meaningful sentence, then we say that the grammar as well as the sentence generated using that grammar is valid.
*   At the end of the parsing, a parse tree is generated as output that will help you to detect ambiguity in the sentence because. Ambiguous sentences often result in multiple parse trees.

Let's see the difference between a top-down parser and a bottom-up parser:

| **Top-down parsing** | **Bottom-up parsing** |
| Top-down parsing is hypothesis-driven. | Bottom-up parsing is data-driven. |
| At each stage of parsing, the parser assumes a structure and takes a word sequentially from the sentence and tests whether the taken word or token fulfills the hypothesis or not. | In this type of parsing, the first words are taken from the input string, then the parser checks whether any predefined categories are there in order to generate a valid sentence structure, and lastly, it tries to combine them into acceptable structures in the grammar. |
| It scans a sentence in a left-to-right manner. When grammar production rules derive lexical items, the parser usually checks with the input to see whether the right sentence is being derived or not. | This kind of parsing starts with the input string of terminals. This type of parsing searches for substrings of the working string because if any string or substring matches the right-hand side production rule of grammar, then it substitutes the left-hand side non-terminal for the matching right-hand side rule. |
| It includes a backtracking mechanism. When it is determined that the wrong rule has been used, it backs up and tries another rule. | It usually doesn't include a backtracking mechanism. |

You will get to know how this parser has been built in the following section.

# Developing a parser from scratch

In this section, we will try to understand the procedure of the most famous Stanford parser, and which algorithm has been used to develop the most successful statistical parser.

In order to get an idea about the final procedure, we need to first understand some building blocks and concepts. Then, we will combine all the concepts to understand the overall procedure of building a statistical parser such as the Stanford parser.

# Types of grammar

In this section, we will see two types of grammar that will help us understand the concept of how a parser works. As a prerequisite, we will simply explain them and avoid getting too deep into the subject. We will make them as simple as possible and we will explore the basic intuition of the concepts that will be used to understand the procedure to develop the parser. Here we go!

There are two types of grammar. You can refer to *Figure 5.5*:

*   Context-free grammar
*   Probabilistic context-free grammar

![](img/7bd04ffa-e515-45d9-ad77-e24594b9a306.png)

Figure 5.5: Types of grammar

# Context-free grammar

We have seen the basic concept of context-free grammar in [Chapter 3](f65e61fc-1d20-434f-b606-f36cf401fc41.xhtml), *Understanding Structure of Sentences*. We have already seen the formal definition of the CFG to find a moment and recall it. Now we will see how the rules of grammar are important when we build a parser.

CFG is also referred to as phrase structure grammar. So, CFG and phrase structure grammar are two terms but refer to one concept. Now, let's see some examples related to this type of grammar and then talk about the conventions that are followed in order to generate a more natural form of grammar rules. Refer to the grammar rules, lexicons, and sentences in *Figure 5.6*:

![](img/420f73a9-3604-49b4-a6bf-9f4ac11b87b3.png)

Figure 5.6: CFG rules, lexicon, and sentences

Here, **S** is the starting point of grammar. **NP** stands for noun phrase and **VP** stands for verb phrase. Now we will apply top-down parsing and try to generate the given sentence by starting from the rule with the right-hand side non-terminal **S** and substitute **S** with **NP** and **VP**. Now substitute **NP** with **N** and **VP** with **V** and **NP**, and then substitute **N** with people. Substitute **V** with **fish**, **NP** with **N**, and **N** with **tank**. You can see the pictorial representation of the given process in *Figure 5.7:*

![](img/22964f07-06d4-4c80-854a-804aa52f7962.png)

Figure 5.7: A parse tree representation of one of the sentences generated by the given grammar

Now try to generate a parse tree for the second sentence on your own. If you play around with this grammar rule for a while, then you will soon discover that it is a very ambiguous one. As well as this we will also discuss the more practical aspect of CFG that is used by linguists in order to derive the sentence structure. This is a more natural form of CFG and is very similar to the formal definition of CFG with one minor change, that is, we define the preterminal symbols in this grammar. If you refer to *Figure 5.7*, you will see that symbols such as **N, V** are called **preterminal symbols**. Now look at the definition of the natural form of CFG in *Figure 5.8*:

![](img/c26804c6-9b02-4957-9056-1ef8fa1843c4.png)

Figure 5.8: A formal representation of a more natural form of CFG

Here, the ***** symbol includes the existence of an empty sequence. We are starting from the **S** symbol, but in a statistical parser we add one more stage, which is TOP or ROOT. Therefore, when we generate the parse tree, the main top most node is indicated by **S.** Please refer to *Figure 5.7* for more information*.* Now we will put an extra node with the symbol ROOT or TOP before **S.**

You may have noticed one weird rule in *Figure 5.6*. **NP** can be substituted using **e**, which stands for an empty string, so let's see what the use of that empty string rule is. We will first look at see an example to get a detailed idea about this type of grammar as well as the empty string rule. We will begin with the concept of a preterminal because it may be new to you. Take a noun phrase in the English language--any phrase containing a determiner such as a, an, or the, along with the noun itself. When you substitute **NP** with the symbol **DT** and **NN**, you enter actual lexical terminals; where we substitute **NP** with **DT** and **NN**, it is called the preterminal symbol. Now let's talk about the empty string rule. We have included this rule because, in real life, you will find various instances where there are missing parts to a sentence. To handle these kinds of instances, we put this empty string rule in grammar. We will now give you an example that will help you.

We have seen the word sequence, **people fish tank**. From this, you can extract two phrases: one is **fish tank** and the second is **people fish**. In both examples, there are missing nouns. We will represent these phrases as **e fish tank** and **people fish e**. Here, **e** stands for empty string. You will notice that in the first phrase, there is a missing noun at the start of the phrase; more technically, there is a missing subject. In the second case, there is a missing noun at the end of the phrase; more technically, there is a missing object. These kinds of situations are very common when dealing with real **natural language** (**NL**) data.

There is one last thing we need to describe, which we will use in the topic on **grammar transformation**. Refer to *Figure 5.6*, where you will find the rules. Keep referring to these grammar rules as you go. The rule that has only an empty string on its right-hand side is called an **empty rule**. You can see that there are some rules that have just one symbol on their right side as well as on their left side; they are called **unary rules** because you can rewrite one category into another category, for example, **NP -> N**. There are also some other rules that have two symbols on their right side, such as **VP -> V NP**. These kinds of rules are called **binary rules.** There are also some rules that have three symbols on their right-hand side; we certain apply some techniques to get rid of the kind of rules that have more than two symbols on the right-hand side. We will look at these shortly.

Now we have looked at CFG, as well as the concepts needed to understand it. You will be able to connect those dots in the following sections. It's now time to move on to the next section, which will give you an idea about probabilistic CFG.

# Probabilistic context-free grammar

In probabilistic grammar, we add the concept of probability. Don't worry - it's one of the most simple extensions of CFG that we've seen so far. We will now look at **probabilistic context-free grammar** (**PCFG**).

Let's define PCFG formally and then explore a different aspect of it. Refer to *Figure 5.9*:

![](img/5bb01103-db44-4d79-8ea5-3076319ed1da.png)

Figure 5.9: PCFGs formal definition

Here, *T*, *N*, *S*, and *R* are similar to CFG; the only new thing here is the probability function, so let's look at that here, the probability function takes each grammar rule and gives us the probability value of each rule. This probability maps to a real number, *R*. The range for *R* is [0,1]. We are not blindly taking any probability value. We enter one constraint where we have defined that the sum of the probability for any non-terminal should add up to 1\. Let's look at an example to understand things. You can see the grammar rules with probability in *Figure 5.10*:

![](img/e01d4e08-6d4e-4144-9d03-46402eabf20c.png)

Figure 5.10: Probabilities for grammar rules

You can see the lexical grammar rules with probability in *Figure 5.11*:

![](img/6c4be3fe-94eb-4509-9a67-a2f9808fe44d.png)

Figure 5.11: Probabilities for lexical rules

As you can see, *Figure 5.10* has three NP rules, and if you look at the probability distribution, you will notice the following:

*   Its probability adds up to 1 (0.1 + 0.2 + 0.7 = 1.0)
*   It is likely that NP is further rewritten as a noun as its probability is 0.7

In the same way, you can see that the first rule at the start of the sentence has a value of 1.0 because of a certain event that occurred first. If you look carefully, you'll notice that we have removed the empty string rule to make our grammar less ambiguous.

So, how are we going to use these probability values? This question leads us to the description of calculating the probability of trees and strings.

# Calculating the probability of a tree

If we want to calculate the probability of a tree, it is quite easy because you need to multiply the probability values of lexicons and grammar rules. This will give us the probability of a tree.

Let's look at an example to understand this calculation. Here, we will take two trees and the sentence for which we have generated trees, **people fish tank with rods**.

Refer to *Figure 5.12* and *Figure 5.13* for tree structures with their respective probability values before we calculate the probability of each tree:

![](img/3640290e-0d53-48e2-81c2-8f0bc269dba3.png)

Figure 5.12: Parse Tree

If we want calculate the probability for the parse tree given in *Figure 5.12*, then the steps of obtaining the probability is given as follows. We start scanning the tree from the top, so our string point is **S**, the top most node of the parse tree. Here, the preposition modifies the verb:

*P(t1)* = 1.0 * 0.7 * 0.4 * 0.5 * 0.6 * 0.7 * 1.0 * 0.2 * 1.0 * 0.7 * 0.1 = 0.0008232

The value 0.0008232 is the probability of the tree. Now you can calculate the same for another parse tree given in *Figure 5.13*. In this parse tree, the preposition modifies the noun. Calculate the tree probability for this parse tree:

![](img/b83b9f10-0d93-4d19-8163-32e20cc1a3ef.png)

Figure 5.13: Second parse tree

If you calculate the parse tree probability, the value should be 0.00024696.

Now let's look at the calculation of the probability of string that uses the concept of the probability of a tree.

# Calculating the probability of a string

Calculating the probability of a string is more complex compared to calculating the probability of a tree. Here, we want to calculate the probability of strings of words, and for that we need to consider all the possible tree structures that generate the string for which we want to calculate the probability. We first need to consider all the trees that have the string as part of the tree and then calculate the final probability by adding the different probabilities to generate the final probability value.

Let's revisit *Figure 5.12* and *Figure 5.13*, which we used to calculate the tree probability. Now, in order to calculate the probability of the string, we need to consider both the tree and the tree probability and then add those. Calculate the probability of the string as follows:

*P(S) = P(t1) +P(t2)*

*= 0.0008232 + 0.00024696*

*= 0.00107016*

Here, *t1* tree has a high probability, so a **VP**-attached sentence structure is more likely to be generated compared to *t2*, which has **NP** attached to it. The reason behind this is that *t1* has a **VP** node with *0.4*, whereas *t2* has two nodes, **VP** with a probability of *0.6* and **NP** with a probability of *0.2* probability. When you multiply this, you will get *0.12*, which is less than *0.4*. So, the *t1* parse tree is the most likely structure.

You should now understand the different types of grammar. Now, it's time to explore the concept of grammar transformation for efficient parsing.

# Grammar transformation

Grammar transformation is a technique used to make grammar more restrictive, which makes the parsing process more efficient. We will use **Chomsky Normal Form** (**CNF**) to transform grammar rules. Let's explore CNF before looking at an example.

Let's see CNF first. It states that all rules should follow the following rules:

*X-> Y Z* or *X-> w* where *X, Y, Z* *ε N* and *w* *ε T*

The meaning of the rule is very simple. You should not have more than two non-terminals on the right-hand side of any grammar rule; you can include the rules where the right-hand side of the rule has a single terminal. To transform the existing grammar into CNF, there is a basic procedure that you can follow:

*   Empty rules and unary rules can be removed using recursive functions.
*   *N*-ary rules are divided by introducing new non-terminals in the grammar rules. This applies to rules that have more than two non-terminals on the right-hand side. When you use CNF, you can get the same string using new transform rules, but its parse structure may differ. The newly generated grammar after applying CNF is also CFG.

Let's look at the intuitive example. We take the grammar rules that we defined earlier in *Figure 5.6* and apply CNF to transform those grammar rules. Let's begin. See the following steps:

1.  We first remove the empty rules. When you have **NP** on the right-hand side, you can have two rules such as **S -> NP VP**, and when you put an empty value for **NP**, you will get **S -> VP**. By applying this method recursively, you will get rid of the empty rule in the grammar.
2.  Then, we must try to remove unary rules. So, in this case, if you try to remove the first unary rule **S -> VP**, then you need to consider all the rules that have **VP** on their left-hand side. When you do this, you need to introduce new rules because **S** will immediately go to **VP**. We will introduce the rule, **S -> V NP**. You need to keep doing this until you get rid of the unary rules. When you remove all the unary rules, such as **S -> V**, then you also need to change your lexical entries.

Refer to *Figure 5.14* for the CNF process:

![](img/e586546b-4f32-4676-9efa-b238a673835f.png)

Figure 5.14: CNF steps 1 to 5

You can see the final result of the CNF process in *Figure 5.15:*

![](img/87990088-3a73-46d7-9fc3-816c1da8070c.png)

Figure 5.15: Step 6 - Final grammar rules after applying CNF

In real life, it is not necessary to apply full CNF, and it can often be quite painful to do so. It just makes parsing more efficient and your grammar rules cleaner. In real-life applications, we keep unary rules as our grammar rules because they tell us whether a word is treated as a verb or noun, as well as the non-terminal symbol information, which means that we have the information of the POS tag.

That's enough of the boring conceptual part. Now it's time to combine all the basic concepts of parsers and parsing to learn the algorithm that is used to develop a parser.

# Developing a parser with the Cocke-Kasami-Younger Algorithm

For the English language, there are plenty of parsers that you can use, CNF if you want to build a parser for any other language, you can use the **Cocke-Kasami-Younger** (**CKY**) algorithm. Here, we will look at some information that will be useful to you in terms of making a parser. We will also look at the main logic of the CKY algorithm.

We need to look at the assumption that we are considering before we start with the algorithm. Our technical assumption is that, here, each of the parser subtrees is independent. This means that if we have a tree node NP, then we just focus on this NP node and not on the node its, derived from; each of the subtrees act independently. The CKY algorithm can give us the result in cubic time.

Now let's look at the logic of the CKY algorithm. This algorithm takes words from the sentences and tries to generate a parse tree using bottom-up parsing. Here, we will define a data structure that is called a **parse triangle** or **chart**. Refer to *Figure 5.16*:

![](img/59c8c4e8-4528-4ba0-ba97-e1933d264000.png)

Figure 5.16: Parse triangle for the CKY algorithm

Its bottom cells represent single words such as **fish**, **people**, **fish**, and **tanks**. The cells in the middle row represent the overlapped word pairs such as **Fish people**, **People fish**, and **fish tanks**. Its third row represents the pair of two words without overlapping such as **Fish people** and **fish tanks**. The last row represents the top or root of the sentence. To understand the algorithm, we first need the grammar rules of rule probability. To understand the algorithm, we should refer to *Figure 5.17*:

![](img/7e868283-8d6c-44ab-b678-3576643852a0.png)

Figure 5.17: To understand the CKY algorithm (Image credit: http://spark-public.s3.amazonaws.com/nlp/slides/Parsing-Probabilistic.pdf page no: 36)

As shown in *Figure 5.17*, to explain the algorithm logic, we have entered the basic probability values in the bottom-most cells. Here, we need to find all the combinations that the fulfill grammar rules. Follow the given steps:

1.  We will first take **NP** from the **people** cell and **VP** from the **fish** cell. In the grammar rules, check whether there is any grammar rule present that takes the sequence, **NP VP**, which you will need to find on the right-hand side of the grammar rule. Here, we found that the rule is **S -> NP VP** with a probability of 0.9.
2.  Now, calculate the probability value, and to find this, you need to multiply the probability value of **NP** given in the people cell, the probability value of **VP** in the **fish** cell, and the probability value of the grammar rule itself. So here, the probability value of **NP** placed in the **people** cell *=* 0.35, the probability value of **VP** placed in the **fish** cell *=0.06* and the probability of the grammar rule **S -> NP VP** *=* 0.9.
3.  We then multiply 0.35 (the probability of **NP** placed in the **people** cell) * 0.06 (the probability of **VP** in the **fish** cell ) * 0.9 (the probability of the grammar rule **S -> NP VP**). Therefore, the final multiplication value *= 0.35 * 0.06 * 0.9 = 0.0189*. *0.0189* is the final probability for the grammar rule if we expand **S** into the **NP VP** grammar rule.

4.  In the same way, you can calculate other combinations, such as **NP** from the **people** cell and **NP** from the **fish** cell, and find the grammar rule, that is, **NP NP** on the right-hand side. Here, the **NP - NP NP** rule exists. So we calculate the probability value, *0.35 * 0.14 * 0.2 = 0.0098*. We continue with this process until we generate the probability value for all the combinations, and then we will see for which combination we have generated the maximum probability. The process of finding the maximum probability is called **Viterbi max score**.
5.  For the combination **S -> NP VP**, we will get the maximum probability when the cells generate the left-hand side non-terminal on its upward cell. So, those two cells generate **S**, which is a sentence.

This is the core logic of the CKY algorithm. Let's look at one concrete example for this concept. For writing purposes, we will rotate the parse triangle 90 degrees clockwise. Refer to *Figure 5.18*:

![](img/e6680024-6a6e-416e-889e-f6de607e9c37.png)

Figure 5.18: Step 1 for the CKY algorithm (Image credit: http://spark-public.s3.amazonaws.com)

Here, *cell (0,1)* is for **fish** and it fills using lexical rules. We have put **N** -> **fish** with *0.2* probability because this is defined in our grammar rule. We have put **V -> fish** with *0.6* probability. Now we focus on some unary rules that have **N** or **V** only on the right-hand side. We have the rules that we need to calculate the probability by considering the grammar rules probability and lexical probability. So, for rule **NP -> N**, the probability is *0.7* and **N -> fish** has the probability value *0.2*. We need to multiply this value and generate the probability of the grammar rule **NP -> N** *= 0.14*. In the same way, we generate the probability for the rule **VP -> V**, and its value is *0.1 * 0.6 = 0.6*. This way, you need to fill up all the four cells.

In the next stage, we follow the same procedure to get the probability for each combination generated from the grammar rules. Refer to *Figure 5.19*:

![](img/04383ecf-5951-4e09-abc1-e8f34884570c.png)

Figure 5.19 : Stage 2 of the CKY algorithm (Image credit: http://spark-public.s3.amazonaws.com)

In *Figure 5.20*, you can see the final probability values, using which you can decide the best parse tree for the given data:

![](img/d12f49a6-e874-475a-b56b-5d6fd3063327.png)

Figure 5.20: The final stage of the CKY algorithm (Image credit: http://spark-public.s3.amazonaws.com)

Now you know how the parse tree has been generated we want to share with you some important facts regarding the Stanford parser. It is built based on this CKY algorithm. There are a couple of technical assumptions and improvisations applied to the Stanford parser, but the following are the core techniques used to build the parser.

# Developing parsers step-by-step

Here, we will look at the steps required to build your own parser with the help of the CKY algorithm. Let's begin summarizing:

1.  You should have tagged the corpus that has a human-annotated parse tree: if it is tagged as per the Penn Treebank annotation format, then you are good to go.
2.  With this tagged parse corpus, you can derive the grammar rules and generate the probability for each of the grammar rules.
3.  You should apply CNF for grammar transformation.

4.  Use the grammar rules with probability and apply them to the large corpus; use the CKY algorithm with the Viterbi max score to get the most likely parse structure. If you are providing a large amount of data, then you can use the ML learning technique and tackle this problem as a multiclass classifier problem. The last stage is where you get the best parse tree for the given data as per the probability value.

That's enough theory; let's now use some of the famous existing parser tools practically and also check what kind of features you can generate from the parse tree.

# Existing parser tools

In this section, we will look at some of the existing parsers and how you can generate some cool features that can be used for ML algorithms or in rule-based systems.

Here, we will see two parsers:

*   The Stanford parser
*   The spaCy parser

# The Stanford parser

Let's begin with the Stanford parser. You can download it from [https://stanfordnlp.github.io/CoreNLP/](https://stanfordnlp.github.io/CoreNLP/). After downloading it, you just need to extract it to any location you like. The prerequisite of running the Stanford parser is that you should have a Java-run environment installed in your system. Now you need to execute the following command in order to start the Stanford parser service:

```py
    $ cd stanford-corenlp-full-2016-10-31/
    $ java -mx4g -cp "*"        edu.stanford.nlp.pipeline.StanfordCoreNLPServer

```

Here, you can change the memory from `-mx4g` to `-mx3g`.

Let's look at the concept of dependency in the parser before can fully concentrating on the coding part.

The dependency structure in the parser shows which words depend on other words in the sentence. In the sentence, some words modify the meaning of other words; on the other hand, some act as an argument for other words. All of these kinds of relationships are described using dependencies. There are several dependencies in the Stanford parser. We will go through some of them. Let's take an example and we will explain things as we go.

The sentence is: The boy put the tortoise on the rug.

For this given sentence, the **head** of the sentence is *put* and it modifies three sections: *boy*, *tortoise*, and *on the rug*. How do you find the head word of the sentence?Find out by asking the following questions:

*   Who put it down? You get the answer: *boy*
*   Then, what thing did he put down? You get the answer: *tortoise*
*   Where is it put? You get the answer: *on the rug*

So, the word **put** modifies three things. Now look at the word *boy*, and check whether there is any modifier for it. Yes, it has a modifier: **the**. Then, check whether there is any modifier for *tortoise*. Yes, it has a modifier: **the**. For the phrase *on the rug* *on* complements *rug* and *rug* acts as the head for this phrase, taking a modifier, *the*. Refer to *Figure 5.21*:

![](img/5bf923f7-b71b-4d89-a5af-6bc2a28137eb.png)

Figure 5.21: Dependency structure of the sentence

The Stanford parser has dependencies like `nsubjpass` (passive nominal subject) , `auxpass` (passive auxiliary) , `prep` (prepositional modifier), `pobj` (object of preposition), `conj` (conjunct), and so on. We won't go into more detail regarding this, but is worth mentioning that dependency parsing also follows the tree structure and it's linked by binary asymmetric relations called **dependencies**. You can find more details about each of the dependencies by accessing the Stanford parser document here: [https://nlp.stanford.edu/software/dependencies_manual.pdf](https://nlp.stanford.edu/software/dependencies_manual.pdf).

You can see the basic example in *Figure 5.22*:

![](img/026b30ae-54c5-46fc-bae7-dd3a088116a5.png)

Figure 5.22: Dependency parsing of the sentence

Now if you want to use the Stanford parser in Python, you have to use the dependency named `pycorenlp`. We will use it to generate the output from the Stanford parser.

You can see the sample code in which we have used the Stanford parser to parse the sentence. You can parse multiple sentences as well. You can find the code at the following GitHub link:

[https://github.com/jalajthanaki/NLPython/tree/master/ch5/parserexample](https://github.com/jalajthanaki/NLPython/tree/master/ch5/parserexample).

You can see the code snippet in *Figure 5.23*:

![](img/157cb610-6435-4dea-a3ea-32aa9fd8e12c.png)

Figure 5.23: Code snippet for the Stanford parser demo

You can see the output of this code in *Figure 5.24*:

![](img/8e8ae2b9-6d21-4567-8d1b-284e810a0ccc.png)

Figure 5.24: Output of the Stanford parser

# The spaCy parser

This parser helps you generate the parsing for the sentence. This is a dependency parser. You can find the code at the following GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch5/parserexample/scpacyparserdemo.py](https://github.com/jalajthanaki/NLPython/blob/master/ch5/parserexample/scpacyparserdemo.py).

You can find the code snippet of this parser in *Figure 5.25*:

![](img/8e0fcafd-dc68-49c7-82d9-a91fa5011eeb.png)

Figure 5.25: spaCy dependency parser code

You can see the output of the spaCy parser in *Figure 5.26*:

![](img/8589e6a8-17ff-4f81-a2a4-1085618b98d7.png)

Figure 5.26: The spaCy parser output

People used the Stanford parser because it provides good accuracy as well as a lot of flexibility in terms of generating output. Using the Stanford parser, you can generate the output in a JSON format, XML format, or text format. You may think that we get the parse tree using the preceding code, but the kind of features that we can derive from the parsing result will be discussed in the next section.

# Extracting and understanding the features

Generally, using the parse result, you can derive many features such as generating noun phrases and POS tags inside the noun phrase; you can also derive the head word from phrases. You can use each word and its tag. You can use the dependency relationships as features. You can see the code snippet in *Figure 5.27*:

![](img/2082d31e-85fd-4bd0-a6b0-5f78955069cf.png)

Figure 5.27: Code to get NP from sentences

The output snippet is in *Figure 5.28*:

![](img/68cef596-0bdf-4b4c-a534-8579fd709b34.png)

Figure 5.28: Output all NP from sentences

You can generate the stem as well as lemma from each word, as we saw in [Chapter 3](f65e61fc-1d20-434f-b606-f36cf401fc41.xhtml), *Understanding Structure of Sentences*.

In real life, you can generate the features easily using libraries, but which features you need to use is critical and depends on your NLP application. Let's assume that you are making a grammar correction system; in that case, you need to consider all the phrases of the sentence as well as the POS tags of each word present in the phrase. If you are developing a question-answer system, then noun phases and verb phrases are the important features that you can select.

Features selection is a bit tricky and you will need to do some iteration to get an idea of which features are good for your NLP application. Try to dump your features in a `.csv` file so you can use the `.csv` file later on in your processing. Each and every feature can be a single column of the `.csv` file. For example, you have NP words stored in one column, lemma in the other column for all words in NP, and so on. Now, suppose you have more than 100 columns; in that case, you would need to find out which are the important columns (features) and which are not. Based on the problem statement and features, you can decide what the most important features are that help us to solve our problem. In [Chapter 8](97808151-90d2-4034-8d53-b94123154265.xhtml), *Machine Learning for NLP Problems,* we will look at features selection in more detail.

# Customizing parser tools

In real life, datasets are quite complex and messy. In that case, it may be that the parser is unable to give you a perfect or accurate result. Let's take an example.

Let's assume that you want to parse a dataset that has text content of research papers, and these research papers belong to the chemistry domain. If you are using the Stanford parser in order to generate a parse tree for this dataset, then sentences that contain chemical symbols and equations may not get parsed properly. This is because the Stanford parser has been trained on the Penn TreeBank corpus, so it's accuracy for the generation of a parse tree for chemical symbols and equation is low. In this case, you have two options - either you search for a parser that can generate parsing for symbols and equations accurately or if you have a corpus that has been tagged, you have the flexibility of retraining the Stanford parser using your tagged data.

You can follow the same tagging notation given in the Penn TreeBank data for your dataset, then use the following command to retrain the Stanford parser on your dataset, save the trained model, and use it later on. You can use the following command to retrain the Stanford Parser:

```py
$ java -mx1500m -cp "stanford-parser.jar"  edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences newline -tokenized -tagSeparator / -outputFormat "penn" englishPCFG.ser.gz /home/xyz/PROJECT/COMPARING_PARSER_NOTES/data/483_18.taggedsents > /home/xyz/PROJECT/COMPARING_PARSER_NOTES/data/483_18.stanford.parsed 

```

# Challenges

Here are some of the challenges related to parsers:

*   To generate a parser for languages such as Hebrew, Gujarati, and so on is difficult and the reason is that we don't have a tagged corpus.
*   Developing parsers for fusion languages is difficult. A fusion language means that you are using another language alongside the English language, with a sentence including more than one language. Processing these kinds of sentences is difficult.

Now that we have understood some features of the parser, we can move on to our next concept, which is POS tagging. This is one of the essential concepts of NLP.

# POS tagging and POS taggers

In this section, we will discuss the long-awaited topic of POS tags.

# Understanding the concept of POS tagging and POS taggers

POS tagging is defined as the process of marking words in the corpus corresponding to their particular part of speech. The POS of the word is dependent on both its definition and its context. It is also called **grammatical tagging** or **word-category disambiguation**. POS tags of words are also dependent on their relationship with adjacent and related words in the given phrase, sentence, and paragraph.

POS tagger is the tool that is used to assign POS tags for the given data. Assigning the POS tags is not an easy task because POS of words is changed as per the sentence structure and meaning. Let's take an example. Let's take the word dogs; generally, we all know that dogs is a plural noun, but in some sentences it acts as a verb. See the sentence: **The sailor dogs the hatch**. Here, the correct POS for *dogs* is the verb and not the plural noun. Generally, many POS taggers use the POS tags that are generated by the University of Pennsylvania. You can find word-level POS tags and definitions at the following link:

[https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).

We will now touch on some of the POS tags. There are 36 POS tags in the Penn Treebank POS list, such as **NN** indicates noun, **DT** for determiner words, and **FW** for foreign words. The word that is new to POS tags is generally assigned the **FW** tag. Latin names and symbols often get the **FW** tag by POS tagger. So, if you have a (lambda) symbol then POS may tagger suggest the **FW** POS tag for it. You can see some word-level POS tags in *Figure 5.29*:

![](img/8084df2c-6e1f-43d9-a1ef-b21563bff63f.png)

Figure 5.29: Some word-level POS tags

There are POS tags available at the phrase-level as well as the clause-level. All of these tags can be found at the following GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch5/POStagdemo/POS_tags.txt](https://github.com/jalajthanaki/NLPython/blob/master/ch5/POStagdemo/POS_tags.txt).

See each of the tags given in the file that we have specified as they are really useful when you evaluating your parse tree result. POS tags and their definitions are very straightforward, so if you know basic English grammar, then you can easily understand them.

You must be curious to know how POS taggers are built. Let's find out the procedure of making your own POS tagger.

# Developing POS taggers step-by-step

To build your own POS tagger, you need to perform the following steps:

1.  You need a tagged corpus.
2.  Select features.
3.  Perform training using a decision tree classifier available in the Python library, `scikit-learn`.
4.  Check your accuracy.
5.  Try to predict the POS tags using your own trained model.

A great part of this section is that we will code our own POS tagger in Python, so you guys will get an idea of how each of the preceding stages are performed in reality. If you don't know what a decision tree algorithm is, do not worry - we will cover this topic in more detail in [Chapter 8](97808151-90d2-4034-8d53-b94123154265.xhtml), *Machine Learning for NLP Applications*.

Here, we will see a practical example that will help you understand the process of developing POS taggers. You can find the code snippet for each stage and you can access the code at the following GitHub link:

[https://github.com/jalajthanaki/NLPython/tree/master/ch5/CustomPOStagger](https://github.com/jalajthanaki/NLPython/tree/master/ch5/CustomPOStagger).

See the code snippet of getting the Pann TreeBank corpus in *Figure 5.30*:

![](img/259c9478-af35-4f67-91e1-b4b747bdc541.png)

Figure 5.30: Load the Penn TreeBank data from NLTK

You can see the feature selection code snippet in *Figure 5.31*:

![](img/8342709c-f9ea-4b81-9b00-5564421c08ad.png)

Figure 5.31: Extracting features of each word

We have to extract the features for each word. You can see the code snippet for some basic transformation such as splitting the dataset into training and testing. Refer to *Figure 5.32*:

![](img/4c8cb7a1-beac-41ae-bd89-e4d16ae656f0.png)

Figure 5.32: Splitting the data into training and testing

See the code to train the model using a decision tree algorithm in *Figure 5.33*:

![](img/a2d9f363-6bb4-43fd-acd6-ca74c40c98c2.png)

Figure 5.33: Actual training using a decision tree algorithm

See the output prediction of POS tags for the sentence that you provided in *Figure 5.34*:

![](img/6956da11-f9bd-4be9-b762-b947e839c581.png)

Figure 5.34: Output of a custom POS tagger

You should have now understood the practical aspect of making your own POS tags, but you can still also use some of the cool POS taggers.

# Plug and play with existing POS taggers

There are many POS taggers available nowadays. Here, we will use the POS tagger available in the Stanford CoreNLP and polyglot library. There are others such as the Tree tagger; NLTK also has a POS tagger that you can use. You can find the code at the following GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch5/POStagdemo](https://github.com/jalajthanaki/NLPython/blob/master/ch5/POStagdemo/).

# A Stanford POS tagger example

You can see the code snippet for the Stanford POS tagger in *Figure 5.35*:

![](img/6133b79c-6dd9-463d-ae4d-05bcb4a45de9.png)

Figure 5.35: Stanford POS tagger code

The output from the Stanford POS tagger can be found in *Figure 5.36*:

![](img/98fd3b01-e8cd-457a-ab3a-7143a5a0c890.png)

Figure 5.36: POS tags generated by the Stanford POS tagger

# Using polyglot to generate POS tagging

You can see the code snippet for the `polyglot` POS tagger in *Figure 5.37*:

![](img/71fdba9e-3649-476b-8b26-6aad00d4a5b4.png)

Figure 5.37: Polyglot POS tagger

The output from the `polyglot` POS tagger can be found in *Figure 5.38*:

![](img/b28aea32-579b-4a97-ab1d-38b5e61215f0.png)

Figure 5.38:The polyglot POS tagger output

# Exercise

Try using the TreeTagger library to generate POS tagging. You can find the installation details at this link:

[http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/.](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)

# Using POS tags as features

Now that we have generated POS tags for our text data using the POS tagger, where can we use them? We will now look at NLP applications that can use these POS tags as features.

POS tags are really important when you are building a chatbot with machine learning algorithms. POS tag sequences are quite useful when a machine has to understand various sentence structures. It is also useful if you are building a system that identifies **multiword express** (**MWE**). Some examples of MWE phrases are be able to, a little bit about, you know what, and so on.

If you have a sentence: **He backed off from the tour plan of Paris**. Here, *backed off* is the MWE. To identify these kinds of MWEs in sentences, you can use POS tags and POS tag sequences as features. You can use a POS tag in sentiment analysis, and there are other applications as well.

# Challenges

The following are some challenges for POS tags:

*   Identifying the right POS tag for certain words in an ambiguous syntax structure is difficult, and if the word carries a very different contextual meaning, then the POS tagger may generate the wrong POS tags.
*   Developing a POS tagger for Indian languages is a bit difficult because, for some languages, you cannot find the tagged dataset.

Now let's move on to the next section, where we will learn how to find the different entities in sentences.

# Name entity recognition

In this section, we will look at a tool called **name entity recognition** (**NER**). The use of this tool is as follows. If you have a sentence, such as **Bank of America announced its earning today**, we as humans can understand that the *Bank of America* is the name of a financial institution and should be referred to as a single entity. However, for machine to handle and recognize that entity is quite challenging. There is where NER tools come into the picture to rescue us.

With the NER tool, you can find out entities like person name, organization name, location, and so on. NER tools have certain classes in which they classify the entities. Here, we are considering the words of the sentence to find out the entities, and if there are any entities present in the sentence. Let's get some more details about what kind of entities we can find in our sentence using some of the available NER tools.

# Classes of NER

NER tools generally segregate the entities into some predefined classes. Different NER tools have different types of classes. The Stanford NER tool has three different versions based on the NER classes:

*   The first version is the three-class NER tool that can identify the entities - whether it's Location, Person, or Organization.
*   The second version is the four-class NER tool that can identify the Location, person, Organization, and Misc. Misc is referred to as a miscellaneous entity type. If an entity doesn't belong to Location, Person, or Organization and is still an entity, then you can tag it as Misc.
*   The third version is a seven-class tool that can identify Person, Location, Organization, Money, Percent, Date, and Time.

The spaCy parser also has an NER package available with the following classes.

*   `PERSON` class identifies the name of a person
*   `NORP` class meaning Nationality, Religious or Political groups
*   `FACILITY` class including buildings, airports, highways, and so on
*   `ORG` class for organization, institution and so on
*   `GPE` class for cities, countries and so on
*   `LOC` class for non-GPE locations such as mountain ranges and bodies of water
*   `PRODUCT` that includes objects, vehicles, food, and so on, but not services
*   `EVENT` class for sports events, wars, named hurricanes, and so on
*   `WORK_OF_ART` class for titles of books, songs, and so on
*   `LANGUAGE` that tags any named language
*   Apart from this, spaCy's NER package has classes such as date, time, percent, money, quantity, ordinal, and cardinal

Now it's time to do some practical work. We will use the Stanford NER tool and spaCy NER in our next section.

# Plug and play with existing NER tools

In this section, we will look at the coding part as well as information on how to practically use these NER tools. We will begin with the Stanford NER tool and then the Spacy NER. You can find the code at the following GitHub link:

[https://github.com/jalajthanaki/NLPython/tree/master/ch5/NERtooldemo](https://github.com/jalajthanaki/NLPython/tree/master/ch5/NERtooldemo).

# A Stanford NER example

You can find the code and output snippet as follows. You need to download the Stanford NER tool at [https://nlp.stanford.edu/software/CRF-NER.shtml#Download](https://nlp.stanford.edu/software/CRF-NER.shtml#Download).

You can see the code snippet in *Figure 5.39*:

![](img/0892d055-ab16-411e-8557-d21b1114c73f.png)

Figure 5.39: Stanford NER tool code

You can see the output snippet in *Figure 5.40*:

![](img/427a4a7c-7cfc-4f25-bc2f-c0267a910e83.png)

Figure 5.40: Output of Stanford NER

# A Spacy NER example

You can find the code and output snippet as follows. You can see the code snippet in *Figure 5.41*:

![](img/58328dac-aebe-4c68-9d52-1abe10e77d4a.png)

Figure 5.41: spaCy NER tool code snippet

You can see the output snippet in *Figure 5.42*:

![](img/fe8df511-f2dd-474a-9913-bf0535b1b44c.png)

Figure 5.42: Output of the spaCy tool

# Extracting and understanding the features

NER tags are really important because they help you to understand sentence structure and help machines or NLP systems to understand the meaning of certain words in a sentence.

Let's take an example. If you are building a proofreading tool, then this NER tool is very useful because NER tools can find a person's name, an organizations' name, currency-related symbols, numerical formats, and so on that will help your proofreading tool identify exceptional cases present in text. Then, according to the NER tag, the system can suggest the necessary changes. Take the sentence, **Bank of America announced its earning today morning**. In this case, the NER tool gives the tag organization for *Bank of America*, which helps our system better understand the meaning of the sentence and the structure of the sentence.

NER tags are also very important if you are building a question-answer system as it is very crucial to extract entities in this system. Once you have generated the entities, you can use a syntactic relationship in order to understand questions. After this stage, you can process the question and generate the answer.

# Challenges

There are certain challenges for the NER system, which are as follows:

*   NER tools train on a closed domain dataset. So, an NER system developed for one domain does not typically perform well on an other domain. This requires a universal NER tool that can work for all domains, and after training it should able to generalize enough to deal with unseen situations.
*   Sometimes you will find words which are the names of locations as well as the name of a person. The NER tool can't handle a case where one word can be expressed as the location name, person name, and organization name. This is a very challenging case for all NER tools. Suppose you have word TATA hospital; the single the words TATA can be the name of a person as well as the name of an organization. In this case, the NER tool can't decide whether TATA is the name of a person or the name of an organization.
*   To build an NER tool specifically for microblogging web platforms is also a challenging task.

Let's move on to the next section, which is about n-gram algorithms. You will get to learn some very interesting stuff.

# n-grams

n-gram is a very popular and widely used technique in the NLP domain. If you are dealing with text data or speech data, you can use this concept.

Let's look at the formal definition of n-grams. An n-gram is a continuous sequence of n items from the given sequence of text data or speech data. Here, items can be phonemes, syllables, letters, words, or base pairs according to the application that you are trying to solve.

There are some versions of n-grams that you will find very useful. If we put n=1, then that particular n-gram is referred to as a unigram. If we put n=2, then we get the bigram. If we put n=3, then that particular n-gram is referred to as a trigram, and if you put n=4 or n=5, then these versions of n-grams are referred to as four gram and five gram, respectively. Now let's take some examples from different domains to get a more detailed picture of n-grams. See examples from NLP and computational biology to understand a unigram in *Figure 5.43*:

![](img/22d797e8-3c08-4f7b-a130-a9910ea82a98.png)

Figure 5.43: Unigram example sequence

You have seen unigrams. Now we will look at the bigram. With bigrams, we are considering overlapped pairs, as you can see in the following example. We have taken the same NLP and computational biology sequences to understand bigrams. See *Figure 5.44*:

![](img/8b9aa773-e7aa-4bfb-876a-082eed4ee00d.png)

Figure 5.44: Bigram example sequence

If you understood the bigram overlapped pairing concept from the example, then a trigram will be easier for you to understand. A trigram is just an extension of the bigram, but if you are still confused, then let's explain it for you in laymans' terms. In the first three rows of *Figure 5.44*, we generated a character-based bigram and the fourth row is a word-based bigram. We will start from the first character and consider the very next character because we are considering n=2 and the same is applicable to words as well. See the first row where we are considering a bigram such as *AG* as the first bigram. Now, in the next iteration, we are considering *G* again and generate *GC*. In the next iteration, we are considering *C* again and so on. For generating a trigram, see the same examples that we have looked at previously for. Refer to *Figure 5.45*:

![](img/e6cf426f-2386-4eb9-bc01-a593893c4c0e.png)

Figure 5.45: Trigram example sequence

The preceding examples are very much self-explanatory. You can figure out how we are taking up the sequencing from the number of n. Here, we are taking the overlapped sequences, which means that if you are taking a trigram and taking the words **this**, **is**, and **a** as a single pair, then next time, you are considering **is**, **a**, and **pen**. Here, the word *is* overlaps, but these kind of overlapped sequences help store context. If we are using large values for n-five-gram or six-gram, we can store large contexts but we still need more space and more time to process the dataset.

# Understanding n-gram using a practice example

Now we are going to implement n-gram using the `nltk` library. You can see the code at this GitHub link:

[https://github.com/jalajthanaki/NLPython/tree/master/ch5/n_gram](https://github.com/jalajthanaki/NLPython/tree/master/ch5/n_gram).

You can see the code snippet in *Figure 5.46*:

![](img/06f637d3-27a3-490c-b05e-33e2b85e2a31.png)

Figure 5.46: NLTK n-gram code

You can see the output code snippet in *Figure 5.47*:

![](img/ee9c7255-a8f7-4d7f-ac85-0e3036b5a6f4.png)

Figure 5.47: Output of the n-gram

# Application

In this section, we will see what kinds of applications n-gram has been used in:

*   If you are making a plagiarism tool, you can use n-gram to extract the patterns that are copied, because that's what other plagiarism tools do to provide basic features
*   Computational biology has been using n-grams to identify various DNA patterns in order to recognize any unusual DNA pattern; based on this, biologists decide what kind of genetic disease a person may have

Now let's move on to the next concept, which is an easy but very useful concept for NLP applications: Bag of words.

# Bag of words

**Bag of words** (**BOW**) is the technique that is used in the NLP domain.

# Understanding BOW

This BOW model makes our life easier because it simplifies the representation used in NLP. In this model, the data is in the form of text and is represented as the bag or multiset of its words, disregarding grammar and word order and just keeping words. Here, text is either a sentence or document. Let's an example to give you a better understanding of BOW.

Let's take the following sample set of documents:

Text document 1: John likes to watch cricket. Chris likes cricket too.

Text document 2: John also likes to watch movies.

Based on these two text documents, you can generate the following list:

```py
List  of words= ["John", "likes", "to", "watch", "cricket", "Chris", "too", "also", "movies"] 

```

This list is called **BOW**. Here, we are not considering the grammar of the sentences. We are also not bothered about the order of the words. Now it's time to see the practical implementation of BOW. BOW is often used to generate features; after generating BOW, we can derive the term-frequency of each word in the document, which can later be fed to a machine learning algorithm. For the preceding documents, you can generate the following frequency list:

Frequency count for Document 1: [1, 2, 1, 1, 2, 1, 1, 0, 0]

Frequency count for Document 2: [1, 1, 1, 1, 0, 0, 0, 1, 1]

So, how did we generate the list of frequency counts? In order to generate the frequency count of Document 1, consider the list of words and check how many times each of the listed words appear in Document 1\. Here, we will first take the word, *John*, which appears in Document 1 once; the frequency count for Document 1 is 1\. **Frequency count for Document 1: [1]**. For the second entry, the word *like* appears twice in Document 1, so the frequency count is 2\. **Frequency count for Document 1: [1, 2].** Now, we will take the third word from our list and the word is *to***.** This word appears in Document 1 once, so we make the third entry in the frequency count as 1\. **Frequency count for Document 1: [1, 2, 1].** We have generated the frequency count for Document 1 and Document 2 in the same way. We will learn more about frequency in the upcoming section, TF-IDF, in this chapter.

# Understanding BOW using a practical example

In this section, we will look at the practical implementation of BOW using `scikit-learn`. You can find the code at this GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch5/bagofwordsdemo/BOWdemo.py](https://github.com/jalajthanaki/NLPython/blob/master/ch5/bagofwordsdemo/BOWdemo.py).

See the code snippet in *Figure 5.48*:

![](img/77214fed-ef02-4e3a-99ce-7c5a3cfd8791.png)

Figure 5.48: BOW scikit-learn implementation

The first row of the output belongs to the first document with the word, `words`, and the second row belongs to the document with the word, `wprds`. You can see the output in *Figure 5.49*:

![](img/2f298bed-8b6e-40c8-b447-4fdc6d00033a.png)

Figure 5.49: BOW vector representation

# Comparing n-grams and BOW

We have looked at the concepts of n-grams and BOW. So, let's now see how n-grams and BOW are different or related to each other.

Let's first discuss the differences. Here, the difference is in terms of their usage in NLP applications. In n-grams, word order is important, whereas in BOW it is not important to maintain word order. During the NLP application, n-gram is used to consider words in their real order so we can get an idea about the context of the particular word; BOW is used to build vocabulary for your text dataset.

Now let's look at some meaningful relationships between n-grams and BOW that will give you an idea of how n-grams and BOW are related to each other. If you are considering n-gram as a feature, then BOW is the text representation derived using a unigram. So, in that case, an n-gram is equal to a feature and BOW is equal to a representation of text using a unigram (one-gram) contained within.

Now, let's check out an application of BOW.

# Applications

In this section, we will look at which applications use BOW as features in the NLP domain:

*   If you want to make an NLP application that classifies documents in different categories, then you can use BOW.
*   BOW is also used to generate frequency count and vocabulary from a dataset. These derived attributes are then used in NLP applications such as sentiment analysis, Word2vec, and so on.

Now it's time to look at some of the semantic tools that we can use if we want to include semantic-level information in our NLP applications.

# Semantic tools and resources

Trying to get the accurate meaning of a natural language is still a challenging task in the NLP domain, although we do have some techniques that have been recently developed and resources that we can use to get semantics from natural language. In this section, we will try to understand these techniques and resources.

The latent semantic analysis algorithm uses t**erm frequency - inverse document Frequency** (**tf-idf**) and the concept of linear algebra, such as cosine similarity and Euclidean distance, to find words with similar meanings. These techniques are a part of distributional semantics. The other one is word2vec. This is a recent algorithm that has been developed by Google and can help us find the semantics of words and words that have similar meanings. We will explore word2vec and other techniques in [Chapter 6](c4861b9e-2bcf-4fce-94d4-f1e2010831de.xhtml), *Advance Features Engineering and NLP Algorithms*.

Apart from Word2vec, another powerful resource is `WordNet`, which is the largest corpus available to us and it's tagged by humans. It also contains sense tags for each word. These databases are really helpful for finding out the semantics of a particular word.

You can have a look at `WordNet` at the following link: [https://wordnet.princeton.edu/](https://wordnet.princeton.edu/)
Here, we have listed some of the most useful resources and tools for generating semantics. There is a lot of room for improvement in this area.

We have seen most of the NLP domain-related concepts and we have also seen how we can derive features using these concepts and available tools. Now it's time to jump into the next section, which will give us information about statistical features.

# Basic statistical features for NLP

In the last section, we looked at most of the NLP concepts, tools, and algorithms that can be used to derive features. Now it's time to learn about some statistical features as well. Here, we will explore the statistical aspect. You will learn how statistical concepts help us derive some of the most useful features.

Before we jump into statistical features, as a prerequisite, you need to understand basic mathematical concepts, linear algebra concepts, and probabilistic concepts. So here, we will seek to understand these concepts first and then understand the statistical features.

# Basic mathematics

We will begin with the basics of linear algebra and probability; this is because we want you to recall and memorize the necessary concepts so it will help you in this chapter as well as the upcoming chapters. We will explain the necessary math concepts as and when needed.

# Basic concepts of linear algebra for NLP

In this section, we will not look at all the linear algebra concepts in great detail. The purpose of this section is to get familiar with the basic concepts. Apart from the given concepts, there are many other concepts that can be used in NLP applications. Here, we will cover only the much needed concepts. We will give you all the necessary details about algorithms and their mathematical aspects in upcoming chapters. Let's get started with the basics.

There are four main terms that you will find consistently in NLP and ML:

*   **Scalars**: They are just single, the real number
*   **Vectors**: They are a one-dimensional array of the numbers
*   **Matrices**: They are two-dimensional arrays of the numbers
*   **Tensors**: They are n-dimensional arrays of the numbers

The pictorial representation is given in *Figure 5.50*:

![](img/967508ae-8029-42fc-8ef2-1eaa80af81c7.png)

Figure 5.50: A pictorial representation of scalar, vector, matrix, and tensor (Image credit: http://hpe-cct.github.io/programmingGuide/img/diagram1.png)

Matrix manipulation operations are available in the `NumPy` library. You can perform vector-related operations using the `SciPy` and `scikit-learn` libraries. We will suggest certain libraries because their sources are written to give you optimal solutions and provide you with a high-level API so that you don't need to worry about what's going on behind the scenes. However, if you want to develop a customized application, then you need to know the math aspect of each manipulation. We will also look at the concept of linear regression, gradient descent, and linear algebra. If you really want to explore math that is related to machine learning and deep learning, then the following learning materials can help you.

Part one of this book will really help you:
[http://www.deeplearningbook.org/](http://www.deeplearningbook.org/)

A cheat sheet of statistics, linear algebra, and calculus can be found at this link:
[https://github.com/jalajthanaki/NLPython/tree/master/Appendix2/Cheatsheets/11_Math](https://github.com/jalajthanaki/NLPython/tree/master/Appendix2/Cheatsheets/11_Math).

If you are new to math, we recommend that you check out these videos:
[https://www.khanacademy.org/math/linear-algebra](https://www.khanacademy.org/math/linear-algebra) [https://www.khanacademy.org/math/probability](https://www.khanacademy.org/math/probability) [https://www.khanacademy.org/math/calculus-home](https://www.khanacademy.org/math/calculus-home)
[https://www.khanacademy.org/math/calculus-home/multivariable-calculus](https://www.khanacademy.org/math/calculus-home/multivariable-calculus) [https://www.khanacademy.org/math](https://www.khanacademy.org/math) [If you want to see the various vector similarity concepts, then this article will help you:](https://www.khanacademy.org/math/calculus-home/multivariable-calculus) [http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/](http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/)

Now let's jump into the next section, which is all about probability. This is one of the core concepts of probabilistic theory.

# Basic concepts of the probabilistic theory for NLP

In this section, we will look at some of the concepts of probabilistic theory. We will also look at some examples of them so that you can understand what is going on. We will start with probability, then the concept of an independent event, and then conditional probability. At the end, we will look at the Bayes rule.

# Probability

Probability is a measure of the likelihood that a particular event will occur. Probability is quantified as a number and the range of probability is between 0 and 1\. 0 means that the particular event will never occur and 1 indicates that the particular event will definitely occur. Machine learning techniques use the concept of probability widely. Let's look at an example just to refresh the concept. Refer to *Figure 5\. 51*:

![](img/8da1e11f-fd1e-4f0e-ba03-93ddc50419ae.png)

Figure 5.51: Probability example (Image credit: http://www.algebra-class.com/image-files/examples-of-probability-3.gif)

Now let's see what dependent and independent events are.

# Independent event and dependent event

In this section, we will look at what dependent events and independent events are. After that, we will see how to decide if an event is dependent or not. First, let's begin with definitions.

If the probability of one event doesn't affect the probability of the other event, then this kind of event is called an independent event. So technically, if you take two events, A and B, and if the fact that A occurs does not affect the probability of B occurring, then it's called an independent event. Flipping a fair coin is an independent event because it doesn't depend on any other previous events.

Sometimes, some events affect other events. Two events are said to be dependent when the probability of one event occurring influences the other event's occurrence.

For example, if you were to draw two cards from a deck of 52 cards, and on your first draw you had an ace, the probability of drawing another ace on the second draw has changed because you drew an ace the first time. Let's calculate these different probabilities to see what's going on.

There are four aces in a deck of 52 cards. See *Figure 5.52*:

![](img/e3321a03-afb3-4ccd-9bf5-ce59dccbdb36.png)

Figure 5.52: Equation of probability

On your first draw, the probability of getting an ace is in *Figure 5.53*:

![](img/93b3d6c4-1664-4737-8ad8-7a254bd3c42b.png)

Figure 5.53: Calculation step (Image credit: https://dj1hlxw0wr920.cloudfront.net/userfiles/wyzfiles/02cec729-378c-4293-8a5c-3873e0b06942.gif)

Now if you don't return this drawn card to the deck, the probability of drawing an ace on the second round is given in the following equation. See *Figure 5.54*:

![](img/881756b3-18bf-4985-b5ff-559bddc014a0.png)

Figure 5.54: Dependent event probability equation (Image credit: https://dj1hlxw0wr920.cloudfront.net/userfiles/wyzfiles/7a45b393-0275-47ac-93e1-9669f5c31caa.gif)

See the calculation step in *Figure 5.55*:

![](img/f5d8f123-48bf-4717-b544-97232a6d1084.png)

Figure 5.55: Calculation step (Image credit: https://dj1hlxw0wr920.cloudfront.net/userfiles/wyzfiles/11221e29-96ea-44fb-b7b5-af614f1bec96.gif)

See the final answer in *Figure 5.56*:

![](img/5bf68296-740a-45d5-9aec-1c60bd3a475b.png)

Figure 5.56: Final answer of the example (Image credit: https://dj1hlxw0wr920.cloudfront.net/userfiles/wyzfiles/78fdf71e-fc1c-41d2-8fb8-bc2baeb25c25.gif)

As you can see, the preceding two probability values are different, so we say that the two events are dependent because the second event depends on the first event.

The mathematical condition to check whether the events are dependent or independent is given as: events A and B are independent events if, and only if, the following condition will be satisfied:

*P(A ∩ B) = P(A) * P(B)*

Otherwise, *A* and *B* are called dependent events.

Now let's take an example to understand the defined condition.

**Example**: A poll finds that 72% of the population of Mumbai consider themselves football fans. If you randomly pick two people from the population, what is the probability that the first person is a football fan and the second is as well? That the first one is and the second one isn't?

**Solution**: The first person being a football fan doesn't have any impact on whether the second randomly selected person is a football fan or not. Therefore, the events are independent.

The probability can be calculated by multiplying the individual probabilities of the given events together. If the first person and second person both are football fans, then P(A∩B) = P(A) P(B) = .72 * .72 = .5184.

For the second question: The first one is a football fan, the second one isn't:

*P(A∩ not B) = P(A) P( B' ) = .72 * ( 1 - 0.72) = 0.202*.

In this part of the calculation, we multiplied by the complement.

Here, events *A* and *B* are independent because the equation *P(A∩B) = P(A) P(B)* holds true.

Now it's time to move on to the next concept called conditional probability.

# Conditional probability

In this section, we will look at a concept called conditional probability. We will use the concept of a dependent event and independent event to understand the concept of conditional probability.

The conditional probability of an event *B* is the probability that the event will occur given the knowledge that an event, *A*, has already occurred. This probability is written as *P(B|A)*, the notation for the probability of *B* given *A*. Now let's see how this conditional probability turns out when events are independent. Where events *A* and *B* are independent, the conditional probability of event *B* given event *A* is simply the probability of event B, that is, *P(B)*. What if events *A* and *B* are not independent? Then, the probability of the intersection of *A* and *B* means that the probability that both events occur is defined by the following equation:

*P(A and B) = P(A) * P(B|A)*

Now we will look at an example.

**Example**: Jalaj's two favorite food items are tea and pizza. Event A represents the event that I drink tea for my breakfast. B represents the event that I eat pizza for lunch. On randomly selected days, the probability that I drink tea for breakfast, P(A), is 0.6\. The probability that I eat pizza for lunch, P(B), is 0.5 and the conditional probability that I drink tea for breakfast, given that I eat pizza for lunch, P(A|B) is 0.7\. Based on this, please calculate the conditional probability of P(B|A). P(B|A) will indicate the probability that I eat pizza for lunch, given that I drink tea for breakfast. In layman's terms, find out the probability of having pizza for lunch when drinking tea for breakfast.

**Solution**

*P(A) = 0.6 , P(B) =0.5 , P(A|B) =0.7*

Here, two events are dependent because the probability of B being true has changed the probability of A being true. Now we need to calculate P(B|A).

See the equation *P(A and B) = P(A) * P(B|A)*. To find out P(B|A), we first need to calculate P(A and B):

*P(A and B) = P(B) * P(A|B) = P(A) * P(B|A)*

Here, we know that *P(B) = 0.5 and P(A|B) =0.7*

*P(A and B) = 0.5 * 0.7 = 0.35*

*P(B|A) = P(A and B) / P(A) = 0.35 / 0.6 = 0.5833*

So, we have found the conditional probability for dependent events.

Now we have seen the basics of probability that we will use in upcoming chapters to understand ML algorithms. We will define additional concepts as we go. The `scikit-learn`, TensorFlow, SparkML, and other libraries already implement major probability calculation, provide us with high-level APIs, and have options that can change the predefined parameter and set values according to your application. These parameters are often called **hyperparameters**. To come up with the best suited values for each of the parameters is called **hyperparameter tuning**. This process helps us optimize our system. We will look at hyperparameter tuning and other major concepts in [Chapter 8](97808151-90d2-4034-8d53-b94123154265.xhtml), *Machine Learning for NLP Applications*.

This is the end of our prerequisite section. From this section onwards, we look at see some statistical concepts that help us extract features from the text. Many NLP applications also use them.

# TF-IDF

The concept TF-IDF stands for **term frequency-inverse document frequency**. This is in the field of numerical statistics. With this concept, we will be able to decide how important a word is to a given document in the present dataset or corpus.

# Understanding TF-IDF

This is a very simple but useful concept. It actually indicates how many times a particular word appears in the dataset and what the importance of the word is in order to understand the document or dataset. Let's give you an example. Suppose you have a dataset where students write an essay on the topic, My Car. In this dataset, the word **a** appears many times; it's a high frequency word compared to other words in the dataset. The dataset contains other words like **car**, **shopping**, and so on that appear less often, so their frequency are lower and they carry more information compared to the word, **a**. This is the intuition behind TF-IDF.

Let's explain this concept in detail. Let's also look at its mathematical aspect. TF-IDF has two parts: Term Frequency and Inverse Document Frequency. Let's begin with the term frequency. The term is self-explanatory but we will walk through the concept. The term frequency indicates the frequency of each of the words present in the document or dataset. So, its equation is given as follows:

*TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)*

Now let's look at the second part - inverse document frequency. IDF actually tells us how important the word is to the document. This is because when we calculate TF, we give equal importance to every single word. Now, if the word appears in the dataset more frequently, then its term frequency (TF) value is high while not being that important to the document. So, if the word **the** appears in the document 100 times, then it's not carrying that much information compared to words that are less frequent in the dataset. Thus, we need to define some weighing down of the frequent terms while scaling up the rare ones, which decides the importance of each word. We will achieve this with the following equation:

*IDF(t) = log[10](Total number of documents / Number of documents with term t in it).*

So, our equation is calculate TF-IDF is as follows.

*TF * IDF = [ (Number of times term t appears in a document) / (Total number of terms in the document) ] * log10(Total number of documents / Number of documents with term t in it).*

Note that in TF-IDF, - is hyphen, not the minus symbol. In reality, TF-IDF is the multiplication of TF and IDF, such as *TF * IDF*.

Now, let's take an example where you have two sentences and are considering those sentences as different documents in order to understand the concept of TF-IDF:

Document 1: This is a sample.

Document 2: This is another example.

Now to calculate TF-IDF, we will follow these steps:

1.  We first calculate the frequency of each word for each document.
2.  We calculate IDF.
3.  We multiply TF and IDF.

Refer to *Figure 5.57*:

![](img/8e8559f0-b3ed-4ffe-b474-6016d574dfa5.png)

Figure 5.57: TF-IDF example

Now, let's see the calculation of IDF and TF * IDF in *Figure 5.58*:

![](img/c1278a75-815d-417b-b884-4e54c9344ce7.png)

Figure 5.58: TF-IDF example

# Understanding TF-IDF with a practical example

Here, we will use two libraries to calculate TF-IDF - textblob and scikit-learn. You can see the code at this GitHub link:

[https://github.com/jalajthanaki/NLPython/tree/master/ch5/TFIDFdemo](https://github.com/jalajthanaki/NLPython/tree/master/ch5/onehotencodingdemo).

# Using textblob

You can see the code snippet in *Figure 5.59*:

![](img/a5fbe677-d716-4aab-9325-fd83780e4757.png)

Figure 5.59: TF-IDF using textblob

The output of the code is in *Figure 5.60*:

![](img/0c7946ba-5a96-4aa1-9638-93c26969e884.png)

Figure 5.60: Output of TF-IDF for the word short

# Using scikit-learn

We will try to generate the TF-IDF model using a small Shakespeare dataset. For a new given document with a TF-IDF score model, we will suggest the top three keywords for the document. You can see the code snippet in *Figure 5.61*:

![](img/15efd78d-2e2a-493a-ad28-797f7ca62476.png)

Figure 5.61: Using scikit-learn to generate a TF-IDF model

You can see the output in *Figure 5.62*:

![](img/be479c79-eef3-4acc-b3a9-874c2e4480b6.png)

Figure 5.62: Output of the TF-IDF model

Now it's time to see where we can use this TF-IDF concept, so let's look at some applications.

# Application

In this section, we will look at some cool applications that use TF-IDF:

*   In general, text data analysis can be performed by TF-IDF easily. You can get information about the most accurate keywords for your dataset.
*   If you are developing a text summarization application where you have a selected statistical approach, then TF-IDF is the most important feature for generating a summary for the document.
*   Variations of the TF-IDF weighting scheme are often used by search engines to find out the scoring and ranking of a document's relevance for a given user query.
*   Document classification applications use this technique along with BOW.

Now let's look at the concept of vectorization for an NLP application.

# Vectorization

Vectorization is an important aspect of feature extraction in the NLP domain. Transforming the text into a vector format is a major task.

Vectorization techniques try to map every possible word to a specific integer. There are many available APIs that make your life easier. `scikit-learn` has `DictVectorizer` to convert text to a one-hot encoding form. The other API is the `CountVectorizer`, which converts the collection of text documents to a matrix of token counts. Last but not least, there are a couple of other APIs out there. We can also use word2vec to convert text data to the vector form. Refer to this link's *From text* section for more details:

[http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text).

Now let's look at the concept of one-hot encoding for an NLP application. This one-hot encoding is considered as part of vectorization.

# Encoders and decoders

The concept of encoding in NLP is quite old as well as useful. As we mentioned earlier, it is not easy to handle categorical data attributes present in our dataset. Here, we will explore the encoding technique named one-hot encoding, which helps us convert our categorical features in to a numerical format.

# One-hot encoding

In an NLP application, you always get categorical data. The categorical data is mostly in the form of words. There are words that form the vocabulary. The words from this vocabulary cannot turn into vectors easily.

Consider that you have a vocabulary with the size N. The way to approximate the state of the language is by representing the words in the form of one-hot encoding. This technique is used to map the words to the vectors of length n, where the n^(th) digit is an indicator of the presence of the particular word. If you are converting words to the one-hot encoding format, then you will see vectors such as 0000...001, 0000...100, 0000...010, and so on. Every word in the vocabulary is represented by one of the combinations of a binary vector. Here, the nth bit of each vector indicates the presence of the nth word in the vocabulary. So, how are these individual vectors related to sentences or other words in the corpus? Let's look at an example that will help you understand this concept.

For example, you have one sentence, *Jalaj likes NLP*. Suppose after applying one-hot encoding, this sentence becomes 00010 00001 10000\. This vector is made based on the vocabulary size and encoding schema. Once we have this vector representation, then we can perform the numerical operation on it. Here, we are turning words into vectors and sentences into matrices.

# Understanding a practical example for one-hot encoding

In this section, we will use `scikit-learn` to generate one-hot encoding for a small dataset. You can find the code at this GitHub link:

[https://github.com/jalajthanaki/NLPython/tree/master/ch5/onehotencodingdemo](https://github.com/jalajthanaki/NLPython/tree/master/ch5/onehotencodingdemo).

You can see the code snippet in *Figure 5.63*:

![](img/79dd41bf-e6ce-4af1-8601-716404d78099.png)

Figure 5.63: Pandas and scikit-learn to generate one-hot encoding

You can see the output in *Figure 5.64*:

![](img/9131b126-45aa-4853-a880-8f0b1fb7c3a5.png)

Figure 5.64: Output of one-hot encoding

# Application

These techniques are very useful. Let's see some of the basic applications for this mapping technique:

*   Many artificial neural networks accept input data in the one-hot encoding format and generate output vectors that carry the sematic representation as well
*   The word2vec algorithm accepts input data in the form of words and these words are in the form of vectors that are generated by one-hot encoding

Now it's time to look at the decoding concept. Decoding concepts are mostly used in deep learning nowadays. So here, we will define the decoder in terms of deep learning because we will use this encoding and decoding architecture in [Chapter 9](f414d38e-b88e-4239-88bd-2d90e5ce67ab.xhtml), *Deep Learning for NLU and NLG Problems*, to develop a translation system.

An encoder maps input data to a different feature representation; we are using one-hot encoding for the NLP domain. A decoder maps the feature representation back to the input data space. In deep learning, a decoder knows which vector represents which words, so it can decode words as per the given input schema. We will see the detailed concept of encoder-decoder when we cover the sequence-to-sequence model.

Now, let's look at the next concept called **normalization**.

# Normalization

Here, we will explain normalization in terms of linguistics as well as statistics. Even though they are different, the word normalization can create a lot of confusion. Let's resolve this confusion.

# The linguistics aspect of normalization

The linguistics aspect of normalization includes the concept text normalization. Text normalization is the process of transforming the given text into a single canonical form. Let's take an example to understand text normalization properly. If you are making a search application and you want the user to enter **John**, then John becomes a search string and all the strings that contain the word **John** should also pop up. If you are preparing data to search, then people prefer to take the stemmed format; even if you search **flying** or **flew**, ultimately these are forms that are derived from the word **fly**. So, the search system uses the stemmed form and other derived forms are removed. If you recall Chapter 3, *Understanding Structure of Sentences*, then you will remember that we have already discussed how to derive lemma, stem, and root.

# The statistical aspect of normalization

The statistical aspect of normalization is used to do features scaling. If you have a dataset where one data attribute's ranges are too high and the other data attributes' ranges are too small, then generally we need to apply statistical techniques to bring all the data attributes or features into one common numerical range. There are many ways to perform this transformation, but here we will illustrate the most common and easy method of doing this called **min-max scaling**. Let's look at equation and mathematical examples to understand the concept.

Min-max scaling brings the features in the range of [0,1]. The general formula is given in *Figure 5.65*:

![](img/4d1a6acf-7595-4eea-bb16-6323a59ef1b9.jpg)

Figure 5.65: The min-max normalization equation

Suppose you have features values such as *[1, 8, 22, 25]*; i you apply the preceding formula and calculate the value for each of the elements, then you will get the feature with a range of [0,1]. For the first element, *z = 1 - 1/ 25 -1 = 0*, for the second element, *z =8 -1 /25-1 = 0.2917*, and so on. The `scikit-learn` library has an API that you can use for min-max scaling on the dataset.

In the next section, we will cover the language model.

# Probabilistic models

We will discuss one of the most famous probabilistic models in NLP, which has been used for a variety of applications - the language model. We will look at the basic idea of the language model. We are not going to dive deep into this, but we will get an intuitive idea on how the language model works and where we can use it.

# Understanding probabilistic language modeling

There are two basic goals of the **language model** (**LM**):

*   The goal of LM is to assign probability to a sentence or sequence of words
*   LM also tells us about the probability of the upcoming word, which means that it indicates which is the next most likely word by observing the previous word sequence

If any model can compute either of the preceding tasks, it is called a language model. LM uses the conditional probability chain rule. The chain rule of conditional probability is just an extension of conditional probability. We have already seen the equation:

*P(A|B) = P(A and B) / P(B)*

*P(A and B) = P(A,B) = P(A|B) P(B)*

Here, P(A,B) is called **joint probability**. Suppose you have multiple events that are dependent, then the equation to compute joint probability becomes more general:

*P(A,B,C,D) = P(A) P(B|A)P(C|A,B)P(D|A,B,C)*

*P(x1,x2,x3,...,xn) =P(x1)P(x2|x1)P(x3|x1,x2)...P(xn|x1,x2,x3,...xn-1)*

The preceding equation is called chain rule for conditional probability. LM uses this to predict the probability of upcoming words. We often calculate probability by counting the number of times a particular event occurs and dividing it by the total number of possible combinations, but we can't apply this to language because, with certain words, you can generate millions of sentences. So, we are not going to use the probability equation; we are using an assumption called **Markov Assumption** to calculate probability instead. Let's understand the concept intuitively before looking at a technical definition of it. If you have very a long sentence and you are trying to predict what the next word in the sentence sequence will be, then you actually need to consider all the words that are already present in the sentence to calculate the probability for the upcoming word. This calculation is very tedious, so we consider only the last one, two or three words to compute the probability for the upcoming word; this is called the Markov assumption. The assumption is that you can calculate the probability of the next word that comes in a sequence of the sentence by looking at the last word two. Let's take an example to understand this. If you want to calculate the probability of a given word, then it is only dependent on the last word. You can see the equation here:

*P(the | its water is so transparent that) = P(the | that) or you can consider last two words P(the | its water is so transparent that) = P(the | transparent that)*

A simple LM uses a unigram, which means that we are just considering the word itself and calculating the probability of an individual word; you simply take the probability of individual words and generate a random word sequence. If you take a bigram model, then you consider that one previous word will decide the next word in the sequence. You can see the result of the bigram model in *Figure 5.66*:

![](img/cca09d81-e3d0-4671-912d-29684c6e2954.png)

Figure 5.66: Output using a bigram LM

How can we count the n-gram probability that is a core part of LM? Let's look at the bigram model. We will see the equation and then go through the example. See the equation in *Figure 5.67*:

![](img/56ac8ba9-9e7d-4a60-a2cf-bf8afa430ee2.png)

Figure 5.67: Equation to find out the next most likely word in the sequence

The equation is easy to understand. We need to calculate how many times the words *wi-1* and *wi* occurred together, and we also need to count how many times the word *wi-1* occurred. See the example in *Figure 5.68*:

![](img/1d194dbd-69b3-4b63-961b-a462a8056fb5.png)

Figure 5.68: Example to find out the most likely word using LM

As you can see, we is followed by *<s>* twice in three given sentences, so we have *P(I|<s>) =2/3*, and for every word, we will calculate the probability. Using LM, we can come to know how the word pairs are described in the corpus as well as what the more popular word pairs that occur in the corpus are. If we use a four-gram or five-gram model, it will give us a good result for LM because some sentences have a long-dependency relationship in their syntax structure with subject and verbs. So, with a four-gram and five-gram model, you can build a really good LM.

# Application of LM

LM has a lot of great applications in the NLP domain. Most NLP applications use LM at some point. Let's see them:

*   Machine translation systems use LM to find out the probability for each of the translated sentences in order to decide which translated sentence is the best possible translation for the given input sentence
*   To spell the correct application, we can use a bigram LM to provide the most likely word suggestion
*   We can use LM for text summarization
*   We can use LM in a question answering system to rank the answers as per their probability

# Indexing

**Indexing** is quite a useful technique. This is used to convert the categorical data to its numerical format. In an NLP application, you may find that the data attributes are categorical and you want to convert them to a certain numerical value. In such cases, this indexing concept can help you. We can use the SparkML library, which has a variety of APIs to generate indexes. SparkML has an API named StringIndexer that uses the frequency of the categorical data and assigns the index as per the frequency count. So, the most frequent category gets an index value of 0\. This can sometimes be a naïve way of generating indexing, but in some analytical applications, you may find this technique useful. You can see the example at this link:

[https://spark.apache.org/docs/latest/ml-features.html#stringindexer](https://spark.apache.org/docs/latest/ml-features.html#stringindexer).

SparkML has the API, IndexToString, which you can use when you need to convert your numerical values back to categorical values. You can find the example at this link:

[https://spark.apache.org/docs/latest/ml-features.html#indextostring](https://spark.apache.org/docs/latest/ml-features.html#indextostring).

# Application

Here are some applications which use indexing for extracting features:

*   When we are dealing with a multiclass classifier and our target classes are in the text format and we want to convert our target class labels to a numerical format, we can use StingIndexer
*   We can also generate the text of the `target` class using the IndexToString API

Now it's time to learn about a concept called ranking.

# Ranking

In many applications, ranking plays a key role. The concept of **ranking** is used when you search anything on the web. Basically, the ranking algorithm is used to find the relevance of the given input and generated output.

Let's look at an example. When you search the web, the search engine takes your query, processes it, and generates some result. It uses the ranking algorithm to find the most relevant link according to your query and displays the most relevant link or content at the top and least relevant at the end. The same thing happens when you visit any online e-commerce website; when you search for a product, they display the relevant product list to you. To make their customer experience enjoyable, they display those products that are relevant to your query, whose reviews are good, and that have affordable prices. These all are the parameters given to the ranking algorithm in order to generate the most relevant products.

The implementation of the ranking algorithm is not a part of this book. You can find more information that will be useful here:
[https://medium.com/towards-data-science/learning-to-rank-with-python-scikit-learn-327a5cfd81f](https://medium.com/towards-data-science/learning-to-rank-with-python-scikit-learn-327a5cfd81f).

Indexing and ranking are not frequently used in the NLP domain, but they are much more important when you are trying to build an application related to analytics using machine learning. It is mostly used to learn the user's preferences. If you are making a Google News kind of NLP application, where you need to rank certain news events, then ranking and indexing plays a major role. In a question answering system, generating ranks for the answers is the most critical task, and you can use indexing and ranking along with a language model to get the best possible result Applications such as grammar correction, proofreading, summarization systems, and so on don't use this concept.

We have seen most of the basic features that we can use in NLP applications. We will be using most of them in Chapter 8, *Machine Learning for NLP Application*, where we will build some real-life NLP applications with ML algorithms. In the upcoming section, we will explain the advantages and challenges of features engineering.

# Advantages of features engineering

Features engineering is the most important aspect of the NLP domain when you are trying to apply ML algorithms to solve your NLP problems. If you are able to derive good features, then you can have many advantages, which are as follows:

*   Better features give you a lot of flexibility. Even if you choose a less optimal ML algorithm, you will get a good result. Good features provide you with the flexibility of choosing an algorithm; even if you choose a less complex model, you get good accuracy.
*   If you choose good features, then even simple ML algorithms do well.
*   Better features will lead you to better accuracy. You should spend more time on features engineering to generate the appropriate features for your dataset. If you derive the best and appropriate features, you have won most of the battle.

# Challenges of features engineering

Here, we will discuss the challenges of features engineering for NLP applications. You must be thinking that we have a lot of options available in terms of tools and algorithms, so what is the most challenging part? Let's find out:

*   In the NLP domain, you can easily derive the features that are categorical features or basic NLP features. We have to convert these features into a numerical format. This is the most challenging part.
*   An effective way of converting text data into a numerical format is quite challenging. Here, the trial and error method may help you.
*   Although there are a couple of techniques that you can use, such as TF-IDF, one-hot encoding, ranking, co-occurrence matrix, word embedding, Word2Vec, and so on to convert your text data into a numerical format, there are not many ways, so people find this part challenging.

# Summary

In this chapter, we have seen many concepts and tools that are widely used in the NLP domain. All of these concepts are the basic building blocks of features engineering. You can use any of these techniques when you want to generate features in order to generate NLP applications. We have looked at how parse, POS taggers, NER, n-grams, and bag-of-words generate Natural Language-related features. We have also explored the how they are built and what the different ways to tweak some of the existing tools are in case you need custom features to develop NLP applications. Further, we have seen basic concepts of linear algebra, statistics, and probability. We have also seen the basic concepts of probability that will be used in ML algorithms in the future. We have looked at some cool concepts such as TF-IDF, indexing, ranking, and so on, as well as the language model as part of the probabilistic model.

In the next chapter, we will look at advanced features such as word2vec, Doc2vec, Glove, and so on. All of these algorithms are part of word embedding techniques. These techniques will help us convert our text features into a numerical format efficiently; especially when we need to use semantics. The next chapter will provide you with much more detailed information about the word2Vec algorithm. We will cover each and every technicality behind the word2vec model. We will also understand how an **artificial neural network** (**ANN**) is used to generate the semantic relationship between words, and then we will explore an extension of this concept from word level, sentence level, document level, and so on. We will build an application that includes some awesome visualization for word2vec. We will also discuss the importance of vectorization, so keep reading!