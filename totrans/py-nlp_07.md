

# Rule-Based System for NLP

We learned to derive various features by using the concepts of linguistics and statistics in [Chapter 5](07f71ca1-6c8a-492d-beb3-a47996e93f04.xhtml), *Feature Engineering and NLP Algorithms* and [Chapter 6](c4861b9e-2bcf-4fce-94d4-f1e2010831de.xhtml), *Advanced Feature Engineering and NLP Algorithms*. For developing an NLP application, these features are going to be fed into the algorithms. These algorithms take features as input. As you know, we are referring to algorithms as black boxes that perform some kind of magic and gives us the appropriate output. Refer to *Figure 7.1*, which demonstrates our journey so far:

![](img/177ea531-388f-40dd-aaf5-590ead0a3f4e.png)

Figure 7.1: Stages we have learned so far

Congratulations, you have learned a lot about NLP, and specifically about the NLU!

Now, it is high time for us to explore the algorithms which we use to develop NLP applications. We refer to these algorithms, techniques, or approaches as our black boxes and their logic is works as some magic for us. Now, it's time to dive deep into these black boxes and understand the magic.

Algorithms (implementation techniques or approaches) for NLP applications can be divided into two parts. Refer to *Figure 7.2*:

![](img/6ac4fc1a-6734-4476-bf61-e64c3b495e59.png)

Figure 7.2: Algorithms or approaches or implementation techniques for black boxes

We will look at the **rule-based** (**RB**) system in this chapter and machine learning approaches in [Chapter 8](97808151-90d2-4034-8d53-b94123154265.xhtml), *Machine Learning for NLP Problems* and [Chapter 9](f414d38e-b88e-4239-88bd-2d90e5ce67ab.xhtml), *Deep Learning for NLP and NLG Problems*.

In this chapter, we are going to focus on the rule-based system. We are going to touch upon the following topics:

*   Understanding of the RB system
*   Purpose of having the RB system
*   Architecture of the RB system
*   Understanding the RB system development life cycle
*   Applications
*   Developing NLP applications using the RB system
*   Comparing the RB approach with other approaches
*   Advantages
*   Disadvantages
*   Challenges
*   Recent trends for the RB system

So, let's get started!

# Understanding of the rule-based system

RB systems are also known as **knowledge-based systems**. But first, we will see what the RB system means and what it does for us? What kind of NLP applications can be implemented by using this approach? For a better understanding, I will explain the concepts with the help of the applications.

# What does the RB system mean?

The rule-based system is defined as by using available knowledge or rules, we develop such a system which uses the rules, apply the available system rules on a corpus and try to generate or inference the results. Refer *Figure 7.3*, which will give you an idea about the RB system:

![](img/d8a699c8-d489-42d4-9204-daaa0baff37c.png)

Figure 7.3: Rule based system input/output flow

In short, you can say that the RB system is all about applying real-life rules or experiences to a available corpus, manipulating information as per the rules, and deriving certain decisions or results. Here, rules are generated or created by humans.

The RB system is used for interpreting available corpus (information) in a useful manner. Here, rules act as core logic for the RB system. The corpus is interpreted based on rules or knowledge, so our end result is dependent on these two factors, one is rules and the second is our corpus.

Now I will explain one of the **AI** (**Artificial Intelligence**) applications for getting the core essence of the RB system.

As humans, we all do very complicated work every day to perform some tasks. To perform tasks, we use our prior experiences or follow rules to successfully complete the task.

Take an example: If you are driving a car, you are following some rules. You have prior knowledge of these rules. Now, if you think about the self-driving car, then that car should react or perform the entire task that a human was doing previously. But cars don't understand how to drive automatically without a driver. To develop this kind of driver less car is quite complicated, as well as challenging.

Anyhow, you want to create a self-driving car. You know there are so many rules that the car needs to learn in order to perform as well as a human driver. Here you have a few major challenges:

*   This is a kind of complicated application
*   Lots of rules as well as situations need to be learned by the car
*   The accuracy of the self-driving car should be high enough to launch it on the market for the consumer

So, to solve the challenges, we follow various steps:

1.  We first try to reduce the problem statement to small chunks of a problem which is a subset of our original problem statement.
2.  We try to solve small chunks of the problem first.
3.  To solve it, we are trying to come up with generalized rules that help us to solve our problem as well as help us to achieve our end goal.

For our version of the driver less (self-driving) car, we need to think from the software perspective. So, what is the first step the car should learn? Think!

The car should learn to see and identify objects on the road. This is the first step for our car and we define some generalized rules which the car will use to learn and decide whether there is any object on the road?, then drive based on that. What should the speed of the car when it sees road conditions? And so on, (think right now using the rule-based system, and for some time don't think about the deep learning aspect to solve this step).

For every small part of our task, we try to define rules and feed that rule logic into the RB system. Then, we check whether that rule worked out properly on the given input data. We will also measure the performance of the system after getting the output.

Now, you must be thinking this is a book about NLP, so why am I giving an example of a generalized AI application? The reason behind it is that the self-driving car example is easy to relate to and can be understood by everyone. I want to highlight some of the points that also help us to understand the purpose of having a rule-based system.

Let's take one general example and understand the purpose:

*   This self-driving car example helps you in identifying that sometimes a task that is very easy for a human to perform is so much more complicated for machines to do by themselves
*   These kinds of complicated tasks need high accuracy! I mean very high!
*   We don't expect our system to cover and learn about all situations, but whatever rules we feed into the system, it should learn about those situations in the best manner
*   In the RB system, the coverage of various scenarios is less but accuracy of the system should be high. That is what we need
*   Our rules are derived from real-life human experience or by using knowledge of humans.
*   Development and implementation of rules is done by humans

All these points help us to decide when and where to use a rule-based system. This leads us to define our purpose of having a rule-based system. So let's jump into the next section where we define a rule of thumb for using the rule-based approach for any NLP or AI-related application.

# Purpose of having the rule-based system

Generally, the rule-based system is used for developing NLP applications and generalized AI applications. There are bunch of questions that we need to answer to generate a clear picture about the rule-based system.

# Why do we need the rule-based system?

The rule-based system tries to mimic human expert knowledge for the NLP applications. Here, we are going to address the factors that will help you to understand the purpose of the RB system:

*   Available corpus size is small
*   Output is too subjective
*   Easy for humans of a specific domain to generate some specialized rules
*   Difficult for machines to generate specialized rules by just observing small amounts of data
*   System output should be highly accurate

All the preceding factors are very much critical if you want to develop NLP application using the RB system. How do the preceding factors help you to decide whether you should choose the RB approach or not?

You need to ask the following questions:

*   Do you have a large amount of data or a small amount of data?
    *   If you have a small amount of data, then ask the next question and if you have a large amount of data, then you have many other options
*   Regarding the NLP application that you want to develop, is its output subjective or generalized?
    *   If you have a small amount of data and the output of the application which you want to develop is too subjective and you know, with a small amount of data, the machine cannot generalize the patterns, then choose the RB system
*   The NLP application that you want to develop should have very high accuracy:
    *   If the application that you want to develop should have high accuracy, almost the same as a human by using a small dataset, then choose the RB system
    *   Here, you should also keep in mind that human experts create rules for the system. According to that system, generate the output, so the RB system is highly accurate but does not cover all scenarios

The preceding questions define why and in what kind of situations we can use the RB system. If I needed to summarize the preceding questions, I would describe it like this: If you have small amount of data and you know you need a highly accurate system where it is easy for a human expert to identify various scenarios for making rules and its output but it is very difficult for machines to identify generalized rules by themselves accurately, then the RB system is for you! The output of the RB system should mimic the experiences of the human expert. This is the thumb rule for choosing the RB system.

We will see in [Chapter 9](f414d38e-b88e-4239-88bd-2d90e5ce67ab.xhtml), *Deep Learning for NLP and NLG Problems*, that there is a better approach when you have very large amount of data. For this chapter, the RB approach helps us to generate very accurate NLP applications.

# Which kind of applications can use the RB approach over the other approaches?

As we defined earlier, the RB system is developed with the help of human domain experts. Let's take some examples in this section which can help to prove our rule of thumb:

*   Say, we want to build the machine translation system from English to available Indian corpora and they are too small. The translation system should be accurate enough in order to develop it. We need human experts who know English as well as Gujarati. We don't want to address all the different levels of translation at a time, so we need to cover small chunks of the problem first and then on top of the developed prototype, we will build other chunks. So, here also, I would like to choose the RB system. What do you think?
*   Say we want to develop a grammar correction system for the English language. Suppose we have a small amount of parallel corpora (documents with grammatical mistakes and the same documents without grammatical mistakes), and by using the available corpus we need to make an accurate grammar correction application which identifies, as well as corrects, the grammatical mistakes. So, in this kind of application, which approach would you take? Think for a minute and then come up with your answers! Here, I would like to go with the RB system as per our rule of thumb.

# Exercise

*   If you wanted to develop a basic chatbot system, which approach would you take?
    *   RB approach
    *   ML approach
*   If you want to predict the sentiment of given sentences, which approach would you take?
    *   RB approach
    *   ML approach
    *   Hybrid approach
    *   None of them

# What kind of resources do you need if you want to develop a rule-based system?

Now you have understood why we are using the RB system and for which kinds of application we use it. The third important aspect is what do we need if we want to develop the RB system for any NLP or AI applications?

There are three main resources that we need to consider at this point. Refer to *Figure 7.4*:

![](img/af4d10b2-7aee-473c-8aff-783a8f11ab28.png)

Figure 7.4: Resources for implementing RB system

Now, let's see the details of each resource that helps us to define RB system components:

*   Domain expert (human expert/knowledge expert): For developing applications using the RB system, first and foremost, we need a domain expert, a person who knows almost everything about the domain.
    Suppose you want to build a machine translation system, then your domain expert could be a person who has deep knowledge of linguistics for the source and target languages. He can come up with rules by using his expertise and experience.
*   System architect (system engineer) of RB system: For defining the architecture of the RB system, you need a team or person who has the following expertise:
    *   Basic knowledge of the domain
    *   Deep knowledge or high experience in designing system architectures

Architecture is the most important part of the RB system because your architecture is one of the components which decide how efficient your whole system will be. Good architecture design for the RB system will provide good user experience, accurate and efficient output, and apart from that, it will make life easy for coders and other technical teams such as support or testing teams who will be able to work on the system easily. The system architecture is the responsibility of the system engineer or system architect.

*   Coders (developers or knowledge engineers) for implementing rules: Once rules are developed by domain experts and the system architecture has been designed properly, then coders or developers come into the picture. Coders are our real ninjas! They implement the rules by using programming languages and help to complete the application. Their coding skills are a much needed part of the RB system. Programming can be done using any of the programming or scripting languages such as C, C++, Java, Python, Perl, shell scripts, and so on. You can use any of them as per the architecture, but not all of them in a single system without a streamlined architecture.

We will look at more technical stuff regarding the architecture part a little later in this chapter.

# Architecture of the RB system

I will explain the architecture of the RB system by segregating it into three sections:

*   General architecture of RB system as an expert system
*   Practical architecture of the RB system for NLP applications
*   Custom architecture - RB system for NLP applications
*   Apache **UIMA** (**Unstructured Information Management Architecture**) the RB system for NLP applications

# General architecture of the rule-based system as an expert system

If we described our rule-based system as an expert system, then the architecture of this kind of rule-based system would be the same as in *Figure 7.5*:

![](img/cdb9c8a5-c956-4910-b2fe-21e5bfe8a59c.png)

Figure 7.5: Architecture of the RB system, considering it as an expert system

Let's look at each of the components of the architecture in detail:

*   **Domain expert**:
    *   As we saw in the previous section, domain experts are the ones who have expertise for a specific domain and they can help us to generate the rules to solve our problems
*   **Developers or knowledge engineer:**
    *   Developers use the rules which are created by the domain expert and convert them into a machine-understandable format using their coding skills
    *   Developers encode the rules created by experts
    *   Mostly, this encoding is in the form of pseudo codes
*   **Knowledge base:**
    *   The knowledge base is where all the rules can be put by experts
    *   The domain expert can add, update, or delete the rules
*   **Database or working storage:**
    *   All meta information-related rules can be put in the working storage
    *   Here, we can store rules as well as special scenarios, some lists if available, examples, and so on
    *   We also save data on which we want to apply rules
*   **Inference engine:**
    *   The inference engine is the core part of the system
    *   Here, we put in actual codes for our rules
    *   Rules will be triggered when predefined rules and conditions meet with a user query or on a dataset which we have given to the system as input
*   **User inference:**
    *   Sometimes, our end users also provide some conditions to narrow down their results, so all these user inference will also be considered when our system generates the output
*   **User interface:**
    *   The user interface helps our user to submit their input and in return they will get the output
    *   This provides an interactive environment for our end users
*   **System architect:**
    *   The system architect takes care of the whole architecture of the system
    *   The system architect also decides what is the most efficient architecture for the RB system

We have seen the traditional architecture of the RB system. Now it is time to see what will be the real-life practical architecture of the RB system for NLP applications.

# Practical architecture of the rule-based system for NLP applications

I have already described the general architecture, now we will see the practical architecture of the RB system for NLP applications. Refer to *Figure 7.6*:

![](img/21152cfe-294d-4141-be51-0056e8931963.png)

Figure 7.6: Real life architecture of RB system for NLP application

Let's look at each of the components of the architecture in detail.

Some of the parts, such as domain experts, user interfaces, and system engineer, we have seen in the previous section. So, here, we are focusing on new components:

*   **Knowledge-based editor:**
    *   The domain experts may not know how to code
    *   So we are providing them a knowledge-based editor where they can write or create the rules by using human language
    *   Suppose we are developing a grammar correction system for the English language and we have a linguist who knows how to create rules but doesn't know how to code them
    *   In this case, they can add, update, or delete rules by using the knowledge-based editor
    *   All the created rules are specified in the form of normal human language
*   **Rule translator:**
    *   As we know, all rules are in the form of the human language, so we need to translate or convert them into machine-understandable form
    *   So, the rule translator is the section where pseudo logic for the rules has been defined with examples
    *   Let's consider our grammar correction system example. Here our expert defines a rule if there is a singular subject and plural verb in the sentence, and then changes the verb to the singular verb format
    *   In the rule translator, the defined rule has been converted as if there is a sentence **S** which has a singular subject with the POS tag **PRP$**, **NP** with POS tag of verbs **VBP,** then change the verb to the **VBZ** format. Some examples have also been specified to understand the rules
*   **Rule object classes:**
    *   This rule object class act, as the container for supporting libraries
    *   It contains various prerequisite libraries
    *   It also sometimes contains an optional object class for libraries to optimize the entire system
    *   For the grammar correction system, we can put tools such as parsers, POS taggers, **named entity recognition** (**NER**), and so on in the container to be used by the rule engine
*   **Database or knowledge base:** A database has metadata for rules, such as:
    *   Which supporting libraries have been used from the rule object classes?
    *   What is the category of the rule?
    *   What is priority of the rule?
*   **Rule engine:**
    *   This is the core part, which is the brain of the RB system
    *   By using the rule translator, rule object classes, and knowledge base we need to develop the core code which actually runs on the user query or on the input dataset and generates the output
    *   You can code by using any programming language which is the best fit for your application and its architectures
    *   For our grammar correction system, we will code the rule in this stage and the final code will be put into the rule engine repository

These are all the components that are useful if you are developing an RB system for NLP. Now you must have questions. Can we change the architecture of the system as per our needs? Is it fine? To get answers to these questions, you need to follow the next section.

# Custom architecture - the RB system for NLP applications

According to the needs of different NLP applications, you can change the architecture or components. Customization is possible in this approach. There are some points that need to be taken care of if you are designing a customized RB system architecture. Ask the following questions:

*   Did you analyze and study the problem and the already existing architectures?
    *   Before doing customization, you need to do analysis of your application. If any existing system is there, then study its architecture and take the bad and good out of it
    *   Take enough time for analysis
*   Do you really need custom architecture?
    *   If after the study, you feel that your application architecture needs to be customized, then write down the reasons why you you really need it
    *   State the reasons that you have listed down and can help your system to make it better by asking a series of questions. If yes, then you are on the right track
*   Does it help to streamline the development process?
    *   Does the new architecture actually help your development process better? If it does, then you can consider that architecture
    *   Most of the time, defining a streamline process for developing the RB system is challenging but if your new customized architecture can help you, then it is really a good thing
    *   Does this streamline process actually stabilize your RB system?
*   Is it maintainable?
    *   A customized architecture can help you to maintain the system easily as well as efficiently
    *   If you can add this feature to your customized architecture, then thumbs up!
*   Is it modular?
    *   If it will provide modularity in the RB system then it will be useful because then you can add, remove, or update certain modules easily
*   Is it scalable?
    *   With the help of the new architecture, you can scale up the system. You should also consider this
*   Is it easy to migrate?
    *   If it is with the defined architecture, it should be easy for the team to migrate the system from one platform to another
    *   If we want to migrate a module from one system to another, it should be easy for the technical as well the infrastructure team
*   Is it secure?
    *   System security is a major concern. New architecture should definitely have this feature of security and user privacy if needed
*   Is it easy to deploy?
    *   If you want to deploy some changes in the future, then deployment should be easy
    *   If you want to sell your end product, then the deployment process should be easy enough, which will reduce your efforts and time
*   Is it time saving in terms of development time?
    *   Implementation as well as the development of the RB system by using the architecture should be time saving
    *   The architecture itself should not take too much time to implement
*   Is it easy for our users to use?
    *   The architecture can be complex but for end users it must be user-friendly and easy to use

If you can take all of the preceding points or most of them, then try to implement a small set of problems using the architecture that you think best for the system, then, at the end, ask all the previous questions again and evaluate the output.

If you still get positive answers, then you are good to go! Here, the design is neither right nor wrong; it's all about the best fit for your NLP application.

A **Question-Answering** (**Q/A**) system can use the architecture which is shown in the *Figure 7.7*:

![](img/7441eed7-ab5b-4689-a454-cf5953e265a3.png)

Figure 7.7: Architecture for Question-Answering RB system

You can see a very different kind of architecture. The approach of the Q/A system is an ontology based RB system. Question processing and document processing is the main rule engine for us. Here, we are not thinking of a high-level question answering system. We want to develop a Q/A system for small children who can ask questions about stories and the system will send back the answers as per the rules and available story data.

Let's see each of the components in details:

*   When the user submits the question, the parser parses the question.
*   Parse the question matching the parsing result with the knowledge base, ontology, and keywords thesaurus using the interpreter.
*   Here, we apply the reasoning and facts as well.
*   We derive some facts from the questions and categorized user questions using query classification and reformulation.
*   After, the already-generated facts and categorized queries are sent to the document processing part where the facts are given to the search engine.
*   Answer extraction is the core RB engine for the Q/A system because it uses facts and applies reasoning techniques such as forward chaining or backward chaining to extract all possible answers. Now you will want to know about backward chaining and forward chaining. So, here, I'm giving you just a brief overview. In forward chaining, we start with available data and use inference rules to extract more facts from data until a goal is achieved. This technique is used in expert system to understand what can happen next. And in backward chaining, we start with a list of goals and work backwards to find out which conditions could have happened in the past for the current result. These techniques help us to understand why this happened.
*   Once all possible answers have been generated, then it will be sent back to the user.

I have one question in my mind that I would like to ask you.

What kind of database do you want to select if you develop a Q/A system? Think before you go ahead!

I would like to choose the NoSQL database over the SQL DBs, and there are a couple of reasons behind it. The system should be available for the user 24\7\. Here, we care about our user. The user can access the system anytime, and availability is a critical part. So, I would like to choose the NoSQL database.If, in the future, we want to perform some analytics on the users' questions and answers, then we need to save the users' questions and the system's answers in the database . Read further to understand them:

You can choose your data warehouse or NoSQL DB. If you are new to NoSQL, then you can refer to NoSQL using this link: [https://en.wikipedia.org/wiki/NoSQL,](https://en.wikipedia.org/wiki/NoSQL) and if you are new to the word data warehouse, then you can refer to this link: [https://en.wikipedia.org/wiki/Data_warehouse.](https://en.wikipedia.org/wiki/Data_warehouse) This will help us categorize our users, and we can make some creative changes that really matter to the user. We can also provide customized feed or suggestions to each of the users.

# Exercise

Suppose you are developing a grammar correction system, what kind of system architecture do you design? Try to design it on paper! Let your thoughts come out.

# Apache UIMA - the RB system for NLP applications

In this section, we will look at one of the famous frameworks for the RB system for NLP applications.

Apache UIMA is basically developed by IBM to process unstructured data. You can explore more details by clicking on this link: [https://uima.apache.org/index.html](https://uima.apache.org/index.html)

Here, I want to highlight some points from this framework, which will help you to make your own NLP application using the RB approach.

The following are the features of UIMA:

*   UIMA will provide us with the infrastructure, components, and framework
*   UMIA has an inbuilt RB engine and GATE library for performing preprocessing of text data
*   The following tools are available as part of the components. I have listed down a few of them:
    *   Language identification tool
    *   Sentence segmentation tool
    *   NER tool
*   We can code in Java, Ruta, and C++
*   It is a flexible, modular, and easy-to-use framework
*   C/C++ annotators also supports Python and Perl

Applications/uses of UIMA include:

*   IBM Watson uses UIMA to analyze unstructured data
*   The **clinical Text Analysis and Knowledge Extraction System** (**Apache cTAKES**) uses the UIMA-based system for information extraction from medical records

The challenges of using UIMA include:

*   You need to code rules in either Java, Ruta, or C++. Although, for optimization, many RB systems use C++; getting the best human resources for Ruta is a challenging task
*   If you are new to UIMA, you need some time to become familiar with it

# Understanding the RB system development life cycle

In this section, we will look at the development life cycle for the RB system, which will help you in the future if you want to develop your own. *Figure 7.8* describes the development life cycle of the RB system. This figure is quite self-explanatory, so there is no need for an extra description.

If we follow the stages of the RB development life cycle, then life will be easy for us:

![](img/ac7a6883-ee3f-41db-9ee5-29a1d3f222de.png)

Figure 7.8: RB system development life cycle

# Applications

In this section, I have divided the applications into two sections; one is the NLP application and the other one is the generalized AI application.

# NLP applications using the rule-based system

Here, we mention some of the NLP applications that use the RB system:

*   Sentence boundary detection:
    *   Sentence boundary detection is easy for general English writing but it will be complicated when you are dealing with research papers or other scientific documents
    *   So, handcrafted post-processing rules will help us to identify the sentence boundary accurately
    *   This approach has been used by Grammarly Inc. for the grammar correction system
*   Machine translation:
    *   When we think of a machine translation system, in our mind, we think of the **Google Neural Machine Translation** (**GNMT**) system
    *   For many Indian languages, Google used to use a complex rule-based system with a statistical prediction system, so they have an hybrid system
    *   In 2016, Google launched the neural network based MT system
    *   Many research projects still use the RB system for MT and the majority of them try tapping out the languages which are untapped
*   Template based chatbots:
    *   Nowadays, chatbots are the new trend and craze in the market
    *   A basic version of them is a template-based approach where we have a defined set of questions or keywords and we have mapped the answers to each of the keywords
    *   The good part of this system is matching the keywords. So if you are using any other language but if your chat messages contain keywords which we have defined, then the system is able to send you a proper message as a response
    *   The bad part is, if you make any spelling mistakes then the system will not be able to respond in a proper manner
    *   We will develop this application from scratch. I will explain the coding part in the next section, so keep reading and start your computer!
*   Grammar correction system:
    *   A grammar correction system is also implemented by using rules
    *   In this application, we can define some of the simple rules to very complicated rules as well
    *   In the next section, we will see some of the basic grammar correction rules which we are going to implement using Python
*   Question answering systems:
    *   A question answering system also uses the RB system, but here, there is one different thing going on
    *   The Q/A system uses semantics to get the answer of the submitted question
    *   For putting semantics into the picture, we are using the ontology-based RB approach

# Generalized AI applications using the rule-based system

You have seen the NLP applications which use the RB approach. Now, move into the generalized AI applications, which use the RB approach along with other techniques:

*   Self-driving cars or driver less cars:
    *   At the start of the chapter, I gave the example of the self-driving car to highlight the purpose of having the RB system
    *   The self-driving car also uses a hybrid approach. Many of the big companies, from Google to Tesla, are trying to build self-driving cars, and their experiments are in order to develop the most trustworthy self-driving cars
    *   This application has been developed by using complex RB systems during its initial days
    *   Then, the experiment turned into the direction of ML techniques
    *   Nowadays, companies are implementing deep learning techniques to make the system better
*   Robotics applications:
    *   It has been a long-term goal of the AI community to develop robots which complement human skills
    *   We have a goal where we want to develop robots which help humans to do their work, tasks which are basically time consuming
    *   Suppose there is a robot that helps you with household work. This kind of task can be performed by the robot with the help of defined rules for all possible situations
*   Expert system of NASA:
    *   NASA made the expert system by using the general purpose programming language, **CLIPS** (**C Language Integrated Production System**)

Now, I think that's enough of theories. Now we should try to develop some of the RB applications from scratch. Get ready for coding. We will begin our coding journey in the next section.

# Developing NLP applications using the RB system

In this section, we will see how to develop NLP applications using the RB system. We are developing applications from the beginning. So, first you need the following dependencies.

You can run the following command to install all the dependencies:

```py
 pip install -r pip-requirements.txt 

```

The list of dependencies can be found by clicking on this link: [https://github.com/jalajthanaki/NLPython/blob/master/pip-requirements.txt](https://github.com/jalajthanaki/NLPython/blob/master/pip-requirements.txt)

# Thinking process for making rules

We are talking a lot about rules, but how can these rules actually can be derived? What is the thinking process of a linguist when they are deriving rules for an NLP application? Then, let's begin with this thinking process.

You need to think like a linguist for a while. Remember all the concepts that you have learned so far in this book and be a linguist.

Suppose you are developing rules for a grammar correction system, especially for the English language. So, I'm describing the thought process of a linguist and this thought process helps you when you are developing rules:

*   What should I need to know?
    *   You should know about grammatical rules of the language for which you are creating rules, here that language is English
    *   You should know the structure, word order, and other language related concepts
    *   The preceding two points are prerequisites
*   From where should I start?
    *   If you know all the language-related things, then you need to observe and study incorrect sentences
    *   Now, when you study incorrect sentences, you need to know what mistakes there are in the sentences
    *   After that you need to think about the categories of the mistakes, whether the mistakes are syntax related, or whether they are because of semantic ambiguity
    *   After all this, you can map your language-related knowledge to the mistakes in the sentences
*   How can rules be derived?
    *   Once you find the mistakes in the sentence, then at that moment focus on your thinking process. What does your brain think when you're capturing the mistakes?
    *   Think about how your brain reacts to each of the mistakes that you have identified
    *   You can capture the mistake because you know the grammatical facts of the language or other language related technical stuff (sentence syntax structures, semantics knowledge, and so on). Your brain actually helps you
    *   Your brain knows the right way to interpret the given text using the given language
    *   That is the reason you are able to catch the mistakes. At the same time, you have some solid reason; based on that, you have identified the mistakes
    *   Once you have identified the mistakes, as per the different categories of the mistakes, you can correct the mistakes by changing some parts of the sentences using certain logical rules
    *   You can change the word order, or you can change the subject verb agreement, or you can change some phrases or all of them together
    *   Bingo! At this point, you will get your rule. You know what the mistakes are and you also know what are these steps are for converting incorrect sentences to correct sentences
    *   Your rule logic is nothing but the steps of converting incorrect sentences into correct sentences
*   What elements do I need to take care of?
    *   First, you need to think about a very simple way of correcting the mistakes or incorrect sentences
    *   Try to make pattern-based rules
    *   If pattern-based rule are not possible to derive then check if you can use parsing and/or morphological analyzer results and then check other tools and libraries
    *   By the way, there is one catch here. When you defining rules, you also need to think about how feasible the rule logic is for implementation
    *   Are the tools available or not? If the tools are available then you can code your rules or the developer can code the rules
    *   If the tools aren't available then you need to discard your rules
    *   Research is involved when you define a rule and then check whether there are any tools available which coders can use for coding up the defined rule logic
    *   The selected tools should be capable of coding the exceptional scenarios for rules
    *   Defining rules and researching on tools can be basic tasks for linguists if you have some linguists in your team. If not, then you as coders need to search tools which you can use for coding the rule logic

Without any delay, we will start coding.

# Start with simple rules

I have written a script which scrapes the Wikipedia page entitled Programming language.

Click here to open that page: [https://en.wikipedia.org/wiki/Programming_language](https://en.wikipedia.org/wiki/Programming_language)

Extracting the name of the programming languages from the text of the given page is our goal. Take an example: The page has C, C++, Java, JavaScript, and so on, programming languages. I want to extract them. These words can be a part of sentences or have occurred standalone in the text data content.

Now, see how we can solve this problem by defining a simple rule. The GitHub link for the script is: [https://github.com/jalajthanaki/NLPython/blob/master/ch7/7_1_simplerule.py](https://github.com/jalajthanaki/NLPython/blob/master/ch7/7_1_simplerule.py)

The data file link on GitHub is: [https://github.com/jalajthanaki/NLPython/blob/master/data/simpleruledata.txt](https://github.com/jalajthanaki/NLPython/blob/master/data/simpleruledata.txt)

Here our task can be divided into three parts:

*   Scraping the text data
*   Defining the rule for our goal
*   Coding our rule and generating the prototype and result

# Scraping the text data

In this stage, we are going to scrape the text from the programming language wiki page and export the content into a text file. You can see the code snippet in *Figure 7.9*:

![](img/7862c8ec-1953-4c0e-91ea-8c1fd703632e.png)

Figure 7.9: Code snippets for scraping text data

The output of the scraping data is shown in *Figure 7.10*:

![](img/b0c90cf2-7310-4035-872c-47f7f48fe65e.png)

Figure 7.10: Output of scraping script

# Defining the rule for our goal

Now, if you look at our scraped data, you can find the sentences. Now after analyzing the text, you need to define a rule for extracting only programming language names such as Java, JavaScript, MATLAB, and so on. Then, think for a while about what kind of simple rule or logic can help you to achieve your goal. Think hard and take your time! Try to focus on your thinking process and try to find out the patterns.

If I wanted to define a rule, then I would generalize my problem in the context of the data given to me. During my analysis, I have noticed that the majority of the programming language keywords come with the word language. I have noticed that when language as a word appears in the sentences, then there is a high chance that the actual programming language name also appears in that sentence. For example, the C programming language is specified by an ISO standard. In the given example, the C programming language appears and the word language also appears in the sentence. So, I will perform the following process.

First, I need to extract the sentences which contain language as a word. Now as a second step, I will start to process the extracted sentences and check any capitalized words or camel case words there are in the sentence. Then, if I find any capitalized words or camel case words, I need to extract them and I will put them into the list because most of the programming languages appear as capitalized words or in camel case word format. See the examples: C, C++, Java, JavaScript, and so on. There will be cases where a single sentence contains the name of more than one programming language.

The preceding process is our rule and the logical form of the rule is given here:

*   Extract sentences with language as a word
*   Then try to find out words in the sentence which are in camel case or capitalized form
*   Put all these words in a list
*   Print the list

# Coding our rule and generating a prototype and result

This example gives you the practical essence of the rule making process. This is our first step so we are not focusing on accuracy very much. I know, this is not the only way of solving this problem and this is not the most efficient way. There are also other efficient ways to implement the same problem, but I'm using this one because I felt this solution is the simplest one and easiest to understand.

This example can help you to understand how rules can be coded and, after getting the result of the first prototype, what next steps you can take to improve your output.

See the code snippet in *Figure 7.11*:

![](img/b7e81518-af58-4790-8404-d4ab270cdd14.png)

Figure 7.11: Code for implementation of rule logic for extracting programming language

The output for the preceding code snippet is as follows:

```py
['A', 'Programming', 'The', 'Musa', 'Baghdad,', 'Islamic', 'Golden', 'Age".[1]', 'From', 'Jacquard', 'Thousands', 'Many', 'The', 'Some', 'C', 'ISO', 'Standard)', 'Perl)', 'Some', 'A', 'Some,', 'Traits', 'Markup', 'XML,', 'HTML,', 'Programming', 'XSLT,', 'Turing', 'XML', 'Moreover,', 'LaTeX,', 'Turing', 'The', 'However,', 'One', 'In', 'For', 'Another', 'John', 'C.', 'Reynolds', 'He', 'Turing-complete,', 'The', 'The', 'Absolute', 'The', 'These', 'The', 'An', 'Plankalk\xc3\xbcl,', 'German', 'Z3', 'Konrad', 'Zuse', 'However,', 'John', "Mauchly's", 'Short', 'Code,', 'Unlike', 'Short', 'Code', 'However,', 'At', 'University', 'Manchester,', 'Alick', 'Glennie', 'Autocode', 'A', 'The', 'Mark', 'University', 'Manchester', 'The', 'Mark', 'R.', 'A.', 'Brooker', 'Autocode".', 'Brooker', 'Ferranti', 'Mercury', 'University', 'Manchester.', 'The', 'EDSAC', 'D.', 'F.', 'Hartley', 'University', 'Cambridge', 'Mathematical', 'Laboratory', 'Known', 'EDSAC', 'Autocode,', 'Mercury', 'Autocode', 'A', 'Atlas', 'Autocode', 'University', 'Manchester', 'Atlas', 'In', 'FORTRAN', 'IBM', 'John', 'Backus.', 'It', 'It', 'Another', 'Grace', 'Hopper', 'US,', 'FLOW-MATIC.', 'It', 'UNIVAC', 'I', 'Remington', 'Rand', 'Hopper', 'English', 'The', 'FLOW-MATIC', 'Flow-Matic', 'COBOL,', 'AIMACO', 'The', 'These', 'The', 'Each', 'The', 'Edsger', 'Dijkstra,', 'Communications', 'ACM,', 'GOTO', 'The', 'C++', 'The', 'United', 'States', 'Ada,', 'Pascal', 'In', 'Japan', 'The', 'ML', 'Lisp.', 'Rather', 'One', 'Modula-2,', 'Ada,', 'ML', 'The', 'Internet', 'Perl,', 'Unix', 'Java', 'Pascal', 'These', 'C', 'Programming', 'Current', "Microsoft's", 'LINQ.', 'Fourth-generation', 'Fifth', 'All', 'These', 'A', 'Most', 'On', 'The', 'The', 'Since', 'Programming', 'Backus\xe2\x80\x93Naur', 'Below', 'Lisp:', 'Not', 'Many', 'In', 'Even', 'Using', 'The', 'C', 'The', 'Chomsky', 'The', 'Type-2', 'Some', 'Perl', 'Lisp,', 'Languages', 'In', "Lisp's", "Perl's", 'BEGIN', 'C', 'The', 'The', 'For', 'Examples', 'Many', 'Other', 'Newer', 'Java', 'C#', 'Once', 'For', 'The', 'There', 'Natural', 'A', 'Results', 'A', 'The', 'Any', 'In', 'In', 'The', 'A', 'For', 'The', 'Many', 'A', 'These', 'REXX', 'SGML,', 'In', 'High-level', 'BCPL,', 'Tcl,', 'Forth.', 'In', 'Many', 'Statically', 'In', 'In', 'Most', 'C++,', 'C#', 'Java,', 'Complete', 'Haskell', 'ML.', 'However,', 'Java', 'C#', 'Additionally,', 'Dynamic', 'As', 'Among', 'However,', 'Lisp,', 'Smalltalk,', 'Perl,', 'Python,', 'JavaScript,', 'Ruby', 'Strong', 'An', 'Strongly', 'An', 'Perl', 'JavaScript,', 'In', 'JavaScript,', 'Array,', 'Such', 'Strong', 'Some', 'Thus', 'C', 'Most', 'Core', 'The', 'In', 'However,', 'Indeed,', 'For', 'Java,', 'Smalltalk,', 'BlockContext', 'Conversely,', 'Scheme', 'Programming', 'But', 'A', 'By', 'While', 'Many', 'Many', 'Although', 'The', 'One', 'The', 'As', 'Because', 'This', 'Natural', 'However,', 'Edsger', 'W.', 'Dijkstra', 'Alan', 'Perlis', 'Hybrid', 'Structured', 'English', 'SQL.', 'A', 'The', 'The', 'A', 'An', 'There', 'It', 'Although', 'Proprietary', 'Some', 'Oracle', 'Corporation', 'Java', "Microsoft's", 'C#', 'Common', 'Language', 'Runtime', 'Many', 'MATLAB', 'VBScript.', 'Some', 'Erlang', "Ericsson's", 'Thousands', 'Software', 'Programming', 'When', 'However,', 'The', 'On', 'A', 'A', 'These', 'Programming', 'Programs', 'In', 'When', 'Unix', 'It', 'One', 'CPU', 'Some', 'For', 'COBOL', 'Fortran', 'Ada', 'C', 'Other', 'Various', 'Combining', 'C,', 'Java,', 'PHP,', 'JavaScript,', 'C++,', 'Python,', 'Shell,', 'Ruby,', 'Objective-C', 'C#.[70]', 'There', 'A', 'Languages', 'Ideas', 'The', 'For', 'Java', 'Python', 'In', 'Traditionally,', 'These', 'A', 'More', 'An', 'By', 'Some', 'A', 'For', 'English', 'Other'] 

```

Now as you have seen, our basic rule extracted programming languages but it has also extracted junk data. Now think how you can restrict the rule or how you can put in some constraints so it will give us an accurate output. That will be your assignment.

# Exercise

Please improvise the preceding output by putting in some constraints (Hint: You can apply some preprocessing and regex can also help you.)

# Python for pattern-matching rules for a proofreading application

Now, suppose you want to make a proofreading tool. So, here I will provide you with one very simple mistake that you can find easily in any business mail or in any letter. Then we will try to correct the errors with high accuracy.

The mistake is when people specify a meeting timing in their mail, they may have specified the time as 2pm, or as 2PM, or as 2P.M., or other variations, but the correct format is 2 p.m. or 9 a.m.

This mistake can be fixed by the pattern-based rule. The following is the rule logic.

Suppose the numeric digit of length two starts from 1 to 12\. After this numeric digit, if `am` and `pm` occurred without a space or without a period, then add the space and the proper period symbol.

I will implement it by using a regular expression.

Source pattern:

```py
\b([1-9]|0[1-9]|1[0-2]{1,2})(am)\b 
\b([1-9]|0[1-9]|1[0-2]{1,2})(pm)\b 

```

Target pattern:

```py
r'\b([1-9]|0[1-9]|1[0-2]{1,2})(am)\b', r'\1 a.m.'  
r'\b([1-9]|0[1-9]|1[0-2]{1,2})(pm)\b', r'\1 p.m.' 

```

You can find the code on the GitHub URL at: [https://github.com/jalajthanaki/NLPython/blob/master/ch7/7_2_basicpythonrule.py](https://github.com/jalajthanaki/NLPython/blob/master/ch7/7_2_basicpythonrule.py)

The code snippet is given in *Figure 7.12*:

![](img/3f42586d-dc3e-4282-a65c-ebba44d8f8e4.png)

Figure 7.12: Code snippet for pattern-based rule

The output of the preceding code snippet is:

![](img/90ee2ddc-9ba7-454a-8597-7a7f5b171005.png)

Figure 7.13: Output of pattern-based rule

The given example is a basic one, but it helps you to think about how proofreading can be done. Many simple sets of rules can be applied on the data and, according to the pattern, you will get the corrected result.

# Exercise

Write a similar kind of rule which helps in correcting the timing pattern 11:30am or 5:45pm to 11:30 a.m. or 5:45 p.m.

# Grammar correction

We will make a simple rule about a subject verb agreement rule for simple present tense.

We know that in simple present tense the third-person singular subjects always takes a singular verb with either s/es as the suffix of the verb.

Here are some examples of incorrect sentences:

*   He drink tomato soup in the morning
*   She know cooking
*   We plays game online

We cannot perform a pattern-based correction for these kinds of incorrect sentences. Here, to make a rule, we will parse each sentence and try to check by using the parser result. Can we make any rules? I have parsed sentences to generate the parse result so you can see the parse tree in *Figure 7.14*. This result has been generated by using the Stanford parser:

![](img/864c3aeb-7b89-473a-bad5-cf19fa6b45e7.png)

Figure 7.14: Parsing result for example sentences

We need to first extract the **NP**, which either takes the pronouns **PRP**/**NNP** or **NN**. This rule can be restricted to **PRP** only. We can extract the **PRP** tags from the sentence. After that we need to extract the **VP**. By using the type of pronoun and **VP**, we can suggest the change to the user. I guess you guys remember **NP**, **PRP**, **NNP**, and so on. As we have already shown, these are all kinds of POS tags, in [Chapter 5](07f71ca1-6c8a-492d-beb3-a47996e93f04.xhtml), *Feature Engineering and NLP Algorithm*.

Rule logic:

*   Extract the **NP** with the **PRP** tag
*   Extract the **VP**
*   As per the **PRP**, perform the correction in ****VP****

Let's do the coding for this:

I have installed the Stanford-`corenlp` and `pycornlp` libraries. You have already learned the steps for installing the Stanford parser in [Chapter 5](07f71ca1-6c8a-492d-beb3-a47996e93f04.xhtml), *Feature Engineering and NLP Algorithm.*You guys are going to code this. So, it's a complete code challenge. I have a code in which I have extracted the pattern for you for **PRP** and **VBZ**/**VBP**. Your task is to check whether the combination of **PRP** and **VBP**/**VBZ** is right or wrong. If it is wrong, then raise an alert. You can find the code at: [https://github.com/jalajthanaki/NLPython/blob/master/ch7/7_3_SVArule.py](https://github.com/jalajthanaki/NLPython/blob/master/ch7/7_3_SVArule.py)

You can see the code snippet in *Figure 7.15* and *Figure 7.16*:

![](img/bf229a86-fb2e-48e6-9fb4-efd8833a65c0.png)

Figure 7.15: Stated Stanford corenlp server

I have given you the code but you need to complete it:

![](img/a88b6724-67f7-4890-a3fa-59d1c7ac34b5.png)

Figure 7.16: Code which I have given to you but you need to complete

You can see the output of my incomplete code in *Figure 7.17*:

![](img/1083e9d0-41b5-4bb3-be1a-424cfc7d9d37.png)

Figure 7.17: Output of my incomplete code

# Template-based chatbot application

Here, we will see how we can build a core engine for a chatbot application which can help a loan applicant to apply for the same. We are generating output in JSON format, so any front-end developer can integrate this output on a website.

Here, I'm using the flask web framework and making web services for each question that our chatbot asks.

You need to install MongoDB if you want to save the user data. The installation steps of MongoDB are in this link: [https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)

I have defined functions in the `conversationengine.py` file. The path of this file on GitHub is: [https://github.com/jalajthanaki/NLPython/blob/master/ch7/chatbot/customscripts/conversationengine.py](https://github.com/jalajthanaki/NLPython/blob/master/ch7/chatbot/customscripts/conversationengine.py)

You can see the flask web engine code in the `flaskengin.py` file. The GitHub link is: [https://github.com/jalajthanaki/NLPython/blob/master/ch7/chatbot/customscripts/conversationengine.py](https://github.com/jalajthanaki/NLPython/blob/master/ch7/chatbot/customscripts/conversationengine.py)

The whole folder and package file path is at: [https://github.com/jalajthanaki/NLPython/tree/master/ch7/chatbot](https://github.com/jalajthanaki/NLPython/tree/master/ch7/chatbot)

# Flow of code

So, I have written functions in `conversationengine.py` which generate a JSON response according to the questions you have asked and this JSON response can be used by the frontend developer team to display messages on the chatbot UI.

Then, I have written a web service using flask so you can see the JSON response on the web URL specified in JSON itself.

The `conversationengine.py` is the core rule engine with handcrafted rules and codes. See the code snippet in *Figure 7.18*:

![](img/a0fd264e-9b32-44b0-bde0-cfd089c3c676.png)

Figure 7.18: Code snippet of conversationengine.py

Here, we have used a keywords list and, responses list to implement chatbot. I have also made customized JSON schema to export the conversation and, if you are from a web development background then you can write JavaScript which will help you to display this JSON on the front end with GUI.

Now, let's look at the web services part in *Figure 7.19*:

![](img/a0ad22d9-1614-4e27-a5e0-8c4bfc45dde6.png)

Figure 7.19: Flask web service URLs defined in flaskengin.py

Now, to run the scripts and see the output follow these steps:

1.  First run `flaskengin.py`
2.  Go to the URL: `http://0.0.0.0:5002/`, where you can see `Hello from chatbot Flask!`
3.  You can see the chatbot JSON response by using this URL: `http://0.0.0.0:5002/welcomemsg_chat`
4.  You can see the JSON response in *Figure 7.20*:![](img/266f8f6f-8f8d-4be6-af88-f0d1492a702e.png)

Figure 7.20: JSON response of chatbot

5.  Now, we are providing suggestions to our human user which will help them analyze what the expected input from them is. So, here, you can see the JSON attribute `suggestion_message: ["Hi"]`. So, the user will see the button with the `Hi` label.
6.  If you want to redirect to the next page or next question, then use `next_form_action` URL and put the user argument after `msg = USER ARGUMENT`
7.  For example, I am at the `http://0.0.0.0:5002/welcomemsg_chat` page. Now, you can read the `message_bot`. It says you need to say `Hi to bot`
8.  You can give your `Hi` response like this: `http://0.0.0.0:5002/hi_chat?msg=Hi`
9.  When you are on this URL: `http://0.0.0.0:5002/hi_chat?msg=Hi` you can see the bot will ask for your name now you need to enter your name.
10.  To enter your name and be redirected to the next question, you need to again check what is the value of the URL for the `next_form_action` attribute
11.  Here the value is `/asking_borowers_email_id?msg=`
12.  You need to put your name after the `=` sign so the URL becomes `/asking_borowers_email_id?msg=Jalaj Thanaki`
13.  When you use `http://0.0.0.0:5002/asking_borowers_full_name?msg=Jalaj%20Thanaki`, you can see next question from the bot.
14.  First you need to run the script: [https://github.com/jalajthanaki/NLPython/blob/master/ch7/chatbot/flaskengin.py](https://github.com/jalajthanaki/NLPython/blob/master/ch7/chatbot/flaskengin.py) and then you can check the following URLs:
    *   `http://0.0.0.0:5002/welcomemsg_chat`
    *   `http://0.0.0.0:5002/hi_chat?msg=Hi`
    *   `http://0.0.0.0:5002/asking_borowers_full_name msg=Jalaj%20Thanaki`
    *   `http://0.0.0.0:5002/asking_borowers_email_id?msg=jalaj@gmail.com`
    *   `http://0.0.0.0:5002/mobilenumber_asking?msg=9425897412`
    *   `http://0.0.0.0:5002/loan_chat?msg=100000`
    *   `http://0.0.0.0:5002/end_chat?msg=Bye`

If you want to insert user data in the MongoDB database, then this is possible and is included in the code but commented.

# Advantages of template-based chatbot

*   Easy to implement.
*   Time and cost efficient.
*   Use cases are understood prior to development so user experience will also be good.
*   This is a pattern-matching approach, so if users use English and other languages in their conversation then users also get answers because chatbot identifies keywords which he provides in English, and if English keywords match with the chatbot vocabulary, then chatbot can give you answer.

# Disadvantages of template-based chatbot

*   It cannot work for unseen use cases
*   User should process a rigid flow of conversation
*   Spelling mistakes by users create a problem for chatbot. In this case, we will use deep learning

# Exercise

Develop a template-based chatbot application for a hotel room booking customer support service. Develop some questions and answers and develop the application.

# Comparing the rule-based approach with other approaches

The rule-based approach is a very reliable engine which provides your application with high accuracy. When you compare the RB approach with ML approaches or deep learning approaches, you will find the following points:

*   For the RB approach, you need a domain expert, while for the ML approach, or for the deep learning approach, you don't need a domain expert
*   The RB system doesn't need a large amount of data, whereas ML and deep learning need a very large amount of data
*   For the RB system, you need to find patterns manually, whereas ML and deep learning techniques find patterns on your behalf as per the data and input features
*   The RB system is often a good approach for developing the first cut of your end product, which is still popular in practice

# Advantages of the rule-based system

There are very good advantages to using RB system. The advantages are mentioned as follows:

*   Availability: Availability of the system for the user is not an issue
*   Cost efficient: This system is cost efficient and accurate in terms of its end result
*   Speed: You can optimize the system as you know all the parts of the system. So to provide output in a few seconds is not a big issue
*   Accuracy and less error rate: Although coverage for different scenarios is less, whatever scenarios are covered by the RB system will provide high accuracy. Because of these predefined rules, the error rate is also less
*   Reducing risk: We are reducing the amount of risk in terms of system accuracy
*   Steady response: Output which has been generated by the system is dependent on rules so the output responses are stable, which means it cannot be vague
*   The same cognitive process as a human: This system provides you with the same result as a human, as it has been handcrafted by humans
*   Modularity: The modularity and good architecture of the RB system can help the technical team to maintain it easily. This decreases human efforts and time
*   Uniformity: The RB system is much uniformed in its implementation and its output. This makes life easy for the end user because the output of the system can be easily understood by humans
*   Easy to implement: This approach mimics the human thought process, so the implementation of rules is comparatively easy for developers

# Disadvantages of the rule-based system

The disadvantages of the RB system are as follows:

*   Lot of manual work: The RB system demands deep knowledge of the domain as well as a lot of manual work
*   Time consuming: Generating rules for a complex system is quite challenging and time consuming
*   Less learning capacity: Here, the system will generate the result as per the rules so the learning capacity of the system by itself is much less
*   Complex domains: If an application that you want to build is too complex, building the RB system can take lot of time and analysis. Complex pattern identification is a challenging task in the RB approach

# Challenges for the rule-based system

Let's look at some of the challenges in the RB approach:

*   It is not easy to mimic the behavior of a human.
*   Selecting or designing architecture is the critical part of the RB system.
*   In order to develop the RB system, you need to be an expert of the specific domain which generates rules for us. For NLP we need linguists who know how to analyze language.
*   Natural language is itself a challenging domain because it has so many exception cases and covering those exceptions using rules is also a challenging task, especially when you have a large amount of rules.
*   Arabic, Gujarati, Hindi, and Urdu are difficult to implement in the RB system because finding a domain expert for these languages is a difficult task. There are also less tools available for the described languages to implement the rules.
*   Time consumption of human effort is too high.

# Understanding word-sense disambiguation basics

**Word-sense disambiguation** (**WSD**) is a well-known problem in NLP. First of all, let's understand what WSD is. WSD is used in identifying what the sense of a word means in a sentence when the word has multiple meanings. When a single word has multiple meaning, then for the machine it is difficult to identify the correct meaning and to solve this challenging issue we can use the rule-based system or machine learning techniques.

In this chapter, our focus area is the RB system. So, we will see the flow of how WSD is solved. In order to solve this complex problem using the RB system, you can take the following steps:

*   When you are trying to solve WSD for any language you need to have a lot of data where you can find the various instances of words whose meaning can be different from sentence to sentence
*   Once you have this kind of dataset available, then human experts come into the picture
*   Human experts are used to tag the meaning of a word or words and usually the tags have some predefined IDs. Now, let's take an example: I have the sentences: I went to river bank, and I went to bank to deposit my money.
*   In the preceding sentences, the word bank has multiple meanings and the meaning changes as per the overall sentence. So, the human expert is used to tag these kinds of words. Here, our word is bank
*   So, the human expert tags the word bank in the river bank sense by using a predefined ID. Assume for now that the ID is 100
*   In the second sentence, the word bank is tagged as a financial institution by using the predefined ID. Assume for now that ID is 101
*   Once this tag has been given then the next stage has been started, which is either to choose rule-based engine or supervised machine learning techniques
*   If we decide to go with the rule-based system then human experts need to come up with a certain pattern or rules which help us to disambiguate the sense of the words. Sometimes, for some words, the expert can find the rule by using a parsing result or by using POS tags, but in most case they can't
*   So nowadays, once tagging has been done, the tagged data is used as input to develop a supervised machine learning model which helps humans to identify the senses of the words
*   Sometimes only the rule-based system cannot work in the same way only the machine learning approach alone sometimes can't help you. Here is the same kind of case according to my experience. I think the hybrid approach will give you a better result
*   After tagging the data, we should build the RB system which handles known situations very well and we also have a situation where we can't define rules. To solve that situation, we need to build a machine learning model.
*   You can also use the vectorization concept and deep learning model to solve WSD problems. Your findings on WSD by using deep learning can be a research topic as well.

# Discussing recent trends for the rule-based system

This section is a discussion about how the current market is using the RB system. So many people are asking many questions on different forums and they want to know about the future of the RB system, so I want to discuss with you one important question which will help you to learn the future trends of the NLP market and RB system. I have some of the questions that we will look at.

Are RB systems outdated in the NLP domain? I would like to answer this with NO. The RB system has been used majorly in all NLP applications, grammar correction, speech recognition, machine translation, and so on! This approach is the first step when you start making any new NLP application. If you want to experiment on your idea, then prototypes can be easily developed with the help of the RB approach. For prototyping, you need domain knowledge and basic coding skills. You don't need to know high-level mathematics or ML techniques. For basic prototyping, you should go with the RB system.

Can deep learning and ML-based approaches replace the RB based system? This question is quite an open-ended question. I would like to present some facts at this point which will help you to derive your question. Nowadays, we are flooded with data and we have cheap computation power available to us. The AI industry and AI-based projects are creating a lot of buzz. The preceding two points help deep learning and ML approaches to derive accurate results for NLP as well as other AI applications. These approaches need less human effort compared to the RB system. This is the reason why so many people think that the RB system will not be replaced by the deep learning and ML-based systems. I would argue that the RB system is not going to be replaced totally, but it will complement these approaches. Now you ask, how? So, the answer is, I think I would like to go with hybrid approaches which are much more beneficial for us. We can find patterns or predictions with the help of the ML system and then give those predictions to the RB system, and the RB system can validate the prediction and choose the best one for the users. This will actually help us to overcome one major challenge of the RB system, which is the reduction of human effort and time.

For the preceding questions, there is not any right or wrong answers. It is all about how you can see the questions and NLP domain. I just want to leave a thought for you. Think by yourself and try to come up with your own answer.

# Summary

In this chapter, we have seen all the details related to the rule-based system and how the rule-based approach helps us to develop rapid prototypes for complex problems with high accuracy. We have seen the architecture of the rule-based system. We have learned about the advantages, disadvantages, and challenges for the rule-based system. We have seen how this system is helpful to us for developing NLP applications such as grammar correction systems, chatbots, and so on. We have also discussed the recent trends for the rule-based system.

In the next chapter, we will learn the other main approaches called machine learning, to solve NLP applications. The upcoming chapter will give you all the details about which machine learning algorithms you need to use for developing NLP applications. We will see supervised ML, semi-supervised ML, and unsupervised ML techniques. We will also develop some of the applications from scratch. So keep reading!

This self-driving car exam