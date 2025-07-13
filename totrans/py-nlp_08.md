

# Machine Learning for NLP Problems

We have seen the basic and the advanced levels of feature engineering. We have also seen how rule-based systems can be used to develop NLP applications. In this chapter, we will develop NLP applications, and to develop the applications, we will use **machine learning** (**ML**) algorithms. We will begin with the basics of ML. After this, we will see the basic development steps of NLP applications that use ML. We will mostly see how to use ML algorithms in the NLP domain. Then, we will move towards the features selection section. We will also take a look at hybrid models and post-processing techniques.

This is the outline of this chapter given as follows:

*   Understanding the basics of machine learning
*   Development steps for NLP application
*   Understanding ML algorithms and other concepts
*   Hybrid approaches for NLP applications

Let's explore the world of ML!

# Understanding the basics of machine learning

First of all, we will understand what machine learning is. Traditionally, programming is all about defining all the steps to reach a certain predefined outcome. During this process of programming, we define each of the minute steps using a programming language that help us achieve our outcome. To give you a basic understanding, I'll take a general example. Suppose that you want to write a program that will help you draw a face. You may first write the code that draws the left eye, then write the code that draws the right eye, then the nose, and so on. Here, you are writing the code for each facial attribute, but ML flips this approach. In ML, we define the outcome and the program learns the steps to achieve the defined output. So, instead of writing code for each facial attribute, we provide hundreds of samples of human faces to the machine. We expect the machine to learn the steps that are needed to draw a human face so that it can draw some new human faces. Apart from this, when we provide the new human face as well as some animal face, it should recognize which face looks like a human face.

Let's take some general examples. If you want to recognize the valid license plates of certain states, in traditional programming, you need to write code such as what the shape of the license plate should be, what the color should be, what the fonts are, and so on. These coding steps are too lengthy if you are trying to manually code each single property of the license plate. Using ML, we will provide some example license plates to the machine and the machine will learn the steps so that it can recognize the new valid license plate.

Let's assume that you want to make a program that can play the game Super Mario and win the game as well. So, defining each game rule is too difficult for us. We usually define a goal such as you need to get to the endpoint without dying and the machine learns all the steps to reach the endpoint.

Sometimes, problems are too complicated, and even we don't know what steps should possibly be taken to solve these problems. For example, we are a bank and we suspect that there are some fraudulent activities happening, but we are not sure how to detect them or we don't even know what to look for. We can provide a log of all the user activities and find the users who are not behaving like the rest of the users. The machine learns the steps to detect the anomalies by itself.

ML is everywhere on the internet. Every big tech company is using it in some way. When you see any YouTube video, YouTube updates or provides you with suggestions of other videos that you may like to watch. Even your phone uses ML to provide you with facilities such as iPhone's Siri, Google Assistance, and so on. The ML field is currently advancing very fast. Researchers use old concepts, change some of them, or use other researchers, work to make it more efficient and useful.

Let's look at the basic traditional definition of ML. In 1959, a researcher named Arthur Samuel gave computers the ability to learn without being explicitly programmed. He evolved this concept of ML from the study of pattern recognition and computational learning theory in AI. In 1997, Tom Mitchell gave us an accurate definition that has been useful to those who can understand basic math. The definition of ML as per Tom Mitchell is: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

Let's link the preceding definition with our previous example. To identify a license plate is called task **T**. You will run some ML programs using examples of license plates called experience **E**, and if it successfully learns, then it can predict the next unseen license plate that is called performance measure **P**. Now it's time to explore different types of ML and how it's related to AI.

# Types of ML

In this section, we will look at different types of ML and some interesting sub-branch and super-branch relationships.

ML itself is derived from the branch called a**rtificial intelligence**. ML also has a branch that is creating lot of buzz nowadays called **deep learning**, but we will look at artificial intelligence and deep learning in detail in [Chapter 9](f414d38e-b88e-4239-88bd-2d90e5ce67ab.xhtml), *Deep Learning for NLP and NLG Problems*.

Learning techniques can be divided into different types. In this chapter, we are focusing on ML. Refer to *Figure 8.1*:

![](img/b5760145-4ef9-4fb6-bc37-9bf1acdf7368.png)

Figure 8.1: Subset and superset relationships of ML with other branches (Image credit: https://portfortune.files.wordpress.com/2016/10/ai-vs-ml.jpg)

ML techniques can be divided into three different types, which you can see in *Figure 8.2*:

![](img/94fa2c30-86c3-49b0-9044-c2b78556e670.png)

Figure 8.2: Three types of ML (Image credit: https://cdn-images-1.medium.com/max/1018/1*Yf8rcXiwvqEAinDTWTnCPA.jpeg)

We will look at each type of ML in detail. So, let's begin!

# Supervised learning

In this type of ML, we will provide a labeled dataset as input to the ML algorithm and our ML algorithm knows what is correct and what is not correct. Here, the ML algorithm learns mapping between the labels and data. It generates the ML model and then the generated ML model can be used to solve some given task.

Suppose we have some text data that has labels such as spam emails and non-spam emails. Each text stream of the dataset has either of these two labels. When we apply the supervised ML algorithm, it uses the labeled data and generates an ML model that predicts the label as spam or non-spam for the unseen text stream. This is an example of supervised learning.

# Unsupervised learning

In this type of ML, we will provide an unlabeled dataset as input to the ML algorithm. So, our algorithm doesn't get any feedback on what is correct or not. It has to learn by itself the structure of the data to solve a given task. It is harder to use an unlabeled dataset, but it's more convenient because not everyone has a perfectly labeled dataset. Most data is unlabeled, messy, and complex.

Suppose we are trying to develop a summarization application. We probably haven't summarized the documents corresponding to the actual document. Then, we will use raw and the actual text document to create a summary for the given documents. Here, the machine doesn't get any feedback as to whether the summary generated by the ML algorithm is right or wrong. We will also see an example of a computer vision application. For image recognition, we feed an unlabeled image dataset of some cartoon characters to the machine, and we expect the machine to learn how to classify each of the characters. When we provide an unseen image of a cartoon character, it should recognize the character and put that image in the proper class, which is generated by the machine itself.

# Reinforcement learning

The third type of ML is reinforcement learning. Here, the ML algorithm doesn't give you the feedback right after every prediction, but it generates feedback if the ML model achieves its goal. This type of learning is mostly used in the area of robotics and to develop intelligent bots to play games. Reinforcement learning is linked to the idea of interacting with an environment using the trial and error method. Refer to *Figure 8.3*:

![](img/cc5fc564-d9fd-4745-9373-b79b16a3fd5d.png)

Figure 8.3: Reinforcement learning interacting with environment (Image credit: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/04/aeloop-300x183.png)

To learn the basics, let's take an example. Say you want to make a bot that beats humans at chess. This type of bot would receive feedback only if it won the game. Recently, Google AlphaGo beat the world's best Go player. If you want to read more on this, refer to the following link:

[https://techcrunch.com/2017/05/24/alphago-beats-planets-best-human-go-player-ke-jie/.](https://techcrunch.com/2017/05/24/alphago-beats-planets-best-human-go-player-ke-jie/)

We are not going into detail about this type of ML in this book because our main focus is NLP, not robotics or developing a game bot.

If you really want to learn **reinforcement learning** (**RL**) in detail, you can take up this course:
[https://www.udacity.com/course/reinforcement-learning--ud600](https://in.udacity.com/course/reinforcement-learning--ud600/).

I know you must be interested in knowing the differences between each type of ML. So, pay attention as you read the next paragraph.

For supervised learning, you will get feedback after every step or prediction. In reinforcement learning, we will receive feedback only if our model achieves the goal. In unsupervised learning, we will never get feedback, even if we achieve our goal or our predication is right. In reinforcement learning, it interacts with the existing environment and uses the trial and error method, whereas the other two types do not apply trial and error. In supervised learning, we will use labeled data, whereas in unsupervised learning, we will use unlabeled data, and in reinforcement learning, there are a bunch of goals and decision processes involved. You can refer to *Figure 8.4*:

![](img/a0779102-7e3e-47e8-ab25-5b06fb541b7e.png)

Figure 8.4: Comparison between supervised, unsupervised, and reinforcement learning (Image credit: http://www.techjini.com/wp-content/uploads/2017/02/mc-learning.jpg)

There are so many new things that you will be learning from this section onwards, if you don't understand some of the terminology at first, then don't worry! Just bear with me; I will explain each of the concepts practically throughout this chapter. So, let's start understanding the development steps for NLP applications that use ML.

# Development steps for NLP applications

In this section, we will discuss the steps of developing NLP applications using ML algorithms. These steps vary from domain to domain. For NLP applications, the visualization of data does not play that much of a critical role, whereas the visualization of data for an analytical application will give you a lot of insight. So, it will change from application to application and domain to domain. Here, my focus is the NLP domain and NLP applications, and when we look at the code, I will definitely recall the steps that I'm describing here so that you can connect the dots.

I have divided the development steps into two versions. The first version is taking into account that it's the first iteration for your NLP application development. The second version will help you with the possible steps that you can consider after your first iteration of the NLP application development. Refer to *Figure 8.5*:

![](img/64464987-d5ac-43e1-ae5a-2238e888de2c.png)

Figure 8.5: NLP application development steps version

# Development step for the first iteration

First, we will look at the steps that we can generally use when we develop the first version of the NLP application using ML. I will refer to *Figure 8.6* during my explanation so that you can understand things properly:

![](img/f4899d64-c66f-4f55-8e92-2815628deb04.png)

Figure 8.6: The first version and iteration to develop an application using ML algorithms

I'm going to explain each step:

1.  The first step of this version is understanding your problem statement, application requirements, or the objective that you are trying to solve.
2.  The second step is to get the data that you need to solve your objective or, if you have the dataset, then try to figure out what the dataset contains and what is your need in order to build an NLP application. If you need some other data, then first ask yourself; can you derive the sub-data attributes with the help of the available dataset? If yes, then there may be no need to get another dataset but if not, then try to get a dataset that can help you develop your NLP application.
3.  The third step is to think about what kind of end result you want, and according to that, start exploring the dataset. Do some basic analysis.
4.  The fourth step is after doing a general analysis of the data, you can apply preprocessing techniques on it.
5.  The fifth step is to extract features from the preprocessed data as part of feature engineering.
6.  The sixth is, using statistical techniques, you can visualize the feature values. This is an optional step for an NLP application.
7.  The seventh step is to build a simple, basic model for your own benchmark.
8.  Last but not least, evaluate the basic model, and if it is up to the mark, then good; otherwise, you need more iterations and need to follow another version, which I will be describing in the next section.

# Development steps for the second to nth iteration

We have seen the steps that you can take in the first iteration; now we will see how we can execute the second iteration so that we can improvise our model accuracy as well as efficiency. Here, we are also trying to make our model as simple as possible. All these goals will be part of this development version.

Now we will see the steps that you can follow after the first iteration. For basic understanding, refer to *Figure 8.7*:

![](img/405bc9c6-8ec2-4f01-b723-ebfa91fe1c26.jpg)

Figure 8.7: ML building cycle

Some of the basic steps for the second iteration are as follows:

1.  After the first iteration, you have already built a model, and now you need to improve it. I would recommend you try out different ML algorithms to solve the same NLP application and compare the accuracy. Choose the best three ML algorithms based on accuracy. This will be the first step.
2.  As a second step, generally, you can apply hyperparameter tuning for each of the selected ML algorithms in order to achieve better accuracy.
3.  If parameter optimization doesn't help you so much, then you need to really concentrate on the feature engineering part and this will be your step three.
4.  Now, feature engineering has two major parts: feature extraction and feature selection. So in the first iteration, we have already extracted feature, but in order to optimize our ML model, we need to work on feature selection. We will look at all the feature selection techniques later in this chapter.
5.  In feature selection, you basically choose those feature, variable, or data attributes that are really critical or contribute a lot in order to derive the outcome. So, we will consider only important feature and remove others.
6.  You can also remove outliers, perform data normalization, and apply cross validation on your input data, which will help you improvise your ML model.
7.  After performing all these tricks, if you don't get an accurate result, then you need to spend some time deriving new features and use them.
8.  You can reiterate all the preceding steps until you get a satisfactory outcome.

This is how you can approach the development of an NLP application. You should observe your results and then take sensible, necessary steps in the next iteration. Be smart in your analysis, think about all the problems, and then reiterate to solve them. If you don't analyze your result thoroughly, then reiteration never helps you. So keep calm, think wisely, and reiterate. Don't worry; we will look at the previous process when we develop NLP applications using ML algorithms. If you are on the research side, then I strongly recommend you understand the math behind the ML algorithms, but if you are a beginner and not very familiar with math, then you can read the documentation of the ML library. Those who lay between these two zones, try to figure out the math and then implement it.

Now, it's time to dive deep into the ML world and learn some really great algorithms.

# Understanding ML algorithms and other concepts

Here, we will look at the most widely used ML algorithms for the NLP domain. We will look at algorithms as per the types of ML. First, we will start with supervised ML algorithms, then unsupervised ML algorithms, and lastly, semi-supervised ML algorithms. Here, we will understand the algorithm as well as the mathematics behind it. I will keep it easy so that those who are not from a strong mathematical background can understand the intuitive concept behind the algorithm. After that, we will see how we can practically use these algorithms to develop an NLP application. We will develop a cool NLP application which will help you understand algorithms without any confusion.

So, let's begin!

# Supervised ML

We saw the introduction to supervised machine learning earlier in this chapter. Whatever techniques and datasets we see and use include their outcome, result, or labels that are already given in the dataset. So, this means that whenever you have a labeled dataset, you can use supervised ML algorithms.

Before starting off with algorithms, I will introduce two major concepts for supervised ML algorithms. This will also help you decide which algorithm to choose to solve NLP or any other data science-related problem:

*   Regression
*   Classification

# Regression

Regression is a statistical process that estimates the relationships between variables. Suppose you have a bunch of variables and you want to find out the relationship between them. First, you need to find out which are the dependent variable, and which are the independent variables. Regression analysis helps you understand how the dependent variable changes its behavior or value for given values of independent variables. Here, dependent variables depend on the values of independent variables, whereas independent variables take values that are not dependent on the other variables.

Let's take an example to give you a clear understanding. If you have a dataset that has the height of a human and you need to decide the weight based on the height, this is supervised ML and you already have the age in your dataset. So, you have two attributes, which are also called variables: height, and weight. Now, you need to predict the weight as per the given height. So, think for some seconds and let me know which data attribute or variable is dependent and which is independent. I hope you have some thoughts. So, let me answer now. Here, weight is the dependent data attribute or variable that is going to be dependent on the variable-height. Height is the independent variable. The independent variable is also called a **predictor**(**s**). So if you have a certain mapping or relationship between the dependent variable and independent variables, then you can also predict the weight for any given height.

Note that regression methods are used when our output or dependent variable takes a continuous value. In our example, weight can be any value such as 20 kg, 20.5 kg, 20.6 kg, 60 kg, and so on. For other datasets or applications, the values of the dependent variable can be any real number. Refer to *Figure 8.8*:

![](img/005390a5-d64d-4f8f-a003-a9604980576e.png)

Figure 8.8: Linear regression example

# Classification

In this section, we will look at the other major concept of supervised ML, which is called **classification techniques**. This is also called **statistical classification**.

Statistical classification is used to identify a category for a given new observation. So, we have many categories in which we can put the new observation. However, we are not going to blindly choose any category, but we will use the given dataset, and based on this dataset, we will try to identify the best suited category for the new observation and put our observation in this category or class.

Let's take an example from the NLP domain itself. You have a dataset that contains a bunch of emails and those emails already have a class label, which is either spam or non-spam. So, our dataset is categorized into two classes--spam and non-spam. Now if we get a new email, then can we categorize that particular e-mail into the spam or not-spam class? The answer is yes. So, to classify the new e-mail we use our dataset and ML algorithm and provide the best suited class for the new mail. The algorithm that implements the classification is called a **classifier**. Sometimes, the term classifier also refers to the mathematical function which is implemented by the classifier algorithm that maps the input data to a category.

Note that this point helps you identify the difference between regression and classification. In classification, the output variable takes the class label that is basically a discrete or categorical value. In regression, our output variable takes a continuous value. Refer to *Figure 8.9*:

![](img/d7b7bed5-10ec-4e77-b2ee-519df7118234.png)

Figure 8.9: Classification visualization for intuitive purposes

Now that we have an idea about regression and classification, let's understand the basic terminology that I am going to use constantly while explaining ML algorithms specially for classification:

*   **Instance:** This is referred to as input and generally, they are in the form of vectors. These are vectors of attributes. In the POS tagger example, we used features that we derived from each word and converted them to vectors using `scikit-learns` API `DictVectorizer`. The vector values were fed into the ML algorithm so these input vectors are the instances.
*   **Concept:** The concept is referred to as a function that maps input to output. So, if we have an e-mail content and we are tying to find out whether that e-mail content is spam or non-spam, we have to focus on some certain parameters from the instance or input and then generate the result. The process of how to identify certain output from certain input is called **concept**. For example, you have some data about the height of a human in feet. After seeing the data, you can decide whether the person is tall or short. Here, the concept or function helps you to find the output for a given input or instance. So, if I put this in mathematical format, then the concept is a mapping between an object in a world and membership in a set.
*   **Target concept:** The target concept is referred to as the actual answer or specific function or some particular idea that we are trying to find. As humans, we have understood a lot of concepts in our head, such as by reading the e-mail, we can judge that it's spam or non-spam, and if your judgments is true, then you can get the actual answer. You know what is called **spam** and what is not, but unless we actually have it written down somewhere, we don't know whether it's right or wrong. If we note these actual answers for each of the raw data in our dataset, then it will be much easier for us to identify which e-mails should be considered as spam e-mails and which not. This helps you find out the actual answer for a new instance.
*   **Hypothesis class:** Is the class of all possible functions that can help us classify our instance. We have just seen the target concept where we are trying to find out a specific function, but here we can think of a subset of all the possible and potential functions that can help us figure out our target concept for classification problems. Here, I want to point out that we are seeing this terminology for classification tasks so don't consider the x2 function, because it's a linear function and we are performing classification not regression.
*   **Training dataset:** In classification we are trying to find the target concept or actual answer. Now, how can we actually get this final answer? To get the final answer using ML techniques, we will use some sample set, training set, or training dataset that will help us find out the actual answer. Let's see what a training set is. A training set contains all the input paired with a label. Supervised classification problems need a training dataset that has been labeled with the actual answer or actual output. So, we are not just passing our knowledge to the machine about what is spam or non-spam; we are also providing a lot of examples to the machine, such as this is a spam mail, this is non-spam mail, and so on. So, for the machine it will be easy to understand the target concept.
*   **ML-model:** We will use the training dataset and feed this data to the ML algorithm. Then, the ML algorithm will try to learn the concept using a lot of training examples and generate the output model. This output model can be used later on to predict or decide whether the given new mail is spam or not-spam. This generated output is called the **ML-model**. We will use a generated ML-model and give the new mail as input and this ML-model will generate the answer as to whether the given mail belongs to the spam category or not-spam category.
*   **Candidate:** The candidate is the potential target concept that our ML-model tells us for the new example. So, you can say that the candidate is the predicted target concept by the machine, but we don't know whether the predicted or generated output that is the candidate here is actually the correct answer or not. So, let's take an example. We have provided a lot of examples of emails to the machine. The machine may generalize the concept of spam and not-spam mails. We will provide a new e-mail and our ML-model will say that it's non-spam, however, we need to check whether our ML-model's answer is right or wrong. This answer is referred to as a candidate. How can we check whether the answer generated by the ML-model matches with the target concept or not? To answer your question, I will introduce the next term, that is, testing set.
*   **Testing set:** The testing set looks similar to the training dataset. Our training dataset has e-mails with labels such as spam or non-spam. So, I will take the answer that is considered as the candidate and we will check in our testing set whether it is non-spam or spam. We will compare our answer with the testing set's answer and try to figure out whether the candidate has a true answer or false answer. Suppose that not-spam is the right answer. Now, you will take another e-mail and the ML-model will generate a non-spam answer again. We will again check this with our testing set, and this time the ML-model generates a wrong answer- the mail is actually spam but the ML-model misclassifies it in the non-spam category. So the testing set helps us validate our ML-model. Note that the training and testing sets should not be the same. This is because, if your machine uses the training dataset to learn the concept and you test your ML-model on the training dataset, then you are not evaluating your ML-model fairly. This is considered cheating in ML. So, your training dataset and testing set should always be different; the testing set is the dataset that has never been seen by your machine. We are doing this because we need to check the machines ability on how much the given problem can be generalized. Here, generalized means how the ML-model reacts to unknown and unseen examples. If you are still confused, then let me give you another example. You are a student and a teacher taught you some facts and gave you some examples. Initially, you just memorized the facts. So as to check that you got the right concept, the teacher will give a test and give you novel examples where you need to apply your learning. If you are able to apply your learning perfectly to the new example in the test, then you actually got the concept. This proves that we can generalize the concept that has been taught by a teacher. We are doing the same thing with the machine.

Now let's understand ML algorithms.

# ML algorithms

We have understood enough about the essential concepts of ML, and now we will explore ML algorithms. First, we will see the supervised ML algorithms that are mostly used in the NLP domain. I'm not going to cover all the supervised ML algorithms here, but I'm explaining those that are most widely used in the NLP domain.

In NLP applications, we mostly perform classification applying various ML techniques. So, here, our focus is mostly on the classification type of an algorithm. Other domains, such as analytics use various types of linear regression algorithms, as well as analytical applications but we are not going to look at those algorithms because this book is all about the NLP domain. As some concepts of linear regression help us understand deep learning techniques, we will look at linear regression and gradient descent in great detail with examples in [Chapter 9](f414d38e-b88e-4239-88bd-2d90e5ce67ab.xhtml), *Deep Learning for NLP and NLG Problems*.

We will develop some NLP applications using various algorithms so that you can see how the algorithm works and how NLP applications develop using ML algorithms. We will look at applications such as spam filtering.

Refer to *Figure 8.10*:

![](img/a1f7e7c8-ae21-4498-946a-6cd12db78927.png)

Figure 8.10: Supervised classification ML algorithms that we are going to understand

Now let's start with our core ML part.

**Logistic regression**

I know you must be confused as to why I put logistic regression in the classification category. Let me tell you that it's just the name that is given to this algorithm, but it's used to predict the discrete output, so this algorithm belongs to the classification category.

For this classification algorithm I will give you an idea how the logistic regression algorithm works and we will look at some basic mathematics related to it. Then, we will look the spam filtering application.

First, we will consider binary classes such as spam or not-spam, good or bad, win or lose, 0 or 1, and so on to understand the algorithm and its application. Suppose I want to classify e-mails into the spam and not-spam category. Spam and not-spam are discrete output labels or target concepts. Our goal is to predict whether the new e-mail is spam or not-spam. Not-spam is also called **ham**. In order to build this NLP application, we will use logistic regression.

Let's understand the technicality of the algorithm first.

Here, I'm stating facts related to mathematics and this algorithm in a very simple manner. A general approach to understanding this algorithm is as follows. If you know some part of ML, then you can connect the dots, and if you are new to ML, then don't worry, because we are going to understand every part:

*   We are defining our hypothesis function that helps us generate our target output or target concept
*   We are defining the cost function or error function and we choose the error function in such a way that we can derive the partial derivate of the error function so that we can calculate gradient descent easily
*   We are trying to minimize the error so that we can generate a more accurate label and classify the data accurately

In statistics, logistic regression is also called **logit regression** or the **logit model**. This algorithm is mostly used as a binary class classifier, which means that there should be two different classes to classify the data. The binary logistic model is used to estimate the probability of a binary response and it generates the response based on one or more predictors or independent variables or features. This is the ML algorithm that uses basic mathematics concepts in deep learning as well.

First, I want to explain why this algorithm is called **logistic regression**. The reason is that the algorithm uses a logistic function or sigmoid function. Logistic function and sigmoid function are synonyms.

We use the sigmoid function as a hypothesis function. What do you mean by hypothesis function? Well, as we saw earlier, the machine has to learn mapping between data attributes and the given label in such a way that it can predict the label for new data. This can be achieved by the machine if it learns this mapping via a mathematical function. The mathematical function is the hypothesis function that the machine will use to classify the data and predict labels or the target concept. We want to build a binary classifier, so our label is either spam or ham. So, mathematically, I can assign 0 for ham or not-spam and 1 for spam or vice versa. These mathematically assigned labels are our dependent variables. Now, we need our output labels to be either zero or one. Mathematically, the label is *y* and *y* *ε {0, 1}*. So we need to choose the hypothesis function that will convert our output value to zero or one. The logistic function or sigmoid function does exactly that and this is the main reason why logistic regression uses a sigmoid function as the hypothesis function.

**Logistic or sigmoid function**

The mathematical equation for the logistic or sigmoid function is as shown:

![](img/863d0a74-fdc4-4500-8a23-1feba25ac799.png)

Figure 8.11: Logistic or sigmoid function

You can see the plot showing *g(z)*. Here, *g(z)= Φ(z)*. Refer to *Figure 8.12*:

![](img/b8e22a24-1fe8-4409-a7e5-7239e4cdcbeb.png)

Figure 8.12: Graph of the sigmoid or logistic function

From the preceding graph, you can find the following facts:

*   If you have *z* value greater or equal to zero, then the logistic function gives the output value one
*   If you have value of *z* less than zero, then the logistic function generates the output zero

You can see the mathematical condition for the logistic function as shown:

![](img/8cbb67fc-3b82-4d35-8629-c80017d3f01e.png)

Figure 8.13: Logistic function mathematical property

We can use this function to perform binary classification.

Now it's time to show how this sigmoid function will be represented as the hypothesis function:

![](img/0a6d0f42-1fbe-4895-83db-02ef9a540cfb.png)

Figure 8.14: Hypothesis function for logistic regression

If we take the preceding equation and substitute the value of *z* with *θ^Tx*, then the equation given in *Figure 8.11* is converted to the equation in *Figure 8.15*:

![](img/62212b3a-fafb-4ab7-828d-14cd2085a571.png)

Figure 8.15: Actual hypothesis function after mathematical manipulation

Here, *h[θ]x* is the hypothesis function, *θ^T* is the matrix of the features or independent variables and transpose representation of it, *x* is for all independent variables or all possible features set. In order to generate the hypothesis equation, we replace the *z* value of the logistic function with *θ^Tx*.

Using the hypothesis equation, the machine actually tries to learn mapping between input variables or input features and output labels. Let's talk a bit about the interpretation of this hypothesis function. Can you think of the best way to predict the class label? According to me, we can predict the target class label using the probability concept. We need to generate probability for both classes and whatever class has a high probability will be assigned to that particular instance of features. In binary classification, the value of *y* or the target class is either zero or one. If you are familiar with probability, then you can represent the probability equation given in *Figure 8.16*:

![](img/dadb2f88-1ff5-47d0-bf8c-f88ab92a44c8.png)

Figure 8.16: Interpretation of the hypothesis function using probabilistic representation

So those who are not familiar with probability, *P(y=1|x;θ )* can be read like this - probability of *y =1*, given *x*, and parameterized by *θ*. In simple language, you can say that this hypothesis function will generate the probability value for target output *1* where we give features matrix *x* and some parameter *θ*. We will see later on why we need to generate probability, as well as how we can generate probability values for each of the classes.

Here, we have completed the first step of a general approach to understanding logistic regression.

**Cost or error function for logistic regression**

First, let's understand the cost function or error function. The cost function, loss function, or error function is a very important concept in ML, so we will understand the definition of the cost function.

The cost function is used to check how accurately our ML classifier performs. In our training dataset, we have data and label. When we use the hypothesis function and generate the output, we need to check how near we are to the actual prediction. If we predict the actual output label, then the difference between our hypothesis function output and the actual label is zero or minimum and if our hypothesis function output and actual label are not the same, then we have a big difference between them. If the actual label of an e-mail is spam, that is one, and our hypothesis function also generates the result 1 then the difference between the actual target value and predicted output value is zero, therefore the error in the prediction is also zero. If our predicted output is 1, and the actual output is zero, then we have maximum error between our actual target concept and prediction. So, it is important for us to have minimum error in our prediction. This is the very basic concept of the error function. We will get to the mathematics in some time. There are several types of error functions available, such as r2 error, sum of squared error, and so on. As per the ML algorithm and hypothesis function, our error function also changes.

What will the error function be for logistic regression? What is θ and, if I need to choose some value of θ, how can I approach it? So, here, I will give all the answers.

Let me give you some background on linear regression. We generally use sum of squared error or residual error as the cost function in linear regression. In linear regression, we are trying to generate the line of best fit for our dataset. In the previous example, given height, I want to predict the weight. We first draw a line and measure the distance from each of the data points to the line. We will square these distances, sum them and try to minimize the error function. Refer to *Figure 8.17*:

![](img/4228b2a0-1e45-487d-af77-a217986d8611.png)

Figure 8.17: Sum of squared error representation for reference

You can see the distance of each data point from the line is denoted using small vertical lines. We will take these distances, square them and then sum them. We will use this error function. We have generated a partial derivative with respect to the slope of line *m* and intercept *b*. Here, in *Figure 8.17,* our *b* is approximately *0.9* and *m* is approximately two thirds. Every time, we calculate the error and update the value of *m* and *b* so that we can generate the line of best fit. The process of updating *m* and *b* is called **gradient descent**. Using gradient descent, we update *m* and *b* in such a way that our error function has minimum error value and we can generate the line of best fit. Gradient descent gives us a direction in which we need to plot a line. You can find a detailed example in [Chapter 9](f414d38e-b88e-4239-88bd-2d90e5ce67ab.xhtml), *Deep Learning for NLP and NLG Problems*. So, by defining the error function and generating partial derivatives, we can apply the gradient descent algorithm that helps us minimize our error or cost function.

Now back to the main question: Can we use the error function for logistic regression? If you know functions and calculus well, then probably your answer is no. That is the correct answer. Let me explain this for those who aren't familiar with functions and calculus.

In linear regression, our hypothesis function is linear, so it is very easy for us to calculate the sum of squared errors, but here, we will use the sigmoid function, which is a non-linear function. If you apply the same function that we used in linear regression, it will not turn out well because if you take the sigmoid function, put in the sum of squared error function, and try to visualize all the possible values, then you will get a non-convex curve. Refer to *Figure 8.18*:

![](img/45a5426c-5346-4720-93d3-ede748ddf07a.png)

Figure 8.18: Non-convex and convex curve (Image credit: http://www.yuthon.com/images/non-convex_and_convex_function.png)

In ML, we mostly use functions that are able to provide a convex curve because then we can use the gradient descent algorithm to minimize the error function and reach a global minimum. As you can see in *Figure 8.18*, a non-convex curve has many local minima, so to reach a global minimum is very challenging and time-consuming because you need to apply second order or *n*th order optimization to reach a global minimum, whereas in a convex curve, you can reach a global minimum certainly and quickly.

So, if we plug our sigmoid function into sum of squared error, you get the non-convex function so we are not going to define the same error function that we used in linear regression.

We need to define a different cost function that is convex so that we can apply the gradient descent algorithm and generate a global minimum. We will use the statistical concept called **likelihood**. To derive the likelihood function, we will use the equation of probability that is given in *Figure 8.16* and we will consider all the data points in the training dataset. So, we can generate the following equation, which is called likelihood function. Refer to *Figure 8.19*:

![](img/2ad6cf39-70c4-46c2-9f83-6d84ac1e84f6.png)

Figure 8.19: The likelihood function for logistic regression (Image credit: http://cs229.stanford.edu/notes/cs229-notes1.pdf)

Now, in order to simplify the derivative process, we need to convert the likelihood function to a monotonically increasing function. That can be achieved by taking the natural logarithm of the likelihood function, which is called **log likelihood**. This log-likelihood is our cost function for logistic regression. Refer to the following equation in F*igure 8.20*:

![](img/56af5c51-a448-42a1-99eb-e518f279afb7.png)

Figure 8.20: Cost function for logistic regression

We will plot the cost function and understand the benefits it provides us with. In the *x* axis, we have our hypothesis function. Our hypothesis function range is *0* to *1*, so we have these two points on the *x* axis. Start with the first case, where *y = 1*. You can see the generated curve that is on the top right-hand side in *Figure 8.21*:

![](img/3c093e59-0d98-4584-8f2f-a741bd4c3454.png)

Figure 8.21: Logistic function cost function graph

If you look at any log function plot, then it will look like the graph for error function *y=0*. Here, we flip that curve because we have a negative sign, then you get the curve which we have plotted for *y=1* value. In *Figure 8.21,* you can see the log graph, as well as the flipped graph in *Figure 8.22*:

![](img/b9d20413-29f8-4b0c-878f-056b20e642f7.png)

Figure 8.22: Comparing log(x) and -log(x) graph for better understanding of the cost function (Image credit : http://www.sosmath.com/algebra/logs/log4/log42/log422/gl30.gif)

Here, we are interested in the values *0* and *1,* so we are considering that part of the graph that we have depicted in *Figure 8.21*. This cost function has some interesting and useful properties. If the predicted label or candidate label is the same as the actual target label, then the cost will be zero. You can put this as *y = 1* and if the hypothesis function predicts *H[θ](x) = 1* then *cost = 0*; if *H[θ](x)* tends to *0*, meaning that if it is more toward the zero, then the cost function blows up to *∞*.

For *y = 0*, you can see the graph that is on the top left-hand side in *Figure 8.21.* This case condition also has the same advantages and properties that we saw earlier. It will blow to *∞* when the actual value is *0* and the hypothesis function predicts *1*. If the hypothesis function predicts *0* and the actual target is also *0*, then *cost = 0*.

Now, we will see why we are choosing this cost function. The reason is that this function makes our optimization easy, as we will use maximum log-likelihood because it has a convex curve that will help us run gradient descent.

In order to apply gradient descent, we need to generate the partial derivative with respect to *θ* and generate the equation that is given in *Figure 8.23*:

![](img/f2409931-a8ae-48b4-af4f-5d76e54a5340.png)

Figure 8.23: Partial derivative to perform gradient descent (Image credit: http://2.bp.blogspot.com/-ZxJ87cWjPJ8/TtLtwqv0hCI/AAAAAAAAAV0/9FYqcxJ6dNY/s1600/gradient+descent+algorithm+OLS.png)

This equation is used to update the parameter value of *θ*; ![](img/332b6e21-d2d7-4c80-af9a-678da52e59a3.png) defines the learning rate. This is the parameter that you can use to set how fast or how slow your algorithm should learn or train. If you set the learning rate too high, then the algorithm cannot learn, and if you set it too low, then it will take a lot of time to train. So you need to choose the learning rate wisely.

This is the end of our second point and we can begin with the third part, which is more of an implementation part. You can check out this GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch8/Own_Logistic_Regression/logistic.py](https://github.com/jalajthanaki/NLPython/blob/master/ch8/Own_Logistic_Regression/logistic.py)

This has an implementation of logistic regression and you can find its comparison with the given implementation in the `scikit-learn` library. Here, the code credit goes to Harald Borgen.

We will use this algorithm for spam filtering. Spam filtering is one of the basic NLP applications. Using this algorithm, we want to make an ML-model that classifies the given mail into either the spam or ham category. So, let's make a spam-filtering application. The entire code is on this GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch8/Spamflteringapplication/Spam_filtering_logistic_regression.ipynb](https://github.com/jalajthanaki/NLPython/blob/master/ch8/Spamflteringapplication/Spam_filtering_logistic_regression.ipynb)

In spam filtering, we will use the `CountVectorizer` API of `scikit-learn` to generate the features, then train using `LogisticRegression`. You can see the code snippet in *Figure 8.24*:

![](img/5000911e-5de5-4636-b878-4d45dc08b855.png)

Figure 8.24: Spam filtering using logistic regression

First, we perform some basic text analysis that will help us understand our data. Here, we have converted the text data to a vector format using `scikit-learn` API, `Count Vectorizer()`. This API uses **term frequency-inverse document frequency** (**tf-idf**) underneath. We have divided our dataset into a training dataset and a testing set so that we can check how our classifier model performs on the test dataset. You can see the output in *Figure 8.25*:

![](img/619e5db3-c446-4836-bff6-046b9222724c.png)

Figure 8.25: Output of spam filtering using logistic regression

**Advantages of logistic regression**

The following are the advantages of logistic regression:

*   It can handle non-linear effects
*   It can generate the probability score for each of the classes, which makes interpretation easy

**Disadvantages of logistic regression**

The following are the disadvantages of logistic regression:

*   This classification technique is used for binary classification only. There are other algorithms which we can use if we want to classify data into more than two categories. We can use algorithms like random forest and decision tree for classifying the data into more than two categories.
*   If you provide lots of features as input to this algorithm, then the features space increases and this algorithm doesn't perform well
*   Chances of overfitting are high, which means that the classifier performs well on the training dataset but cannot generalize enough that it can predict the right target label for unseen data

Now it's time to explore the next algorithm, called **decision tree**.

**Decision tree**

**Decision tree** (**DT**) is one of the oldest ML algorithms. This algorithm is very simple, yet robust. This algorithm provides us with a tree structure to make any decision. Logistic regression is used for binary classification, but you can use a decision tree if you have more than two classes.

Let's understand the decision tree by way of an example. Suppose, Chris likes to windsurf, but he has preferences - he generally prefers sunny and windy days to enjoy it and doesn't surf on rainy or overcast days or less windy days either. Refer to *Figure 8.26*:

![](img/abfe5342-5a76-4c78-ada6-9ca85e0b8d49.png)

Figure 8.26: Toy dataset to understand the concept (Image credit: https://classroom.udacity.com)

As you can see, *o* (dots) are the happy weather conditions when Chris likes to wind surf and the *x* (crosses) are the bad weather conditions when Chris doesn't like to wind surf.

The data that I have drawn is not linearly separable, which means that you can't classify or separate the red crosses and blue dots just using a single line. You might think that if the goal is just to separate the blue dots and red crosses, then I can use two lines and achieve this. However, can a single line separate the blue dots and red crosses? The answer is no, and that is the reason why I have told you that this dataset is not linearly separable. So for this kind of scenario, we will use a decision tree.

What does a decision tree actually do for you? In layman's terms, decision tree learning is actually about asking multiple linear questions. Let's understand what I mean by linear questions.

Suppose we ask a question: Is it windy? You have two answers: Yes or No. We have a question that is related to wind, so we need to focus on the *x* axis of *Figure 8.26*. If our answer is: Yes, it's windy, then we should consider the right-hand side area that has red crosses as well as blue dots; if we answer: No, it isn't windy, then we need to consider all the red crosses on the left-hand side. For better understanding, you can refer to *Figure 8.27*:

![](img/37e82d6a-66bf-4869-8bdb-1a141570a935.png)

Figure 8.27: Representation of the question: Is it windy? (Image credit: https://classroom.udacity.com/courses/ud120)

As you can see in *Figure 8.27*, I have put one line that passes through the midpoint on the *x* axis. I have just chosen a midpoint, there is no specific reason for that. So I have drawn a black line. Red crosses on the left-hand side of the line represent: No, it's not windy, and red crosses on the right-hand side of the line represent: Yes, it's windy. On the left-hand side of the line, there are only red crosses and not a single blue dot there. If you select the answer, No, then actually you traverse with the branch labeled as No. The area on the left side has only red crosses, so you end up having all the data points belonging to the same class, which is represented by red crosses, and you will not ask further questions for that branch of the tree. Now, if you select the answer, Yes, then we need to focus on the data points that are on the right-hand side. You can see that there are two types of data points, blue dots as well as red crosses. So, in order to classify them, you need to come up with a linear boundary in such a way that the section formed by the line has only one type of data point. We will achieve this by asking another question: Is it sunny? This time, again, you have two possible answers - Yes or No. Remember that I have traversed the tree branch that has the answer of our first question in the form of Yes. So my focus is on the right-hand side of the data points, because there I have data points that are represented in the form of red crosses as well as blue dots. We have described the sun on the *y* axis, so you need to look at that axis and if you draw a line that passes through the midpoint of the *y* axis, then the section above the line represents the answer, Yes, it is a sunny day. All data points below the line represent the answer, No, it is not a sunny day. When you draw such a line and stop extending that line after the first line, then you can successfully segregate the data points that reside on the right-hand side. So the section above the line contains only blue dots and the section below the line, red crosses. You can see the horizontal line, in *Figure 8.28*:

![](img/11926a62-a06a-4230-ae7a-8c47b3546a38.png)

Figure 8.28: Grey line representing the classification done based on the first question

We can observe that by asking a series of questions or a series of linear questions, we actually classify the red crosses that represent when Chris doesn't surf and blue dots that represent when Chris surfs.

This is a very basic example that can help you understand how a decision tree works for classification problems. Here, we have built the tree by asking a series of questions as well as generating multiple linear boundaries to classify the data points. Let's take one numeric example so that it will be clearer to you. Refer to *Figure 8.29*:

![](img/40caf52c-e4df-495e-8bb7-716249e3c647.png)

Figure 8.29: See the data points in the 2D graph (Image credit: https://classroom.udacity.com/courses/ud120)

You can see the given data points. Let's start with the *x* axis first. Which threshold value on the *x* axis do you want to choose so that you can obtain the best split for these data points? Think for a minute! I would like to select a line that passes the *x* axis at point 3\. So now you have two sections. Mathematically, I choose the best split for the given data points, that is, x < = 3 and *x > 3.* Refer to *Figure 8.30*:

![](img/bba4c6c6-f80b-45cc-a994-1986d34ace47.png)

Figure 8.30: See the first linear question and decision boundary graph (Image credit : https://classroom.udacity.com/courses/ud120)

Let's focus on the left-hand side section first. Which value on the *y* axis would you prefer to choose so that you have only one type of data point in one section after drawing that line? What is the threshold on *y* axis that you select so you have one type of dataset in one section and the other type of dataset in the other section? I will choose the line that passes through the point 2 on the *y* axis. So, the data points above the line belong to one class and the data points below the line belong to the other class. Mathematically, *y < = 2* gives you one class and *y > 2* gives you the other class. Refer to *Figure 8.31*:

![](img/e8db2a8b-b00f-4a4e-a6a5-379ff9433601.png)

Figure 8.31: See the second linear question and decision boundary graph (Image credit: https://classroom.udacity.com/courses/ud120)

Now focus on the right-hand side part; for that part also, we need to choose a threshold with respect to the *y* axis. Here, the best threshold for a separation boundary is *y = 4*, so the section *y < 4* has only red crosses and the section *y > =4* has only blue dots. So finally, with a series of linear questions, we are able to classify our data points. Refer to *Figure 8.32*:

![](img/16d73b3e-f2f7-4275-a62c-d49ce8739440.png)

Figure 8.32: The final linear question and decision boundary graph (Image credit: https://classroom.udacity.com/courses/ud120)

Now you get an idea about the algorithm, but there may be a couple of questions in your mind. We have a visualization of the obtaining line, but how does the decision tree algorithm choose the best possible way to split data points and generate a decision boundary using the given features? Suppose I have more than two features, say ten features; then how will the decision tree know that it needs to use the second feature and not the third feature in the first time? So I'm going to answer all these questions by explaining the math behind a decision tree. We will look at an NLP-related example, so that you can see how a decision tree is used in NLP applications.

I have some questions related to decision trees let's answer them one by one. We will use visualization to obtain a linear boundary, but how does a decision tree recognize using which features and which feature value should it split the data? Let's see the mathematical term that is entropy. So, decision tree uses the concept of entropy to decide where to split the data. Let's understand entropy. Entropy is the measure of the impurity in a tree branch. So, if all data points in a tree branch belong to the same class, then entropy *E = 0*; otherwise, entropy *E > 0* and *E <= 1.* If Entropy *E = 1*, then it indicates that the tree branch is highly impure or data points are evenly split between all the available classes. Let's see an example so that you can understand the concept of entropy and impurity.

We are making a spam filtering application and we have one feature, that is, words and phrases type. Now we will introduce another feature, that is, the minimum threshold count of appearing phrases in the dataset. Refer to *Figure 8.33*:

![](img/5af94fd1-bd66-4e30-a027-c6fa89f9f409.png)

Figure 8.33: Graphs for entropy discussion

Now focus on the right-hand side graph. In this graph, the right-hand side section has only one kind of data points that are denoted by red crosses. So technically, all data points are homogenous as they belong to the same class. So, there is no impurity and the value of entropy is approximately zero. Now if you focus on the left-hand side graph and see its right-hand side section, you will find data points that belong to the other class label. This section has impurity and, thus, has high entropy. So, during the implementation of a decision tree, you need to find out the variables that can be used to define the split points, along with the variables. Another thing that you need to keep in mind is that you are trying to minimize the impurity in the data, so try to split the data according to that. We will see how to choose variables to perform a split in some time.

Now, let's first see the mathematical formula of entropy. Refer to *Figure 8.34*:

![](img/1948c592-b74a-4ff4-bb10-25b4c5adf312.png)

Figure 8.34: Entropy mathematical formula (Image credit: http://dni-institute.in/blogs/wp-content/uploads/2015/08/Entrop.png)

Let's see what *pi* is here. It is the fraction value for a given class. Let's say *i* is the class. *T* is the total value of the available class. You have four data points; if two points belong to class A and the other two belong to class B, then *T = 2*. We perform summation after generating log values using the fraction value. Now it's time to perform mathematical calculations for entropy and then I will let you know how we can use entropy to perform splitting on variables or features. Let's see an example to calculate entropy. You can find the data for the example in *Figure 8.35*:

![](img/b0b448b9-bca8-424f-b001-303d77cd6f3a.png)

Figure 8.35: The dataset values for spam filtering calculation

If you focus on the **Filtering** column, you have two labels with the value **Spam** and two values with **Ham**-that is *SSHH*. Now answer some of the following questions:

*   How many total data rows do we have? The answer is four
*   How many times does the data label *S* occur in the **Filtering** column? The answer is two
*   How many times does the data label *H* occur in the **Filtering** column? The answer is two
*   To generate the fraction value for the class label *S*, you need to perform mathematics using the following formula:
    *   *pS = No. of time S occurred / Total no. of data rows = 2/4 = 0.5*
*   Now we need to calculate *p* for *H* as well:
    *   *pH = No. of time H occurred / Total no. of data rows = 2/4 = 0.5*
*   Now we have all the necessary values to generate the entropy. Focus on the formula given in *Figure 8.34*:
    *   *Entropy = -pS* log[2](pS) -pH*log[2](pH) = -0.5 * log(0.5) -0.5*log(0.5) = 1.0*
*   You can do this calculation using Python's math module

As you can see, we get the entropy *E = 1*. This is the most impure state, where data is evenly distributed among the available classes. So, entropy tells us about the state of the data whether classes are in an impure state or not.

Now we will look at the most awaited question: How will we know on which variable or using which feature we need to perform a split? To understand this, we need to understand information gain. This is one of the core concepts of the decision tree algorithm. Let me introduce the formula for **information gain** (**IG**):

*Information Gain (IG) = Entropy (Parent Node) - [Weight Average] Entropy (Children)*

Now let's take a look at this formula. We are calculating the entropy of the parent node and subtracting the weighted entropy of the children. If we perform splitting on the parent node, decision tree will try to maximize the information gain. Using IG, the decision tree will choose the feature that we need to perform a split on. This calculation is done for all the available features, so the decision tree knows exactly where to split. You need to refer to *Figure 8.33*.

We have calculated entropy for the parent node: *E (Parent Node) = 1*. Now we will focus on words and calculate IG. Let's check whether we should perform a split using words with IG. Here, we are focusing on the **Words** column. So, let me answer some of the questions so that you understand the calculations of IG:

*   There are how many total positive meaning words? The answer is three
*   There are how many total negative meaning words? The answer is one
*   So, for this branch, our entropy *E = 0*. We will use this when we are calculating weighted average entropy for the child node

Here, our decision tree looks as given in *Figure 8.36*:

![](img/a6ef5228-ffa6-4b55-aa2b-f6dbcb7a99d8.png)

Figure 8.36: Decision trees first iteration

You can see that for the right-hand side node, the entropy is zero so there isn't any impurity in that branch, so we can stop there. However, if you look at the left-hand side node, it has the **SSH** class so we need to calculate entropy for each of the class labels. Let's do it step by step for the left-hand side node:

*   *PS* = No. Of *S* label in branch/ Total no. of example in branch = 2/3
*   *PH* = No. Of *H* label in branch/ Total no. of example in branch = 1/3
*   Now entropy *E* = -2/3 log[2] (2/3) -1/3 log[2] (1/3) = 0.918

In the next step, we need to calculate the weighted average entropy of the child nodes.

We have three data points as part of the left-hand branch and one data point as the right-hand branch, which you can see in *Figure 8.36*. So the values and formula looks as follows:

*Weight average entropy of children = Left hand branch data points / Total no of data points * ( entropy for children in that branch) + Right hand branch data points / Total no of data points * ( entropy for children in that branch)*

*Weight average entropy of children = [Weight Average] Entropy (Children) =¾ * 0.918 + ¼ * (0) = 0.6885*

Now it's time to obtain IG:

*IG = Entropy (parent node) - [Weight Average] Entropy (Children). We have both parts with us-E (Parent Node) = 1* and *[Weight Average] Entropy (Children) = 0.6885*

So, the final calculation is as follows:

*IG = 1 - 0.6885 = 0.3115*

Let's focus on the phrase appeared count column and calculate the entropy for phrase count values three, which is *E[three(3)] = 1.0*, entropy for phrase count values four is *E[four(4)] = 1.0*; now *[Weight Average] Entropy (Children) = 1.0 , IG = 1.0 -1.0 = 0*. So, we are not getting any information gain on this split of this feature. So, we should not choose this feature.

Now let's focus on the column of phrases where we have mentioned the phrase category-**Unusual phrases** or **Usual phrases**. When we split data points using this column, we get the **Spam** class in one branch and **Ham** class in another branch. So here, you need to calculate IG on your own, but the *IG = 1*. We are getting the maximum IG. So we will choose this feature for the split. You can see the decision tree in *Figure 8.37*:

![](img/927ac31c-22af-4554-b383-1f0caa5416de.png)

Figure 8.37: Decision tree generated using phrases type feature

If you have a high number of features, then the decision tree performs training very slowly because it calculates IG for each feature and performs a split by choosing the feature that provides maximum IG.

Now it's time to look at the NLP application that uses decision trees. We will redevelop spam filtering but this time, we will use a decision tree.

We have to just change the algorithm for the spam-filtering application and we have taken the same feature set that we generated previously so that you can compare the result of logistic regression and decision tree for spam filtering. Here, we will use the same features that are generated by the `CountVectorizer` API from `scikit-learn`. The code is at this GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch8/Spamflteringapplication/Spam_filtering_logistic_regression.ipynb](https://github.com/jalajthanaki/NLPython/blob/master/ch8/Spamflteringapplication/Spam_filtering_logistic_regression.ipynb)

You can see the code snippet in *Figure 8.38*:

![](img/81a7e45f-f385-464a-b08e-550d8ae22c57.png)

Figure 8.38: Spam filtering using decision tree

You can see the output in *Figure 8.39*:

![](img/f09257b9-7c00-4984-aa3d-a390ffd1ea9b.png)

Figure 8.39: Spam filtering output using decision tree

As you can see, we get low accuracy compared to logistic regression. Now it's time to see some tuning parameters that you can use to improve the accuracy of the ML-model.

**Tunable parameters**

In this section, I will explain some tunable `scikit-learn`. You can check the documentation at:

[http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier).

I will explain the following parameters that you can use:

*   There is one parameter in `scikit-learn`, which is `criterion`. You can set it as either `entropy` or `gini`. The `entropy` or `gini` are used to calculate IG. So they both have a similar mechanism to calculate IG, and decision tree will perform the split on the basis of the IG calculation given by `entropy` or `gini`.
*   There is `min_sample_size` and its default value is `two`. So, the decision tree branch will split until it has more than or equal to two data elements per branch. Sometimes, decision tree tries to fit maximum training data and overfits the training data points. To prevent overfitting, you need to increase the `min_sample_size` from two to more like fifty or sixty.
*   We can use tree pruning techniques, for which we will follow the bottom-up approach.

Now let's see the advantages and disadvantages of decision tree.

**Advantages of decision tree**

The following are the advantages that decision tree provide:

*   Decision tree is simple and easy to develop
*   Decision tree can be interpreted by humans easily and it's a white box algorithm
*   It helps us determine the worst, best, and expected values for different scenarios

**Disadvantages of decision tree**

The following are the disadvantages that decision tree has:

*   If you have a lot of features, then decision tree may have overfitting issues
*   You need to be careful about the parameters that you are passing while training

We have seen the shortcomings of decision tree. Decision tree generally overfits the training dataset. We need to solve the problem using parameter tuning or a variant of the decision tree random forest ML algorithm. We will understand the random forest algorithm next.

**Random forest**

This algorithm is a variant of decision tree that solves the overfitting problem.

Random forest is capable of developing linear regression as well as classification tasks. Here, we are focusing on a classification task. It uses a very simple trick and works very nicely. The trick is that random forest uses a voting mechanism to improve the accuracy of the test result.

The random forest algorithm generates a random subset of the data from the training dataset and uses this to generate a decision tree for each of the subsets of the data. All these generated trees are called **random forest**. Now let's understand the voting mechanism. Once we have generated decision trees, we check the class label that each tree is provided for a specific data point. Suppose that we have generated three random forest decision trees. Two of them are saying some specific data point belongs to class A and the third decision tree predicts that the specific data point belongs to class B. The algorithm considers the higher vote and assigns the class label A for that specific data point.

For random forest, all the calculations for classification are similar to a decision tree. As promised, I will refer to the example that I gave in [Chapter 5](07f71ca1-6c8a-492d-beb3-a47996e93f04.xhtml), *Feature Engineering and NLP Algorithms*, which is the custom POS tagger example. In that example, we used a decision tree. See the code on this GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch5/CustomPOStagger/ownpostag.py](https://github.com/jalajthanaki/NLPython/blob/master/ch5/CustomPOStagger/ownpostag.py).

Let's revisit the example and understand the features and code given in *figure 8.40*:

![](img/d00b6c77-8030-4936-b319-24f36575b253.png)

Figure 8.40: Code snippet for the decision tree algorithm in scikit-learn

**Advantages of random forest**

The following are the advantages of random forest:

*   It helps us prevent overfitting
*   It can be used for regression as well as classification

**Disadvantages of random forest**

The following are the disadvantages of random forest:

*   The random forest model can easily grow, which means that if the random subset of datasets is high, then we will get more decision trees, thus, we will get a group of trees, also referred as a forest of decision trees that may take a lot of memory
*   For high-dimensional feature space, it is hard to interpret each node of the tree, especially when you have a high number of trees in one forest

Now it's time to understand our next ML algorithm - Naive Bayes.

**Naive Bayes**

In this section, we will understand the probabilistic ML algorithm that is used heavily in many data science applications. We will use this algorithm to develop the most famous NLP application - sentiment analysis, but before jumping into the application, we will understand how the Naive Bayes algorithm works. So, let's begin!

The Naive Bayes ML algorithm is based on Bayes theorem. According to this theorem, our most important assumption is that events are independent, which is a naive assumption, and that is the reason this algorithm is called **Naive Bayes**. So, let me give you an idea of independent events. In classification tasks, we have many features. If we use the Naive Bayes algorithm, then we assume that each and every feature that we are going to provide to the classifier is independent from each other, which means that the presence of a particular feature of the class doesn't affect any other feature. Let's take an example. You want to find out the sentiment of the sentence, It is very good! You have features such as bag of words, adjective phrase, and so on. Even if all these features depend on each other or depend on the existence of other features, all the properties carried by these features independently contribute to the probability that this sentence carries positive sentiment. This is the reason we call this algorithm Naive.

This algorithm is really simple, as well as a very powerful one. This works really well if you have a lot of data. It can classify more than two classes, so it is helpful in building a multi-class classifier. So, now, let's look at some points that will tell us how the Naive Bayes algorithm works. Let's understand the mathematics and probabilistic theorem behind it.

We will understand Bayes rule first. In very simple language, you have prior probability for some event, and you find some evidence of the same event in your test data and multiply them. Then you get the posterior probability that helps you derive your final predication. Don't worry about the terminology, we will get into those details.

Let me give you an equation first and then we will take one example so that you know what the calculation that we need to do is. See the equation in *Figure 8.41*:

![](img/1e2eddae-ddd1-4ea5-9b33-efe75abc2a5c.png)

Figure 8.41: Naive-Bayes algorithm uses Bayes theorem equation (Image credit: http://www.saedsayad.com/images/Bayes_rule.png)

Here, *P(c|x)* is the probability of class *c*, class *c* is the target, and *x* are the features or data attributes. *P(C)* is the prior probability of class *c*, *P(x|c)* is the estimation of the likelihood that is the probability of the predictor given a target class, and *P(x)* is the prior probability of the predictor.

Let's use this equation to calculate an example. Suppose there is a medical test that helps identify whether a person has cancer or not. The prior probability of the person having that specific type of cancer is only *1 %*, which means *P(c) = 0.01 = 1%* and so *P(not c) =0.99 =99%*. There is a *90%* chance that the test will show positive if the person has cancer. So prior probability of *P(Positive result | c )* *= 0.9 = 90%* and there is a *10%* chance that even if the person doesn't have cancer, the result will still show positive, so *P(Positive result | not C )* *= 0.1 =10%.*

Now, we need to check whether the person really has cancer. If the result showed positive, that probability is written as *P(c | Positive result)*, and if the person doesn't have cancer but still the result is positive, then that is denoted by *P( not c | Positive result)*. We need to calculate these two probabilities to derive the posterior probability. First, we need to calculate joint probability:

*Joint P(c | Positive result) = P(c) * P(Positive result | c) = 0.01 x 0.9 =0.009*

*Joint P(not c | Positive result) = P(not c) * P(Positive result | not c) = 0.99 x 0.1 = 0.099*

The preceding probability is called **joint probability**. This will be helpful in deriving the final posterior probability. To get the posterior probability, we need to apply normalization:

*P (Positive result) = P(c | Positive result) + P ( not c | Positive result) = 0.009 +0.099 = 0.108*

Now the actual posterior probability is given as follows:

*Posterior probability of P(c | Positive result) = joint probability of P(c | Positive result) / P (Positive result) = 0.009 / 0.108 = 0.083*

*Posterior probability of P( not c | Positive result) = joint probability of P(not c | Positive result) / P (Positive result) = 0.099 / 0.108 = 0.916*

If you sum up the Posterior probability of *P(c | Positive result) + Posterior probability of P(not c | Positive result),* it should be *= 1.* And in this case, it does sum up to *1*.

There is a lot of mathematics going on, so I will draw a diagram for you that will help you understand these things. Refer to *Figure 8.42*:

![](img/89cf14d6-153e-402b-8161-ac72a18d8a60.png)

Figure 8.42: Posterior probability calculation diagram (Image credit: https://classroom.udacity.com/courses/ud120/lessons/2254358555/concepts/30144285350923)

We will extend this concept to an NLP application. Here, we will take an NLP-based basic example. Suppose there are two persons - Chris and Sara. We have the e-mail details of Chris and Sara. They both use words such as life, love, and deal. For simplicity, we are considering only three words. They both use these three words at different frequencies.

Chris uses the word love only 1% of the time in his mail, whereas he uses the word deal 80% of the time, and life 1% of the time. Sara, on the other hand, uses the word love 50% of the time, deal 20% of the time, and life 30 % of the time. If we have a new e-mail, then we need to decide whether it is written by Chris or Sara. Prior probability of P(Chris) = 0.5 and P(Sara) = 0.5.

The mail has the sentence Life Deal so the probability calculation is for P(Chris| "Life Deal") = P(life) * P(deal) * P(Chris) = 0.04 and the calculation for P(Sara |"Life Deal") = P(life) * P(deal) * P(Sara) = 0.03\. Now, let's apply normalization and generate the actual probability. For this, we need to calculate joint probability = P(Chris| "Life Deal") + P(Sara | "Life Deal") = 0.07\. The following are the actual probability values:

P(Chris| "Life Deal") = 0.04 / 0.07 = 0.57

P(Sara| "Life Deal") = 0.03 / 0.07 = 0.43

The sentence Life Deal is more likely to be written by Chris. This is the end of the example and now it's time for practical implementation. Here, we are developing the most famous NLP application, that is, sentiment analysis. We will do sentiment analysis for text data so we can say that sentiment analysis is the text analysis of opinions that are generated by humans. Sentiment analysis helps us analyze what our customers are thinking about a certain product or event.

For sentiment analysis, we will use the bag-of-words approach. You can also use artificial neural networks, but I'm explaining a basic and easy approach. You can see the code on this GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch8/sentimentanalysis/sentimentanalysis_NB.py](https://github.com/jalajthanaki/NLPython/blob/master/ch8/sentimentanalysis/sentimentanalysis_NB.py)

We will use the `TfidVectorizer` API of `scikit-learn` as well as `MultinomialNB` Naive Bayes. See the code snippet in *Figure 8.43*:

![](img/1226199b-9110-4614-a57e-d0aefa33743f.png)

Figure 8.43: Code snippet for sentiment analysis using Naive Bayes

See the output in *Figure 8.44*:

![](img/1936b2e0-c390-4f1c-a9b7-b6b7b8b3b5a4.png)

Figure 8.44: Output for sentiment analysis using Naive Bayes

Now it's time to look at some tuning parameters.

**Tunable parameters**

For this algorithm, sometimes you need to apply smoothing. Now, what do I mean by smoothing? Let me give you a very brief idea about it. Some of the words come in the training data and our algorithm uses that data to generate an ML-model. If the ML-model sees the words that are not in the training data but present in the testing data, then at that time, our algorithm cannot predict things well. We need to solve this situation. So, as a solution, we need to apply smoothing, which means that we are also calculating the probability for rare words and that is the tunable parameter in `scikit-learn`. It's just a flag--if you enable it, it will perform smoothing or if you disable it, then smoothing will not be applied.

**Advantages of Naive Bayes**

The following are the advantages that the Naive Bayes algorithm provides:

*   You can deal with high-dimensional feature space using the Naive Bayes algorithm
*   It can be used to classify more than two classes

**Disadvantages of Naive Bayes**

The following are the disadvantages of the Naive Bayes algorithm:

*   If you have a phrase composed of different words having different meanings, then this algorithm will not help you. You have a phrase, Gujarat Lions. This is the name of a cricket team, but Gujarat is a state in India and lion is an animal. So, the Naive Bayes algorithm takes the individual words and interprets them separately, and so this algorithm cannot correctly interpret Gujarat Lions.
*   If some categorical data appears in the testing dataset only and not in the training data, then Naive Bayes won't provide a prediction for that. So, to solve this kind of problem, we need to apply smoothing techniques. You can read about this on this link:
    [https://stats.stackexchange.com/questions/108797/in-naive-bayes-why-bother-with-laplacian-smoothing-when-we-have-unknown-words-i](https://stats.stackexchange.com/questions/108797/in-naive-bayes-why-bother-with-laplacian-smoothing-when-we-have-unknown-words-i)
*   Now it is time to look at the last classification algorithm, support vector machine. So, let's begin!

**Support vector machine**

This is the last but not least supervised ML algorithm that we will look in this chapter. It is called **support vector machine** (**SVM**). This algorithm is used for classification tasks as well as regression tasks. This algorithm is also used for multi-class classification tasks.

SVM takes labeled data and tries to classify the data points by separating them using a line that is called the **hyperplane**. The goal is to obtain an optimal hyperplane that will be used to categorize the existing as well as new, unseen examples. How to obtain an optimal hyperplane is what we are going to understand here.

Let's understand the term optimal hyperplane first. We need to obtain the hyperplane in such a way that the obtained hyperplane maximizes the distances to its nearest points of all the classes and this distance is called **margin**. Here, we will talk about a binary classifier. Margin is the distance between the hyperplane (or line) and the nearest point of either of the two classes. SVM tries to maximize the margin. Refer to *Figure 8.45*:

![](img/bd699229-166a-4da7-bcb8-ed8928c9fd38.png)

Figure 8.45: SVM classifier basic image (Image credit: http://docs.opencv.org/2.4/_images/optimal-hyperplane.png)

In the given figure, there are three lines *a*, *b*, and *c*. Now, choose the line that you think best separates the data points. I would pick line *b* because it maximizes the margin from two classes and the other lines *a* and *c* don't do that.

Note that SVM first tries to perform the classification task perfectly and then tries to maximize the margin. So for SVM, performing the classification task correctly is the first priority. SVM can obtain the linear hyperplane as well as generate a non-linear hyperplane. So, let's understand the math behind this.

If you have n features, then using SVM, you can draw *n-1* dimensional hyperplane. If you have a two-dimensional feature space, then you can draw a hyperplane that is one-dimensional. If you have a three-dimensional feature space, then you can draw a two-dimensional hyperplane. In any ML algorithm, we actually try minimizing our loss function, so we first define the loss function for SVM. SVM uses the hinge loss function. We use this loss function and try to minimize our loss and obtain the maximum margin for our hyperplane. The hinge loss function equation is given as follows:

*C (x, y, f(x)) = (1 - y * f(x))[+]*

Here, *x* is sample data points, *y* is true label, *f(x)* is the predicted label, and *C* is the loss function. What the *+* sign in the equation denotes is, when we calculate *y*f(x)* and it comes *> = 1*, then we try to subtract it from *1* and get a negative value. We don't want this, and so to denote that, we put the *+* sign:

![](img/0cb54ce8-ce2b-4da9-892a-44b19176c8c1.png)

Now it's time to define the objective function that takes the loss function, as well as a lambda term called a **regularization term**. We will see what it does for us. However, it is also a tuning parameter. See the mathematics equation in F*igure 8.46*:

![](img/2b01145c-e3f4-4c4a-967c-aefc6dbe41e3.png)

Figure 8.46: Objective function with regularization term lambda

SVM has two tuning parameters that we need to take care of. One of the terms is the lambda that denotes the regularization term. If the regularization term is too high, then our ML-model overfits and cannot generalize the unseen data. If it is too low, then it underfits and we get a huge training error. So, we need a precise value for the regularization term as well. We need to take care of the regularization term that helps us prevent overfitting and we need to minimize our loss. So, we take a partial derivative for both of these terms; the following are the derivatives for the regularization term and loss function that we can use to perform gradient descent so that we can minimize our loss and get an accurate regularization value. See the partial derivative equation in *Figure 8.47*:

![](img/ae9b1450-205a-4a82-a91a-580ee5b89769.jpg)

Figure 8.47: Partial derivative for regularization term

See the partial derivative for the loss function in *Figure 8.48*:

![](img/d9bf34f5-7be6-480d-9f3a-c4f8ac58c711.jpg)

Figure 8.48: Partial derivative for loss function

We need to calculate the values of the partial derivative and update the weight accordingly. If we misclassify the data point, then we need to use the following equation to update the weight. Refer to F*igure 8.49*:

![](img/378e7ef1-0c0b-4f48-9170-d8252ee97f27.jpg)

Figure 8.49: Misclassification condition

So if *y* is *< 1*, then we need to use the following equation in *Figure 8.50:*

![](img/0279fae5-b9b4-4f0c-95d5-ce34c9420519.jpg)

Figure 8.50: Weight update rule using this equation for the misclassification condition

Here, the long n shape is called eta and it denotes the learning rate. Learning rate is a tuning parameter that shows you how fast your algorithm should run. This also needs an accurate value because, if it is too high, then the training will complete too fast and the algorithm will miss the global minimum. On the other hand, if it is too slow, then it will take too much time to train and may never converge at all.

If misclassification happens, then we need to update our loss function as well as the regularization term.

Now, what if the algorithm correctly classifies the data point? In this case, we don't need to update the loss function; we just need to update our regularization parameter that you can see using the equation given in *Figure 8.51*:

![](img/bd3e2dcd-ca70-4b8d-a2e3-d3a39b860fe6.jpg)

Figure 8.51: Updating the weight for regularization

When we have an appropriate value of regularization and the global minima, then we can classify all the points in SVM; at that time, the margin value also becomes the maximum.

If you want to use SVM for a non-linear classifier, then you need to apply the kernel trick. Briefly, we can say that the kernel trick is all about converting lower feature space into higher feature space that introduces the non-linear attributes so that you can classify the dataset. See the example in *Figure 8.52*:

![](img/3dc8b638-a228-4517-b19b-78c5d4961dc9.png)

Figure 8.52: Non-linear SVM example

To classify this data, we have **X**, **Y** feature. We introduce the new non-linear feature, *X² + Y²,* which helps us draw a hyperplane that can classify the data correctly.

So, now it's time to implement the SVM algorithm and we will develop the sentiment analysis application again, but this time, I'm using SVM and seeing what the difference is in the accuracy. You can find the code on this GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch8/sentimentanalysis/sentimentanalysis_SVM.py](https://github.com/jalajthanaki/NLPython/blob/master/ch8/sentimentanalysis/sentimentanalysis_SVM.py)

You can see the code snippet in *Figure 8.53*:

![](img/a70fbec9-193c-46a6-8830-64cc0e84aece.png)

Figure 8.53: Sentiment analysis using SVM

You can find the output in *Figure 8.54*:

![](img/c885237b-74fb-470f-808a-254e1479cf8d.png)

Figure 8.54: Output of SVM

Now it's time to look at some tuning parameters. Let's take a look!

**Tunable parameters**

Let's check out some of the SVM tuning parameters that can help us:

*   `scikit-learn` provides a tuning parameter for the kernel trick that is very useful. You can use various types of kernels, such as linear, rbf, and so on.
*   There are other parameters called **C** and **gamma**.
*   C controls the trade-off between a smooth decision boundary and classifying the training points correctly. A large value of C gives you more correct training points.
*   Gamma can be useful if you are trying to set your margin. If you set high values for the gamma, then only nearby data points are taken into account to draw a decision boundary, and if you have low values for gamma, then points that are far from the decision boundary are also taken into account to measure whether the decision boundary maximizes the margin or not.

Now it's time to look at the advantages and disadvantages of SVM.

**Advantages of SVM**

The following are the advantages that the SVM algorithm provides to us:

*   It performs well for complicated datasets
*   It can be used for a multiclass classifier

**Disadvantages of SVM**

The following are the disadvantages of the SVM algorithm:

*   It will not perform well when you have a very large dataset because it takes a lot of training time
*   It will not work effectively when the data is too noisy

This is the end of the supervised ML algorithms. You learned a lot of math and concepts, and if you want to explore more, then you can try out the following exercise.

# Exercise

*   You need to explore **K-Nearest Neighbor** (**KNN**) and its application in the NLP domain
*   You need to explore AdaBoost and its application in the NLP domain

We have covered a lot of cool classification techniques used in NLP and converted the black box ML algorithms to white box. So, now you know what is happening inside the algorithms. We have developed NLP applications as well, so now this is the time to jump into unsupervised ML.

# Unsupervised ML

This is another type of machine learning algorithm. When we don't have any labeled data, then we can use unsupervised machine learning algorithms. In the NLP domain, there is a common situation where you can't find the labeled dataset, then this type of ML algorithm comes to our rescue.

Here, we will discuss the unsupervised ML algorithm called K-means clustering. This algorithm has many applications. Google has used this kind of unsupervised learning algorithms for so many of their products. YouTube video suggestions use the clustering algorithm.

The following image will give you an idea of how data points are represented in unsupervised ML algorithms. Refer to F*igure 8.55*:

![](img/b20b3aa2-e7f2-4b45-854e-e03feb44867d.png)

Figure 8.55: A general representation of data points in an unsupervised ML algorithm

As you can see in *Figure 8.55*, the data points don't have a label associated with them, but visually, you can see that they form some groups or clusters. We will actually try to figure out the structure in the data using unsupervised ML algorithms so that we can derive some fruitful insight for unseen data points.

Here, we will look at the k-means clustering algorithm and develop the document classification example that is related to the NLP domain. So, let's begin!

# k-means clustering

In this section, we will discuss the k-means clustering algorithm. We will get an understanding of the algorithm first. k-means clustering uses the iterative refinement technique.

Let's understand some of the basics about the k-means algorithm. k refers to how many clusters we want to generate. Now, you can choose a random point and put the centroid at this point. The number of centroids in k-means clustering is not more than the value of k, which means not more than the cluster value k.

This algorithm has the two following steps that we need to reiterate:

1.  The first step is to assign the centroid.
2.  The second step is to calculate the optimization step.

To understand the steps of the k-means, we will look at an example. Before that, I would like to recommend that you check out this animated image that will give you a lot of understanding about k-means:

[https://github.com/jalajthanaki/NLPython/blob/master/ch8/K_means_clustering/K-means_convergence.gif](https://github.com/jalajthanaki/NLPython/blob/master/ch8/K_means_clustering/K-means_convergence.gif)[.](https://github.com/jalajthanaki/NLPython/blob/master/ch8/K_means_clustering/K-means_convergence.gif)

Now, let's take an example. You have five data points, which are given in the table, and we want to group these data points into two clusters, so *k = 2*. Refer to *Figure 8.56*:

![](img/5cc4774c-4e6d-419b-a363-650d16e86ea0.png)

Figure 8.56: k-mean clustering data points for calculation

We have chosen point *A(1,1)* and point *C(0,2)* for the assignment of our centroid. This is the end of the assign step, now let's understand the optimization step.

We will calculate the Euclidean distance from every point to this centroid. The equation for the Euclidean distance is given in *Figure 8.57*:

![](img/9ecf9d67-81c4-424a-b0aa-81da3dc1f3f7.jpg)

Figure 8.57: Euclidean distance for the k-means clustering algorithm

Every time, we need to calculate the Euclidean distance from both centroids. Let's check the calculation. The starting centroid mean is *C1 = (1,1)* and *C2 = (0,2)*. Here, we want to make two cluster that is the reason we take two centroids.

**Iteration 1**

For point *A = (1,1)*:

*C1 = (1,1)* so *ED = Square root ((1-1)² + (1-1)²) = 0*

*C2 = (0,2)* so *ED = Square root ((1-0)² + (1-2)²) = 1.41*

Here, *C1 < C2*, so point *A* belongs to cluster 1.

For point *B = (1,0)*:

*C1 = (1,1)* so *ED = Square root ((1-1)² + (0-1)²) = 1*

*C2 = (0,2)* so *ED = Square root ((1-0)² + (0-2)²) = 2.23*

Here, *C1 < C2*, so point *B* belongs to cluster 1.

For point *C = (0,2)*:

*C1 = (1,1)* so *ED = Square root ((0-1)² + (2-1)²) = 1.41*

*C2 = (0,2)* so *ED = Square root ((0-0)² + (2-2)²) = 0*

Here, *C1 > C2*, so point *C* belongs to cluster 2.

For point *D = (2,4)*:

*C1 = (1,1)* so *ED = Square root ((2-1)² + (4-1)²) = 3.16*

*C2 = (0,2)* so *ED = Square root ((2-0)² + (4-2)²) = 2.82*

Here, *C1 > C2*, so point *C* belongs to cluster 2.

For point *E = (3,5)*:

*C1 = (1,1)* so *ED = Square root ((3-1)² + (5-1)²)= 4.47*

*C2 = (0,2)* so *ED = Square root ((3-0)² + (5-2)²)= 4.24*

Here, *C1 > C2*, so point *C* belongs to cluster 2.

After the first iteration, our cluster looks as follows. Cluster *C1* has points *A* and *B*, and *C2* has points *C*, *D*, and *E*. So, here, we need to calculate the centroid mean value again, as per the new cluster point:

*C1 = XA + XB / 2 = (1+1) / 2 = 1*

*C1 = YA + YB / 2 = (1+0) / 2 = 0.5*

So new *C1 = (1,0.5)*

*C2 = Xc + XD + XE / 3 = (0+2+3) / 3 = 1.66*

*C2 = Yc +YD + YE / 3 = (2+4+5) / 3 = 3.66*

So new *C2 = (1.66,3.66)*

We need to do all the calculations again in the same way as *Iteration 1*. So we get the values as follows.

**Iteration 2**

For point *A = (1,1)*:

*C1 = (1,0.5)* so *ED = Square root ((1-1)² + (1-0.5)²) = 0.5*

*C2 = (1.66,3.66)* so *ED = Square root ((1-1.66)² + (1-3.66)²) = 2.78*

Here, *C1 < C2*, so point *A* belongs to cluster 1.

For point *B = (1,0)*:

*C1 = (1,0.5)* so *ED = Square root ((1-1)² + (0-0.5)²) = 1*

*C2 = (1.66,3.66)* so *ED = Square root ((1-1.66)² + (0-3.66)²) = 3.76*

Here, *C1 < C2*, so point *B* belongs to cluster 1.

For point *C = (0,2)*:

*C1 = (1,0.5)* so *ED = Square root ((0-1)² + (2-0.5)²)= 1.8*

*C2 = (1.66, 3.66)* so *ED = Square root ((0-1.66)² + (2-3.66)²)= 2.4*

Here, *C1 < C2*, so point *C* belongs to cluster 1.

For point *D = (2,4)*:

*C1 = (1,0.5)* so *ED = Square root ((2-1)² + (4-0.5)²)= 3.6*

*C2 = (1.66,3.66)* so *ED = Square root ((2-1.66)² + (4-3.66)²)= 0.5*

Here, *C1 > C2*, so point *C* belongs to cluster 2.

For point *E = (3,5)*:

*C1 = (1,0.5)* so *ED = Square root ((3-1)² + (5-0.5)²) = 4.9*

*C2 = (1.66,3.66)* so *ED = Square root ((3-1.66)² + (5-3.66)²) = 1.9*

Here, *C1 > C2*, so point *C* belongs to cluster 2.

After the second iteration, our cluster looks as follows. *C1* has points *A*, *B*, and *C*, and *C2* has points *D* and *E*:

*C1 = XA + XB + Xc / 3 = (1+1+0) / 3 = 0.7*

*C1 = YA + YB + Yc / 3 = (1+0+2 ) / 3 = 1*

So new *C1 = (0.7,1)*

*C2 = XD + XE / 2 = (2+3) / 2 = 2.5*

*C2 = YD + YE / 2 = (4+5) / 2 = 4.5*

So new *C2 = (2.5,4.5)*

We need to do iterations until the clusters don't change. So this is the reason why this algorithm is called an **iterative algorithm**. This is the intuition of the K-means clustering algorithm. Now we will look at a practical example in the document classification application.

# Document clustering

Document clustering helps you with a recommendation system. Suppose you have a lot of research papers and you don't have tags for them. You can use the k-means clustering algorithm, which can help you to form clusters as per the words appearing in the documents. You can build an application that is news categorization. All news from the same category should be combined together; you have a superset category, such as sports news, and this sports news category contains news about cricket, football, and so on.

Here, we will categorize movies into five different genres. The code credit goes to Brandon Rose. You can check out the code on this GitHub link:

[https://github.com/jalajthanaki/NLPython/blob/master/ch8/K_means_clustering/K-mean_clustering.ipynb](https://github.com/jalajthanaki/NLPython/blob/master/ch8/K_means_clustering/K-mean_clustering.ipynb).

See the code snippet in *Figure 8.58*:

![](img/664dc2ba-fce4-4862-a062-af382d9c57d4.png)

Figure 8.58: A short code snippet of the K-means algorithm

See the output in F*igure 8.59*:

![](img/80fbf996-3b27-41fd-972c-a9d442b44c18.png)

Figure 8.59: Output of k-means clustering

You can refer to this link for hierarchical clustering:

[http://brandonrose.org/clustering](http://brandonrose.org/clustering).

# Advantages of k-means clustering

These are the advantages that k-means clustering provides us with:

*   It is a very simple algorithm for an NLP application
*   It solves the main problem as it doesn't need tagged data or result labels, you can use this algorithm for untagged data

# Disadvantages of k-means clustering

These are the disadvantages of k-means clustering:

*   Initialization of the cluster center is a really crucial part. Suppose you have three clusters and you put two centroids in the same cluster and the other one in the last cluster. Somehow, k-means clustering minimizes the Euclidean distance for all the data points in the cluster and it will become stable, so actually, there are two centroids in one cluster and the third one has one centroid. In this case, you end up having only two clusters. This is called the **local minimum** problem in clustering.

This is the end of the unsupervised learning algorithms. Here, you have learned about the k-means clustering algorithm and developed the document classification application. If you want to learn more about this technique, try out the exercise.

# Exercise

You need to explore hierarchical clustering and its application in the NLP domain.

Our next section is very interesting. We will look at semi-supervised machine learning techniques. Here, we will get an overview of them. So, let's understand these techniques.

# Semi-supervised ML

**Semi-supervised ML** or **semi-supervised learning** (**SSL**) is basically used when you have a training dataset that has a target concept or target label for some data in the dataset, and the other part of the data doesn't have any label. If you have this kind of dataset, then you can apply semi-supervised ML algorithms. When we have a very small amount of labeled data and a lot of unlabeled data, then we can use semi-supervised techniques. If you want to build an NLP tool for any local language (apart from English) and you have a very small amount of labeled data, then you can use the semi-supervised approach. In this approach, we will use a classifier that uses the labeled data and generates an ML-model. This ML-model is used to generate labels for the unlabeled dataset. The classifiers are used for high-confidence predictions on the unlabeled dataset. You can use any appropriate classifier algorithm to classify the labeled data.

Semi-supervised techniques are a major research area, especially for NLP applications. Last year, Google Research developed semi supervised techniques that are graph-based:

[https://research.googleblog.com/2016/10/graph-powered-machine-learning-at-google.html](https://research.googleblog.com/2016/10/graph-powered-machine-learning-at-google.html).

For better understanding, you can also read some of the really interesting stuff given here:
[https://medium.com/@jrodthoughts/google-expander-and-the-emergence-of-semi-supervised-learning-1919592bfc49](https://medium.com/@jrodthoughts/google-expander-and-the-emergence-of-semi-supervised-learning-1919592bfc49).
[https://arxiv.org/ftp/arxiv/papers/1511/1511.06833.pdf](https://arxiv.org/ftp/arxiv/papers/1511/1511.06833.pdf).
[http://www.aclweb.org/anthology/W09-2208](http://www.aclweb.org/anthology/W09-2208).
[http://cogprints.org/5859/1/Thesis-David-Nadeau.pdf](http://cogprints.org/5859/1/Thesis-David-Nadeau.pdf).
[https://www.cmpe.boun.edu.tr/~ozgur/papers/896_Paper.pdf](https://www.cmpe.boun.edu.tr/~ozgur/papers/896_Paper.pdf).
[http://graph-ssl.wdfiles.com/local--files/blog%3A_start/graph_ssl_acl12_tutorial_slides_final.pdf](http://graph-ssl.wdfiles.com/local--files/blog%3A_start/graph_ssl_acl12_tutorial_slides_final.pdf).

Those interested in research can develop novel SSL techniques for any NLP application.

Now we have completed our ML algorithms section. There are some critical points that we need to understand. Now it's time to explore these important concepts.

# Other important concepts

In this section, we will look at those concepts that help us know how the training on our dataset using ML algorithms is going, how you should judge whether the generated ML-model will be able to generalize unseen scenarios or not, and what signs tell you that your ML-model can't generalize the unseen scenarios properly. Once you detect these situations, what steps should you take? What are the widely used evaluation matrices for NLP applications?

So, let's find answers to all these questions. I'm going to cover the following topics. We will look at all of them one by one:

*   Bias-variance trade-off
*   Underfitting
*   Overfitting
*   Evaluation matrix

# Bias-variance trade-off

Here, we will look at a high-level idea about the bias-variance trade-off. Let's understand each term one by one.

Let's first understand the term bias. When you are performing training using an ML algorithm and you see that your generated ML-model doesn't perform differently with respect to your first round of training iteration, then you can immediately recognize that the ML algorithm has a high bias. In this situation, ML algorithms have no capacity to learn from the given data so it's not learning new things that you expect your ML algorithm to learn. If your algorithm has very high bias, then eventually it just stops learning. Suppose you are building a sentiment analysis application and you have come up with the ML-model. Now you are not happy with the ML-model's accuracy and you want to improve the model. You will train by adding some new features and changing some algorithmic parameters. Now this newly generated model will not perform well or perform differently on the testing data, which is an indication for you that you may have high bias. Your ML algorithm won't converge in the expected way so that you can improve the ML-model result.

Let's understand the second term, variance. So, you use any ML algorithm to train your model and you observe that you get very good training accuracy. However, you apply the same ML-model to generate the output for an unseen testing dataset and your ML-model doesn't work well. This situation, where you have very good training accuracy and the ML-model doesn't turn out well for the unseen data, is called a **high variance** situation. So, here, the ML-model can only replicate the predictions or output that it has seen in the training data and doesn't have enough bias that it can generalize the unseen situation. In other words, you can say that your ML algorithm is trying to remember each of the training examples and, at the end, it just mimics that output on your testing dataset. If you have a high variance problem, then your model converges in such a way that it tries to classify each and every example of the dataset in a certain category. This situation leads us to overfitting. I will explain what overfitting is, so don't worry! We will be there in a few minutes.

To overcome both of the preceding bad situations, we really need something that lies in the middle, which means no high bias and no high variance. The art of generating the most bias and best variance for ML algorithms leads to the best optimal ML-Model. Your ML-model may not be perfect, but it's all about generating the best bias-variance trade-off.

In the next section, you will learn the concepts of *Underfitting* and *Overfitting* as well as tricks that help you get rid of these high bias and high variance scenarios.

# Underfitting

In this section, we will discuss the term underfitting. What is underfitting and how is it related to the bias-variance trade-off?

Suppose you train the data using any ML algorithm and you get a high training error. Refer to *Figure 8.60*:

![](img/dd8bffe6-9815-44f7-aac3-69da10bd7ebd.png)

Figure 8.60: Graph indicating a high training error (Image credit: http://www.learnopencv.com/wp-content/uploads/2017/02/bias-variance-tradeoff.png)

The preceding situation, where we get a very high training error, is called **underfitting**. ML algorithms just can't perform well on the training data. Now, instead of a linear decision boundary, we will try a higher degree of polynomials. Refer to *Figure 8.61*:

![](img/fdd3ffb3-49e1-438b-9d58-0f287f2df8a3.png)

Figure 8.61: High bias situation (Image credit: http://www.learnopencv.com/wp-content/uploads/2017/02/bias-variance-tradeoff.png)

This graph has a very squiggly line and it cannot do well on the training data. In other words, you can say that it is performing the same as per the previous iteration. This shows that the ML-model has a high bias and doesn't learn new things.

# Overfitting

In this section, we will look at the term overfitting. I put this term in front of you when I was explaining variance in the last section. So, it's time to explain overfitting and, in order to explain it, I want to take an example.

Suppose we have a dataset and we plot all the data points on a two dimensional plane. Now we are trying to classify the data and our ML algorithm draws a decision boundary in order to classify the data. You can see *Figure 8.62*:

![](img/711503bc-3aa2-4e84-a68f-76edf05d2df4.png)

Figure 8:62 Overfitting and variance

(Image credit: http://www.learnopencv.com/wp-content/uploads/2017/02/bias-variance-tradeoff-test-error.png)

If you look at the left-hand side graph, then you will see the linear line used as a decision boundary. Now, this graph shows a training error, so you tune the parameter in your second iteration and you will get really good training accuracy; see the right-hand side graph. You hope that you will get good testing accuracy as well for the testing data, but the ML-model does really bad on the testing data prediction. So this situation, where an algorithm has very good training accuracy but doesn't perform well on the testing data, is called **overfitting**. This is the situation where ML-models have high variance and cannot generalize the unseen data.

Now that you have seen underfitting and overfitting, there are some rules of thumb that will help you so that you can prevent these situations. Always break your training data into three parts:

*   60% of the dataset should be considered as training dataset
*   20% of the dataset should be considered as validation dataset or development dataset, which will be useful in getting intermediate accuracy for your ML algorithm so that you can capture the unexpected stuff and change your algorithm according to this
*   20% of the dataset should be held out to just report the final accuracy and this will be the testing dataset

You should also apply k-fold cross validation. k indicates how many times you need the validation. Suppose we set it to three. We divide our training data into three equal parts. In the first timestamp of the training algorithm, we use two parts and test on a single part so technically, it will train on 66.66% and will be tested on 33.34%. Then, in the second timestamp, the ML algorithm uses one part and performs testing on two parts, and at the last timestamp, it will use the entire dataset for training as well as testing. After three timestamps, we will calculate the average error to find out the best model. Generally, for a reasonable amount of the dataset, k should be taken as 10.

You cannot have 100% accuracy for ML-models and the main reason behind this is because there is some noise in your input data that you can't really remove, which is called **irreducible error**.

So, the final equation for error in an ML algorithm is as follows:

*Total error = Bias + Variance + Irreducible Error*

You really can't get rid of the irreducible error, so you should concentrate on bias and variance. Refer to *Figure 8.63*, which will be useful in showing you how to handle the bias and variance trade-off:

![](img/78840f2e-bd6c-4f8c-b3ad-da68d8db83cc.png)

Figure 8.63: Steps to get rid of high bias or high variance situations (Image credit: http://www.learnopencv.com/wp-content/uploads/2017/02/Machine-Learning-Workflow.png)

Now that we have seen enough about ML, let's look at the evaluation matrix, that is quite useful.

# Evaluation matrix

For our code, we check the accuracy but we really don't understand which attributes play a major part when you evaluate an ML-model. So, here, we will consider a matrix that is widely used for NLP applications.

This evaluation matrix is called **F1 score** or **F-measure**. It has three main components; before this, let's cover some terminology:

*   **True positive** (**TP**): This is a data point labeled as A by the classifier and is from class A in reality.
*   **True Negative** (**TN**): This is an appropriate rejection from any class in the classifier, which means that the classifier won't classify data points into class A randomly, but will reject the wrong label.
*   **False Positive** (**FP**): This is called a **type-I error** as well. Let's understand this measure by way of an example: A person gives blood for a cancer test. He actually doesn't have cancer but his test result is positive. This is called FP.
*   **False Negative** (**FN**): This is called a **type-II error** as well. Let's understand this measure by way of an example: A person gives blood for a cancer test. He has cancer but his test result is negative. So it actually overlooks the class labels. This is called FN.
*   **Precision**: Precision is a measure of exactness; what percentage of data points the classifier labeled as positive and are actually positive:
    *precision=TP / TP + FP*
*   **Recall**: Recall is the measure of completeness; what percentage of positive data points did the classifier label as positive:
    *Recall = TP / TP + FN*
*   **F measure**: This is nothing but the weighed measure of precision and recall. See the equation:
    F= 2 * precision * recall / precision + recall

Apart from this, you can use a confusion matrix to know each of the TP, TN, FP, and FN. You can use the area under an ROC curve that indicates how much your classifier is capable of discriminating between negative and positive classes. *ROC = 1.0* represents that the model predicted all the classes correctly; area of 0.5 represents that a model is just making random predictions.

If you want to explore the new terms and techniques, you can do the following exercise.

# Exercise

Read about undersampling and oversampling techniques.

Now it's time to understand how we can improvise our model after the first iteration, and sometimes, feature engineering helps us a lot in this. In [Chapter 5](07f71ca1-6c8a-492d-beb3-a47996e93f04.xhtml), *Feature Engineering and NLP Algorithms* and [Chapter 6](c4861b9e-2bcf-4fce-94d4-f1e2010831de.xhtml), *Advance Feature Engineering and NLP Algorithms,* we explained how to extract features from text data using various NLP concepts and statistical concepts as part of feature engineering. Feature engineering includes feature extraction and feature selection. Now it's time to explore the techniques that are a part of feature selection. Feature extraction and feature selection give us the most important features for our NLP application. Once we have these features set, you can use various ML algorithms to generate the final outcome.

Let's start understanding the feature selection part.

# Feature selection

As I mentioned earlier, feature extraction and feature selection are a part of feature engineering, and in this section, we will look at feature selection. You might wonder why we are learning feature selection, but there are certain reasons for it and we will look at each of them. First, we will see basic understanding of feature selection.

Features selection is also called variable selection, attribute selection, or variable subset selection. Features selection is the process of selecting the best relevant features, variables, or data attributes that can help us develop more efficient machine learning models. If you can identify which features contribute a lot and which contribute less, you can select the most important features and remove other less important ones.

Just take a step back and first understand what the problems are that we are trying to solve using features selection.

Using features selection techniques, we can get the following benefits:

*   Selecting the relevant and appropriate features will help you simplify your ML-model. This will help you interpret the ML-Model easily, as well as reduce the complexity of the ML-Model.
*   Choosing appropriate features using feature selection techniques will help us improve our ML-Models accuracy.
*   Feature selection helps the machine learning algorithms train faster.
*   Feature selection also prevents overfitting.
*   It helps us get rid of the curse of dimensionality.

# Curse of dimensionality

Let's understand what I mean by the curse of dimensionality because this concept will help us understand why we need feature selection techniques. The curse of dimensionality says that, as the number of features or dimensions grows, which means adding new features to our machine learning algorithm, then the amount of data that we need to generalize accurately grows exponentially. Let's see with an example.

Suppose you have a line, one-dimensional feature space, and we put five points on that line. You can see that each point takes some space on this line. Each point takes one-fifth of the space on the line. Refer to *Figure 8.64*:

![](img/80a6664c-574a-4a71-98a0-c9950f2a8371.png)

Figure 8.64: A one-dimensional features space with five data points

If you have two-dimensional feature space, then we need more than five data points to fill up this space. So, we need 25 data points for these two dimensions. Now each point is taking up 1/25 of the space. See *Figure 8.65*:

![](img/56867430-d6f1-4c6c-9adb-50da1672a48a.png)

Figure 8.65: A two-dimensional features space with 25 data points

If you have a three-dimensional feature space, which means that we have three features, then we need to fill up the cube. You can see this in *Figure 8.66*:

![](img/356be119-7aef-4f17-bf71-1fcab63a3243.png)

Figure 8.66: A three-dimensional features space with 1/25 data points

However, to fill up the cube, you need exactly 125 data points, as you can see in *Figure 8.66* (assume there are 125 points). So every time we add features, we need more data. I guess you will all agree that the growth in data points increases exponentially from 5, 25, 125, and so on. So, in general, you need *Xd* feature space, where *X* is your number of data points in training and *d* is the number of features or dimensions. If you just blindly put more and more features so that your ML algorithm gets a better understanding of the dataset, what you're actually doing is forcing your ML algorithm to fill the larger features space with data. You can solve this using a simple method. In this kind of situation, you need to give more data to your algorithm, not features.

Now you really think I'm restricting you to adding new features. So, let me clarify this for you. If it is necessary to add features, then you can; you just need to select the best and minimum amount of features that help your ML algorithm learn from it. I really recommend that you don't add too many features blindly.

Now, how can we derive the best features set? What is the best features set for the particular application that I'm building and how can I know that my ML algorithm will perform well with this features set? I will provide the answers to all these questions in the next section *Features selection techniques*. Here, I will give you a basic idea about feature selection techniques. I would recommend that you practically implement them in the NLP applications that we have developed so far.

# Feature selection techniques

Make everything as simple as possible but not simpler

This quote by Albert Einstein, is very true when we are talking about feature selection techniques. We have seen that to get rid of the curse of dimensionality, we need feature selection techniques. We will look at the following feature selection techniques:

*   Filter method
*   Wrapper method
*   Embedded method

So, let's begin with each method.

**Filter method**

Feature selection is altogether a separate activity and independent of the ML algorithm. For a numerical dataset, this method is generally used when we are preprocessing our data, and for the NLP domain, it should be performed once we convert text data to a numerical format or vector format. Let's first see the basic steps of this method in *Figure 8.67*:

![](img/2290d45b-3fa3-45d5-bb44-2a9cb6240912.png)

Figure 8.67: Filter method for feature selection (Image credit: https://upload.wikimedia.org/wikipedia/commons/2/2c/Filter_Methode.png)

The steps are very clear and self-explanatory. Here, we use statistical techniques that give us a score and based on this, we will decide whether we should keep the feature or just remove it or drop it. Refer to *Figure 8.68*:

![](img/66d34e05-93ba-47b1-b49e-b622806ab67c.png)

Figure 8.68: Features selection techniques list

Let me simplify *Figure 8.68* for you:

*   If feature and response are both continuous, then we will perform correlation
*   If feature and response are both are categorical, then we will use Chi-Square; in NLP (we mostly use this)
*   If feature are continuous and response is categorical, then we will use **linear discriminant analysis** (**LDA**)
*   If feature are categorical and response is continuous, then we will use Anova

I will concentrate more on the NLP domain and explain the basics of LDA and Chi-Square.

LDA is generally used to find a linear combination of features that characterize or separate more than one class of a categorical variable, whereas Chi-Square is mainly used in NLP as compared to LDA. Chi-Square is applied to a group of categorical features to get an idea of the likelihood of correlation or association between features using their frequency distribution.

**Wrapper method**

In this method, we are searching for the best features set. This method is computationally very expensive because we need to search for the best features subset for every iteration. See the basic steps in *Figure 8.69*:

![](img/fc0c4985-6bf8-4ea4-84d2-2d6bacddddea.png)

Figure 8.69: Wrapper method steps (Image credit: https://upload.wikimedia.org/wikipedia/commons/0/04/Feature_selection_Wrapper_Method.png)

There are three submethods that we can use to select the best features subset:

*   Forward selection
*   Backward selection
*   Recursive features elimination

In forward selection, we start with no features and add the features that improve our ML-model, in each iteration. We continue this process until our model does not improve its accuracy further.

Backward selection is the other method where we start with all the features and, in each iteration, we find the best features and remove other unnecessary features, and repeat until no further improvement is observed in the ML-model.

Recursive feature elimination uses a greedy approach to find out the best performing features subset. It repeatedly creates the models and keeps aside the best or worst performing features for each iteration. The next time, it uses the best features and creates the model until all features are exhausted; lastly, it ranks the features based on its order of elimination of them.

**Embedded method**

In this method, we combine the qualities of the filter and wrapper methods. This method is implemented by the algorithms that have their own built-in feature selection methods. Refer to F*igure 8.70*:

![](img/e92eddc2-b32a-42c2-ac6a-b510690f9140.png)

Figure 8.70: Embedded features selection method (Image credit: https://upload.wikimedia.org/wikipedia/commons/b/bf/Feature_selection_Embedded_Method.png)

The most popular examples of these methods are LASSO and RIDGE regression, which has some built-in parameters to reduce the chances of overfitting.

You can refer to these links, which will be very useful for you:
[https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/).
[http://machinelearningmastery.com/an-introduction-to-feature-selection/](http://machinelearningmastery.com/an-introduction-to-feature-selection/).
[http://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/](http://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/).

We will look at dimensionality reduction in the next section.

# Dimensionality reduction

Dimensionality reduction is a very useful concept in machine learning. If we include a lot of features to develop our ML-model, then sometimes we include features that are really not needed. Sometimes we need high-dimensional features space. What are the available ways to make certain sense about our features space? So we need some techniques that help us remove unnecessary features or convert our high-dimensional features space to two-dimensional or three-dimensional features so that we can see what all is happening. By the way, we have used this concept in [Chapter 6](c4861b9e-2bcf-4fce-94d4-f1e2010831de.xhtml), *Advance Features Engineering and NLP* *Algorithms*, when we developed an application that generated word2vec for the game of thrones dataset. At that time, we used **t-distributed stochastic neighbor embedding** (**t-SNE**) dimensionality reduction technique to visualize our result in two-dimensional space.

Here, we will look at the most famous two techniques, called **principal component analysis** (**PCA**) and t-SNE, which is used to visualize high-dimensional data in two-dimensional space. So, let's begin.

**PCA**

PCA is a statistical method that uses an orthogonal transformation to convert a set of data points of possibly correlated features to a set of values of linearly uncorrelated features, called **principal components**. The number of principal components is less than or equal to the number of the original features. This technique defines transformation in such a way that the first principal component has the largest possible variance to each succeeding feature.

Refer to *Figure 8.71*:

![](img/e7686ca9-493b-4208-a31f-14e82363e109.png)

Figure 8.71: PCA (Image credit: https://www.nature.com/article-assets/npg/nbt/journal/v26/n3/images/nbt0308-303-F1.gif)

This graph helps a lot in order to understand PCA. We have taken two principal components and they are orthogonal to each other as well as making variance as large as possible. In *c* graph, we have reduced the dimension from two-dimensional to one-dimensional by projecting on a single line.

The disadvantage of PCA is that when you reduce the dimensionality, it loses the meaning that the data points represent. If interpretability is the main reason for dimensionality reduction, then you should not use PCA; you can use t-SNE.

**t-SNE**

This is the technique that helps us visualize high-dimensional non-linear space. t-SNE tries to preserve the group of local data points that are close together. This technique will help you when you want to visualize high-dimensional space. You can use this to visualize applications that use techniques such as word2vec, image classification, and so on. For more information, you can refer to this link:

[https://lvdmaaten.github.io/tsne/](https://lvdmaaten.github.io/tsne/).

# Hybrid approaches for NLP applications

Hybrid approaches sometimes really help us improve the result, of our NLP applications. For example, if we are developing a grammar correction system, a module that identifies multiword expressions such as kick the bucket, and a rule-based module that identifies the wrong pattern and generates the right pattern. This is one kind of hybrid approach. Let's take a second example for the same NLP application. You are making a classifier that identifies the correct articles (determiners - a, an, and the) for the noun phrase in a sentence. In this system, you can take two categories - a/an and the. We need to develop a classifier that will generate the determiner category, either a/an or the. Once we generate the articles for the noun phrase, we can apply a rule-based system that further decides the actual determiner for the first category a/an. We also know some English grammar rules that we can use to decide whether we should go with a or an. This is also an example a of hybrid approach. For better sentiment analysis, we can also use hybrid approaches that include lexical-based approach, ML-based approach, or word2vec or GloVe pretrained models to get really high accuracy. So, be creative and understand your NLP problem so that you can take advantages from different types of techniques to make your NLP application better.

# Post-processing

Post processing is a kind of rule-based system. Suppose you are developing a machine translation application and your generated model makes some specific mistakes. You want that **machine translation** (**MT**) model to avoid these kinds of mistakes, but avoiding that takes a lot of features that make the training process slow and make the model too complex. On the other hand, if you know that there are certain straightforward rules or approximations that can help you once the output has been generated in order to make it more accurate, then we can use post-processing for our MT model. What is the difference between a hybrid model and post-processing? Let me clear your confusion. In the given example, I have used word approximation. So rather than using rules, you can also apply an approximation, such as applying a threshold value to tune your result, but you should apply approximation only when you know that it will give an accurate result. This approximation should complement the NLP system to be generalized enough.

# Summary

In this chapter, we have looked at the basic concepts of ML, as well as the various classification algorithms that are used in the NLP domain. In NLP, we mostly use classification algorithms, as compared to linear regression. We have seen some really cool examples such as spam filtering, sentiment analysis, and so on. We also revisited the POS tagger example to provide you with better understanding. We looked at unsupervised ML algorithms and important concepts such as bias-variance trade-off, underfitting, overfitting, evaluation matrix, and so on. We also understood features selection and dimensionality reduction. We touched on hybrid ML approaches and post-processing as well. So, in this chapter, we have mostly understood how to develop and fine-tune NLP applications.

In the next chapter, we will see a new era of machine learning--deep learning. We will explore the basic concepts needed for AI. After that, we will discuss the basics of deep learning including linear regression and gradient descent. We will see why deep learning has become the most popular technique in the past few years. We will see the necessary concepts of math that are related to deep learning, explore the architecture of deep neural networks, and develop some cool applications such as machine translation from the NLU domain and text summarization from the NLG domain. We will do this using TensorFlow, Keras, and some other latest dependencies. We will also see basic optimization techniques that you can apply to traditional ML algorithms and deep learning algorithms. Let's dive deep into the deep learning world in the next chapter!


