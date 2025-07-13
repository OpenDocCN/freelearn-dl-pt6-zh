
# Video Games by Reinforcement Learning

Contrary to supervised learning, where an algorithm has to associate an input with an output, in reinforcement learning you have another kind of maximization task. You are given an environment (that is, a situation) and you are required to find a solution that will act (something that may require to interact with or even change the environment itself) with the clear purpose of maximizing a resulting reward. Reinforcement learning algorithms, then, are not given any clear, explicit goal but to get the maximum result possible in the end. They are free to find the way to achieve the result by trial and error. This resembles the experience of a toddler who experiments freely in a new environment and analyzes the feedback in order to find out how to get the best from their experience. It also resembles the experience we have with a new video game: first, we look for the best winning strategy; we try a lot of different things and then we decide how to act in the game.

At the present time, no reinforcement learning algorithm has the general learning capabilities of a human being. A human being learns more quickly from several inputs, and a human can learn how to behave in very complex, varied, structured, unstructured and multiple environments. However, reinforcement learning algorithms have proved able to achieve super-human capabilities (yes, they can be better than a human) in very specific tasks. A reinforcement learning algorithm can achieve brilliant results if specialized on a specific game and if given enough time to learn (an example is AlphaGo [https://deepmind.com/research/alphago/](https://deepmind.com/research/alphago/)— the first computer program to defeat a world champion at Go, a complex game requiring long-term strategy and intuition).

In this chapter, we are going to provide you with the challenging project of getting a reinforcement learning algorithm to learn how to successfully manage the commands of the Atari game Lunar Lander, backed up by deep learning. Lunar Lander is the ideal game for this project because reinforcement learning algorithm can work successfully on it, the game has few commands and it can be successfully completed just by looking at few values describing the situation in the game (there is no need even to look at the screen in order to understand what to do, in fact, the first version of the game dates back to the 1960s and it was textual).

Neural networks and reinforcement learning are not new to each other;  in the early 1990s, at IBM, Gerry Tesauro programmed the famous TD-Gammon, combining feedforward neural networks with temporal-difference learning (a combination of Monte Carlo and dynamic programming) in order to train TD-Gammon to play world-class backgammon, which a board game for two players to be played using a couple of dices. If curious about the game,  you can read everything about the rules from the US Backgammon Federation: [http://usbgf.org/learn-backgammon/backgammon-rules-and-terms/rules-of-backgammon/](http://usbgf.org/learn-backgammon/backgammon-rules-and-terms/rules-of-backgammon/). At the time, the approach worked well with backgammon, due to the role of dices in the game that made it a non-deterministic game. Yet, it failed with every other game problem which was more deterministic. The last few years, thanks to the Google deep learning team of researchers, proved that neural networks can help solve problems other than backgammon, and that problem solving can be achieved on anyone's computer. Now, reinforcement learning is at the top of the list of next big things in deep learning and machine learning as you can read from Ian Goodfellow, an AI research scientist at Google Brain, who is putting it top of the list: [https://www.forbes.com/sites/quora/2017/07/21/whats-next-for-deep-learning/#6a8f8cd81002](https://www.forbes.com/sites/quora/2017/07/21/whats-next-for-deep-learning/#6a8f8cd81002).

# The game legacy

Lunar Lander is an arcade game developed by Atari that first appeared in video game arcades around 1979\. Developed in black and white vector graphics and distributed in specially devised cabinets, the game showed, as a lateral view, a lunar landing pod approaching the moon, where there were special areas for landing. The landing areas varied in width and accessibility because of the terrain around them, which gave the user different scores when the lander landed. The player was provided with information about altitude, speed, amount of fuel available, score, and time taken so far. Given the force of gravity attracting the landing pod to the ground, the player could rotate or thrust (there were also inertial forces to be considered) the landing pod at the expense of some fuel. The fuel was the key to the game.

The game ended when the landing pod touched the moon after running out of fuel. Until the fuel ran out, you kept on playing, even if you crashed. The commands available to the player were just four buttons, two for rotating left and right; one for thrusting from the base of the landing pod, pushing the module in the direction it is orientated; and the last button was for aborting the landing by rotating the landing pod upright and using a powerful (and fuel consuming) thrust in order to prevent the landing pod from crashing.

The interesting aspect of such a game is that there are clearly costs and rewards, but some are immediately apparent (like the quantity of fuel you are spending in your attempt) and others that they are all delayed until the time the landing pod touches the soil (you will know if the landing was a successful one only once it comes to a full stop). Maneuvering to land costs fuel, and that requires an economic approach to the game, trying not to waste too much. Landing provides a score. The more difficult and the safer the landing, the higher the score.

# The OpenAI version

As stated in the documentation available at its website ([https://gym.openai.com/](https://gym.openai.com/)), OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. The toolkit actually consists of a Python package that runs with both Python 2 and Python 3, and the website API, which is useful for uploading your own algorithm's performance results and comparing them with others (an aspect of the toolkit that we won't be exploring, actually).

The toolkit embodies the principles of reinforcement learning, where you have an environment and an agent: the agent can perform actions or inaction in the environment, and the environment will reply with a new state (representing the situation in the environment) and a reward, which is a score telling the agent if it is doing well or not. The Gym toolkit provides everything with the environment, therefore it is you that has to code the agent with an algorithm that helps the agent to face the environment. The environment is dealt by `env`, a class with methods for reinforcement learning which is instantiated when you issue the command to create it for a specific game: `gym.make('environment')`. Let's examine an example from the official documentation:

```py
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
      observation = env.reset()
      for t in range(100):
                  env.render()
                  print(observation)
                  # taking a random action
                  action = env.action_space.sample()
                  observation, reward, done, info = \   
                                             env.step(action)
                  If done:
                     print("Episode finished after %i \
                           timesteps" % (t+1))
                     break
```

In this example, the run environment is `CartPole-v0`. Mainly a control problem, in the `CartPole-v0` game, a pendulum is attached to a cart that moves along a friction less track. The purpose of the game is to keep the pendulum upright as long as possible by applying forward or backward forces to the cart, and you can look at the dynamics of the game by watching this sequence on YouTube, which is part of a real-life experiment held at the Dynamics and Control Lab, IIT Madras and based on Neuron-like adaptive elements that can solve difficult control problems: [https://www.youtube.com/watch?v=qMlcsc43-lg](https://www.youtube.com/watch?v=qMlcsc43-lg).

The Cartpole problem is described in *Neuron like adaptive elements that can solve difficult learning control problems* ([http://ieeexplore.ieee.org/document/6313077/](http://ieeexplore.ieee.org/document/6313077/)) by BARTO, Andrew G.; SUTTON, Richard S.; ANDERSON, Charles W. in IEEE transactions on systems, man, and Cybernetics.

Here is a brief explanation of the env methods, as applied in the example:

*   `reset()`: This resets the environment's state to the initial default conditions. It actually returns the start observations.
*   `step(action)`: This moves the environment by a single time step. It returns a four-valued vector made of variables: `observations`, `reward`, `done`, and `info`. Observations are a representation of the state of the environment and it is represented in each game by a different vector of values. For instance, in a game involving physics such as `CartPole-v0`, the returned vector is composed of the cart's position, the cart's velocity, the pole's angle, and the pole's velocity. The reward is simply the score achieved by the previous action (you need to total the rewards in order to figure out the total score at each point). The variable `done` is a Boolean flag telling you whether you are at a terminal state in the game (game over). `info` will provide diagnostic information, something that you are expected not to use for your algorithm, but just for debugging. 
*   `render( mode='human', close=False)`: This renders one time frame of the environment. The default mode will do something human-friendly, such as popping up a window. Passing the `close` flag signals the rendering engine to close any such windows.

The resulting effect of the commands is as follows:

*   Set up the `CartPole-v0` environment
*   Run it for 1,000 steps
*   Randomly choose whether to apply a positive or negative force to the cart
*   Visualize the results

The interesting aspect of this approach is that you can change the game easily, just by providing a different string to the `gym.make` method (try for instance `MsPacman-v0` or `Breakout-v0` or choose any from the list you can obtain by `gym.print(envs.registry.all())`) and test your approach to solving different environments without changing anything in your code. OpenAI Gym makes it easy to test the generalization of your algorithm to different problems by using a common interface for all its environments. Moreover, it provides a framework for your reasoning, understanding and solving of agent-environment problems according to the schema. At time *t-1* a state and reward are pushed to an agent, and the agent reacts with an action, producing a new state and a new reward at time *t*:

![](img/15d89b8b-a5c6-482b-8d00-d4e2f6b8b81e.png)

Figure 1: How the environment and agent interact by means of state, action, and reward

In every distinct game in OpenAI Gym, both the action space (the commands the agent responds to) and the `observation_space` (the representation state) change. You can see how they have changed by using some `print` commands, just after you set up an environment:

```py
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
```

# Installing OpenAI on Linux (Ubuntu 14.04 or 16.04)

We suggest installing the environment on an Ubuntu system. OpenGym AI has been created for Linux systems and there is little support for Windows. Depending on the previous settings of your system, you may need to install some additional things first:

```py
apt-get install -y python3-dev python-dev python-numpy libcupti-dev libjpeg-turbo8-dev make golang tmux htop chromium-browser git cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```

We suggest of working with Anaconda, so install Anaconda 3, too. You can find everything about installing this Python distribution at [https://www.anaconda.com/download/](https://www.anaconda.com/download/).

After setting the system requirements, installing OpenGym AI with all its modules is quite straightforward:

```py
git clone https://github.com/openai/gym
cd gym
pip install -e .[all]
```

For this project, we are actually interested in working with the Box2D module, which is a 2D physics engine providing a rendering of real-world physics in a 2D environment, as commonly seen in pseudo-realistic video games. You can test that the Box2D module works by running these commands in Python:

```py
import gym
env = gym.make('LunarLander-v2')
env.reset()
env.render()
```

If the provided code runs with no problem, you can proceed with the project. In some situations, Box2D may become difficult to run and, for instance, there could be problems such as those reported in [https://github.com/cbfinn/gps/issues/34](https://github.com/cbfinn/gps/issues/34), though there are many other examples around. We have found that installing the Gym in a `conda` environment based on Python 3.4 makes things much easier:

```py
conda create --name gym python=3.4 anaconda gcc=4.8.5
source activate gym
conda install pip six libgcc swig
conda install -c conda-forge opencv
pip install --upgrade tensorflow-gpu
git clone https://github.com/openai/gym
cd gym
pip install -e .
conda install -c https://conda.anaconda.org/kne pybox2d
```

This installation sequence should allow you to create a `conda` environment that's appropriate for the project we are going to present in this chapter.

# Lunar Lander in OpenAI Gym

LunarLander-v2 is a scenario developed by Oleg Klimov, an engineer at OpenAI, inspired by the original Atari Lunar Lander ([https://github.com/olegklimov](https://github.com/olegklimov)). In the implementation, you have to take your landing pod to a lunar pad that is always located at coordinates *x=0* and *y=0*. In addition, your actual *x* and *y* position is known since their values are stored in the first two elements of the state vector, the vector that contains all the information for the reinforcement learning algorithm to decide the best action to take at a certain moment.

This renders the task accessible because you won't have to deal with fuzzy or uncertain localization of your position with respect to the objective (a common problem in robotics).

![](img/a99d1857-625c-476b-8829-5605e0eded24.png)

Figure 2: LunarLander-v2 in action

At each moment, the landing pod has four possible actions to choose from:

*   Do nothing
*   Rotate left
*   Rotate right
*   Thrust

There is then a complex system of reward to make things interesting:

*   Reward for moving from the top of the screen to the landing pad and reaching zero speed ranges from 100 to 140 points (landing outside the landing pad is possible)

*   If the landing pod moves away from landing pad without coming to a stop, it loses some of the previous rewards
*   Each episode (the term used to point out a game session) completes when the landing pod crashes or it comes to rest, respectively providing additional -100 or +100 points
*   Each leg in contact with the ground is +10
*   Firing the main engine is -0.3 points per frame (but fuel is infinite)
*   Solving the episode grants 200 points

The game works perfectly with discrete commands (they are practically binary: full thrust or no thrust) because, as the author of the simulation says, according to Pontryagin's maximum principle it's optimal to fire the engine at full throttle or completely turn it off.

The game is also solvable using some simple heuristics based on the distance to the target and using a **proportional integral derivative** (**PID**) controller to manage the descending speed and angle. A PID is an engineering solution to control systems where you have feedback. At the following URL, you can get a more detailed explanation of how they work: [https://www.csimn.com/CSI_pages/PIDforDummies.html](https://www.csimn.com/CSI_pages/PIDforDummies.html).

# Exploring reinforcement learning through deep learning

In this project, we are not interested in developing a heuristic (a still valid approach to solving many problems in artificial intelligence) or constructing a working PID. We intend instead to use deep learning to provide an agent with the necessary intelligence to operate a Lunar Lander video game session successfully.

Reinforcement learning theory offers a few frameworks to solve such problems:

*   **Value-based learning**: This works by figuring out the reward or outcome from being in a certain state. By comparing the reward of different possible states, the action leading to the best state is chosen. Q-learning is an example of this approach.
*   **Policy-based learning**: Different control policies are evaluated based on the reward from the environment. It is decided upon the policy achieving the best results.
*   **Model-based learning**: Here, the idea is to replicate a model of the environment inside the agent, thus allowing the agent to simulate different actions and their consequent reward.

In our project, we will use a value-based learning framework; specifically, we will use the now classical approach in reinforcement learning based on Q-learning, which has been successfully controlled games where an agent has to decide on a series of moves that will lead to a delayed reward later in the game. Devised by C.J.C.H. Watkins in 1989 in his Ph.D. thesis, the method, also called **Q-learning**, is based on the idea that an agent operates in an environment, taking into account the present state, in order to define a sequence of actions that will lead to an ultimate reward:

![](img/99587134-1e20-4285-b204-3cff85be23f7.png)

In the above formula, it is described how a state *s*, after an action *a*, leads to a reward, *r*, and a new state *s'*. Starting from the initial state of a game, the formula applies a series of actions that, one after the other, transforms each subsequent state until the end of the game. You can then imagine a game as a series of chained states by a sequence of actions. You can then also interpret the above formula how an initial state *s* is transformed into a final state *s'* and a final reward *r* by a sequence of actions *a*.   

In reinforcement terms, a **policy** is how to best choose our sequence of actions, *a*.  A policy can be approximated by a function, which is called *Q*, so that given the present state, *s*, and a possible action, *a*, as inputs, it will provide an estimate of the maximum reward, *r*, that will derive from that action:

![](img/d7fd4657-9ec3-4acf-94ae-c06421c76173.png)

This approach is clearly greedy, meaning that we just choose the best action at a precise state because we expect that always choosing the best action at each step will lead us to the best outcome. Thus, in the greedy approach, we do not consider the possible chain of actions leading to the reward, but just the next action, *a*. However, it can be easily proved that we can confidently adopt a greedy approach and obtain the maximum reward using such a policy if such conditions are met:

*   we find the perfect policy oracle, *Q**
*   we operate in an environment where information is perfect (meaning we can know everything about the environment)
*   the environment adheres to the **Markov principle (see the tip box)**

the Markov principle states that the future (states, rewards) only depends on the present and not the past, therefore we can simply derive the best to be done by looking at the present state and ignoring what has previously happened.

In fact, if we build the *Q* function as a recursive function, we just need to explore (using a breadth-first search approach) the ramifications to the present state of our action to be tested, and the recursive function will return the maximum reward possible.

Such an approach works perfectly in a computer simulation, but it makes little sense in the real world:

*   Environments are mostly probabilistic. Even if you perform an action, you don't have the certainty of the exact reward.
*   Environments are tied to the past, the present alone cannot describe what could be the future because the past can have hidden or long-term consequences.
*   Environments are not exactly predictable, so you cannot know in advance the rewards from an action, but you can know them afterward (this is called an **a posteriori** condition).
*   Environments are very complex. You cannot figure out in a reasonable time all the possible consequences of an action, hence you cannot figure out with certainty the maximum reward deriving from an action.

The solution is then to adopt an approximate *Q* function, one that can take into account probabilistic outcomes and that doesn't need to explore all the future states by prediction. Clearly, it should be a real approximation function, because building a search table of values is unpractical in complex environments (some state spaces could take continuous values, making the possible combinations infinite). Moreover, the function can be learned offline, which implies leveraging the experience of the agent (the ability to memorize becomes then quite important).

There have been previous attempts to approximate a *Q* function by a neural network, but the only successful application has been `TD_Gammon`, a backgammon program that learned to play by reinforcement learning powered by a multi-layer perceptron only. `TD_Gammon` achieved a superhuman level of play, but at the time its success couldn't be replicated in different games, such as chess or go.

That led to the belief that neural networks were not really suitable for figuring out a *Q* function unless the game was somehow stochastic (you have to throw a dice in backgammon). In 2013, a paper on deep reinforcement learning, *Playing Atari with deep reinforcement learning(*[https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)*)* Volodymyr Minh, et al, applied to old Atari games demonstrated the contrary.

Such paper demonstrates how a *Q* function could be learned using neural networks to play a range of Atari arcade games (such as Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, and Space Invaders) just by processing video inputs (by sampling frames from a 210 x 160 RGB video at 60 Hz) and outputting joystick and fire button commands. The paper names the method a **Deep Q-Network** (**DQN**), and it also introduces the concepts of experience replay and exploration versus exploitation, which we are going to discuss in the next section. These concepts help to overcome some critical problems when trying to apply deep learning to reinforcement learning:

*   Lack of plenty of examples to learn from—something necessary in reinforcement learning and even more indispensable when using deep learning for it
*   Extended delay between an action and the effective reward, which requires dealing with sequences of further actions of variable length before getting a reward
*   Series of highly correlated sequences of actions (because an action often influences the following ones), which may cause any stochastic gradient descent algorithm to overfit to the most recent examples or simply converge non-optimally (stochastic gradient descent expects random examples, not correlated ones)

The paper, *Human-level control through deep reinforcement learning* ([http://www.davidqiu.com:8888/research/nature14236.pdf](http://www.davidqiu.com:8888/research/nature14236.pdf)), by Mnih and other researchers just confirms DQN efficacy where more games are explored by using it and performances of DQN are compared to human players and classical algorithms in reinforcement learning.

In many games, DQN proved better than human skills, though the long-term strategy is still a problem for the algorithm. In certain games, such as *Breakout*, the agent discovers cunning strategies such as digging a tunnel through the wall in order to send the ball through and destroy the wall in an effortless manner. In other games, such as *Montezuma's Revenge*, the agent remains clueless.

In the paper, the authors discuss at length how the agent understands the nuts and bolts of winning a Breakout game and they provide a chart of the response of the DQN function demonstrating how high reward scores are assigned to behaviors that first dig a hole in the wall and then let the ball pass through it.

# Tricks and tips for deep Q-learning

Q-learning obtained by neural networks was deemed unstable until some tricks made it possible and feasible. There are two power-horses in deep Q-learning, though other variants of the algorithm have been developed recently in order to solve problems with performance and convergence in the original solution. Such new variants are not discussed in our project: double Q-learning, delayed Q-learning, greedy GQ, and speedy Q-learning. The two main DQN power-horses that we are going to explore are **experience replay** and the decreasing trade-off between **exploration and exploitation**.

With experience replay, we simply store away the observed states of the game in a queue of a prefixed size since we discard older sequences when the queue is full. Contained in the stored data, we expect to have a number of tuples consisting of the present state, the applied action, the consequently obtained state, and the reward gained. If we consider a simpler tuple made of just the present state and the action, we have the observation of the agent operating in the environment, which we can consider the root cause of the consequent state and of the reward. We can consider now the tuple (present state and action) as our predictor ( *x* vector) with respect to the reward. Consequently, we can use the reward directly connected to the action and the reward that will be achieved at the end of the game.

Given such stored data (which we can figure out as the memory of our agent), we sample a few of them in order to create a batch and use the obtained batch to train our neural network. However, before passing the data to the network, we need to define our target variable, our *y* vector. Since the sampled states mostly won't be the final ones, we will probably have a zero reward or simply a partial reward to match against the known inputs (the present state and the chosen action). A partial reward is not very useful because it just tells part of the story we need to know. Our objective is, in fact, to know the total reward we will get at the end of the game, after having taken the action from the present state we are evaluating (our *x* value).

In this case, since we don't have such information, we simply try to approximate the value by using our existing *Q* function in order to estimate the residual reward that will be the maximum consequence of the (state, action) tuple we are considering. After obtaining it, we discount its value using the Bellman equation.

You can read an explanation of this now classic approach in reinforcement learning in this excellent tutorial by Dr. Sal Candido, a software engineer at Google: [http://robotics.ai.uiuc.edu/~scandido/?Developing_Reinforcement_Learning_from_the_Bellman_Equation](http://robotics.ai.uiuc.edu/~scandido/?Developing_Reinforcement_Learning_from_the_Bellman_Equation)), where the present reward is added to the discounted future reward.

Using a small value (approaching zero) for discounting makes the *Q* function more geared toward short-term rewards, whereas using a high discount value (approaching one) renders the *Q* function more oriented to future gains.

The second very effective trick is using a coefficient for trading between exploration and exploitation. In exploration, the agent is expected to try different actions in order to find the best course of action given a certain state. In exploitation, the agent leverages what it learned in the previous explorations and simply decides for what it believes the best action to be taken in that situation.

Finding a good balance between exploration and exploitation is strictly connected to the usage of the experience replay we discussed earlier. At the start of the DQN algorithm optimization, we just have to rely on a random set of network parameters. This is just like sampling random actions, as we did in our simple introductory example to this chapter. The agent in such a situation will explore different states and actions, and help to shape the initial *Q* function. For complex games such as *Lunar Lander* using random choices won't take the agent far, and it could even turn unproductive in the long run because it will prevent the agent from learning the expected reward for tuples of state and action that can only be accessed if the agent has done the correct things before. In fact, in such a situation the DQN algorithm will have a hard time figuring out how to appropriately assign the right reward to an action because it will never have seen a completed game. Since the game is complex, it is unlikely that it could be solved by random sequences of actions.

The correct approach, then, is to balance learning by chance and using what has been learned to take the agent further in the game to where problems are yet to be solved. This resembles finding a solution by a series of successive approximations, by taking the agent each time a bit nearer to the correct sequence of actions for a safe and successful landing. Consequently, the agent should first learn by chance, find the best things to be done in a certain set of situations, then apply what has been learned and get access to new situations that, by random choice, will be also solved, learned, and applied successively.

This is done using a decreasing value as the threshold for the agent to decide whether, at a certain point in the game, to take a random choice and see what happens or leverage what it has learned so far and use it to make the best possible action at that point, given its actual capabilities. Picking a random number from a uniform distribution [*0*,*1*], the agent compares it with an epsilon value, and if the random number is larger than the epsilon it will use its approximate neural *Q* function. Otherwise, it will pick a random action from the options available. After that, it will decrease the epsilon number. Initially, epsilon is set at the maximum value, *1.0*, but depending on a decaying factor, it will decrease with time more or less rapidly, arriving at a minimum value that should never be zero (no chance of taking a random move) in order for there to always be the possibility of learning something new and unexpected (a minimal openness factor) by serendipity.

# Understanding the limitations of deep Q-learning

Even with deep Q-learning, there are some limitations, no matter whether you approximate your *Q* function by deriving it from visual images or other observations about the environment:

*   The approximation takes a long time to converge, and sometimes it doesn't achieve it smoothly: you may even witness the learning indicators of the neural network worsening instead of getting better for many epochs.
*   Being based on a greedy approach, the approach offered by Q-learning is not dissimilar from a heuristic: it points out the best direction but it cannot provide detailed planning. When dealing with long-term goals or goals that have to be articulated into sub-goals, Q-learning performs badly.
*   Another consequence of how Q-learning works is that it really doesn't understand the game dynamics from a general point of view but from a specific one (it replicates what it experienced as effective during training). As a consequence, any novelty introduced into the game (and never actually experienced during training) can break down the algorithm and render it completely ineffective. The same goes when introducing a new game to the algorithm; it simply won't perform.

# Starting the project

After this long detour into reinforcement learning and the DQN approach, we are finally ready to start coding, having all the basic understanding of how to operate an OpenAI Gym environment and how to set a DQN approximation of a *Q* function. We simply start importing all the necessary packages:

```py
import gym
from gym import wrappers
import numpy as np
import random, tempfile, os
from collections import deque
import tensorflow as tf
```

The `tempfile` module generates temporary files and directories that can be used as a temporary storage area for data files. The `deque `command, from the `collections` module, creates a double-ended queue, practically a list where you can append items at the start or at the end. Interestingly, it can be set to a predefined size. When full, older items are discarded in order to make the place for new entries.

We will structure this project using a series of classes representing the agent, the agent's brain (our DQN), the agent's memory, and the environment, which is provided by OpenAI Gym but it needs to be correctly connected to the agent. It is necessary to code a class for this.

# Defining the AI brain

The first step in the project is to create a `Brain` class containing all the neural network code in order to compute a Q-value approximation. The class will contain the necessary initialization, the code for creating a suitable TensorFlow graph for the purpose, a simple neural network (not a complex deep learning architecture but a simple, working network for our project—you can replace it with more complex architectures), and finally, methods for fit and predict operations.

We start from initialization. As inputs, first, we really need to know the size of the state inputs (`nS`) corresponding to the information we receive from the game, and the size of the action output (`nA`) corresponding to the buttons we can press to perform actions in the game. Optionally, but strongly recommended, we also have to set the scope. In order to define the scope a string will help us to keep separate networks created for different purposes, and in our project, we have two, one for processing the next reward and one for guessing the final reward.

Then, we have to define the learning rate for the optimizer, which is an Adam.

The Adam optimizer is described in the following paper: [https://arxiv.org/abs/1412.6980.](https://arxiv.org/abs/1412.6980)It is a very efficient gradient-based optimization method that requires very little to be tuned in order to work properly. The Adam optimization is a stochastic gradient descent algorithm similar to RMSprop with Momentum. This post, [https://theberkeleyview.wordpress.com/2015/11/19/berkeleyview-for-adam-a-method-for-stochastic-optimization/](https://theberkeleyview.wordpress.com/2015/11/19/berkeleyview-for-adam-a-method-for-stochastic-optimization/), from the UC Berkeley Computer Vision Review Letters, provides more information. From our experience, it is one of the most effective solutions when training a deep learning algorithm in batches, and it requires some tuning for the learning rate.

Finally, we also provide:

*   A neural architecture (if we prefer to change the basic one provided with the class)
*   Input the `global_step`, a global variable that will keep track of the number of training batches of examples that have been feed to the DQN network  up to that moment
*   The directory in which to store the logs for TensorBoard, the standard visualization tool for TensorFlow

```py
class Brain:
    """
    A Q-Value approximation obtained using a neural network.
    This network is used for both the Q-Network and the Target Network.
    """
    def __init__(self, nS, nA, scope="estimator",
                 learning_rate=0.0001,
                 neural_architecture=None,
                 global_step=None, summaries_dir=None):
        self.nS = nS
        self.nA = nA
        self.global_step = global_step
        self.scope = scope
        self.learning_rate = learning_rate
        if not neural_architecture:
            neural_architecture = self.two_layers_network
        # Writes Tensorboard summaries to disk
        with tf.variable_scope(scope):
            # Build the graph
            self.create_network(network=neural_architecture,              
                                learning_rate=self.learning_rate)
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, 
                                          "summaries_%s" % scope)
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = \
                               tf.summary.FileWriter(summary_dir)
            else:
                self.summary_writer = None
```

The command  `tf.summary.FileWriter` initializes an event file in a target directory (`summary_dir`) where we store the key measures of the learning process. The handle is kept in `self.summary_writer`, which we will be using later for storing the measures we are interested in representing during and after the training for monitoring and debugging what has been learned.

The next method to be defined is the default neural network that we will be using for this project. As input, it takes the input layer and the respective size of the hidden layers that we will be using. The input layer  is defined by the state that we are using, which could be a vector of measurements, as in our case, or an image, as in the original DQN paper) 

Such layers are simply defined using the higher level ops offered by the `Layers` module of TensorFlow ([https://www.tensorflow.org/api_guides/python/contrib.layers](https://www.tensorflow.org/api_guides/python/contrib.layers)). Our choice goes for the vanilla `fully_connected`, using the `ReLU` (rectifier) `activation` function for the two hidden layers and the linear activation of the output layer. 

The predefined size of 32 is perfectly fine for our purposes, but you may increment it if you like. Also, there is no dropout in this network. Clearly, the problem here is not overfitting, but the quality of what is being learned, which could only be improved by providing useful sequences of unrelated states and a good estimate of the final reward to be associated. It is in the useful sequences of states, especially under the light of the trade-off between exploration and exploitation, that the key to not having the network overfit resides. In a reinforcement learning problem, you have overfitted if you fall into one of these two situations:

*   Sub-optimality: the algorithm suggests sub-optimal solutions, that it is, our lander learned a rough way to land and it sticks to it because at least it lands
*   Helplessness: the algorithm has fallen into a learned helplessness; that is, it has not found a way to land correctly, so it just accepts that it is going to crash in the least bad way possible

These two situations can prove really difficult to overcome for a reinforcement learning algorithm such as DQN unless the algorithm can have the chance to explore alternative solutions during the game. Taking random moves from time to time is not simply a way to mess up things, as you may think at first sight, but a strategy to avoid pitfalls.

With larger networks than this one, on the other hand, you may instead have a problem with a dying neuron requiring you to use a different activation, `tf.nn.leaky_relu` ([https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu](https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu)), in order to obtain a working network.

A dead `ReLU` ends up always outputting the same value, usually a zero value, and it becomes resistant to backpropagation updates.

The activation `leaky_relu` has been available since TensorFlow 1.4\. If you are using any previous version of TensorFlow, you can create an `ad hoc` function to be used in your custom network:

`def leaky_relu(x, alpha=0.2):       return tf.nn.relu(x) - alpha * tf.nn.relu(-x)`

We now proceed to code our `Brain` class, adding some more functions to it:

```py
def two_layers_network(self, x, layer_1_nodes=32, 
                                layer_2_nodes=32):

    layer_1 = tf.contrib.layers.fully_connected(x, layer_1_nodes, 
                                        activation_fn=tf.nn.relu)
    layer_2 = tf.contrib.layers.fully_connected(layer_1, 
                                          layer_2_nodes, 
                               activation_fn=tf.nn.relu)
    return tf.contrib.layers.fully_connected(layer_2, self.nA, 
                                           activation_fn=None)
```

The method`create_network` combines input, neural network, loss, and optimization. The loss is simply created by taking the difference between the original reward and the estimated result, squaring it, and taking the average through all the examples present in the batch being learned. The loss is minimized using an Adam optimizer.

Also, a few summaries are recorded for TensorBoard: 

*   The average loss of the batch, in order to keep track of the fit during training
*   The maximum predicted reward in the batch, in order to keep track of extreme positive predictions, pointing out the best-winning moves
*   The average predicted reward in the batch, in order to keep track of the general tendency of predicting good moves

Here is the code for `create_network`, the TensorFlow engine of our project:

```py
    def create_network(self, network, learning_rate=0.0001):

        # Placeholders for states input
        self.X = tf.placeholder(shape=[None, self.nS], 
                                 dtype=tf.float32, name="X")
        # The r target value
        self.y = tf.placeholder(shape=[None, self.nA], 
                                 dtype=tf.float32, name="y")
        # Applying the choosen network
        self.predictions = network(self.X)
        # Calculating the loss
        sq_diff = tf.squared_difference(self.y, self.predictions)
        self.loss = tf.reduce_mean(sq_diff)
        # Optimizing parameters using the Adam optimizer
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, 
                        global_step=tf.train.get_global_step(),                                      
                        learning_rate=learning_rate, 
                        optimizer='Adam')
        # Recording summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.scalar("max_q_value", 
                             tf.reduce_max(self.predictions)),
            tf.summary.scalar("mean_q_value", 
                             tf.reduce_mean(self.predictions))])
```

The class is completed by a `predict` and a `fit` method. The `fit` method takes as input the state matrix, `s`, as the input batch and the vector of reward `r` as the outcome. It also takes into account how many epochs you want to train (in the original papers it is suggested using just a single epoch per batch in order to avoid overfitting too much to each batch of observations). Then, in the present session, the input is fit with respect to the outcome and summaries (previously defined as we created the network). 

```py
    def predict(self, sess, s):
        """
        Predicting q values for actions
        """
        return sess.run(self.predictions, {self.X: s})

    def fit(self, sess, s, r, epochs=1):
        """
        Updating the Q* function estimator
        """
        feed_dict = {self.X: s, self.y: r}
        for epoch in range(epochs):
            res = sess.run([self.summaries, self.train_op, 
                            self.loss, 
                            self.predictions,
                            tf.train.get_global_step()], 
                            feed_dict)
            summaries, train_op, loss, predictions, 
                                       self.global_step = res

        if self.summary_writer:
            self.summary_writer.add_summary(summaries,
self.global_step)
```

As a result,  `global step` is returned, which is a counter that helps to keep track of the number of examples used in training up so far, and then recorded for later use.

# Creating memory for experience replay

After defining the brain (the TensorFlow neural network), our next step is to define the memory, that is the storage for data that will power the learning process of the DQN network. At each training episode each step, made of a state and an action, is recorded together with the consequent state and the final reward of the episode (something that will be known only when the episode completes).

Adding a flag telling if the observation is a terminal one or not completes the set of recorded information. The idea is to connect certain moves not just to the immediate reward (which could be null or modest) but the ending reward, thus associating every move in that session to it.

The class memory is simply a queue of a certain size, which is then filled with information on the previous game experiences, and it is easy to sample and extract from it. Given its fixed size, it is important that older examples are pushed out of the queue, thus allowing the available examples to always be among the last ones.

The class comprises an initialization, where the data structure takes origin and its size is fixed, the `len` method (so we know whether the memory is full or not, which is useful, for instance, in order to wait for any training at least until we have plenty of them for better randomization and variety for learning), `add_memory` for recording in the queue, and `recall_memory` for recovering all the data from it in a list format:

```py
class Memory:
    """
    A memory class based on deque, a list-like container with 
    fast appends and pops on either end (from the collections 
    package)
    """
    def __init__(self, memory_size=5000):
        self.memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.memory)

    def add_memory(self, s, a, r, s_, status):
        """
        Memorizing the tuple (s a r s_) plus the Boolean flag status,
        reminding if we are at a terminal move or not
        """
        self.memory.append((s, a, r, s_, status))

    def recall_memories(self):
        """
        Returning all the memorized data at once
        """
        return list(self.memory)
```

# Creating the agent

The next class is the agent, which has the role of initializing and maintaining the brain (providing the *Q-value* function approximation) and the memory. It is the agent, moreover, that acts in the environment. Its initialization sets a series of parameters that are mostly fixed given our experience in optimizing the learning for the Lunar Lander game. They can be explicitly changed, though, when the agent is first initialized:

*   `epsilon = 1.0` is the initial value in the exploration-exploitation parameter. The `1.0` value forces the agent to completely rely on exploration, that is, random moving.
*   `epsilon_min = 0.01` sets the minimum value of the exploration-exploitation parameter: a value of `0.01` means that there is a 1% chance that the landing pod will move randomly and not based on *Q* function feedback. This always provides a minimum chance to find another optimal way of completing the game, without compromising it.
*   `epsilon_decay = 0.9994` is the decay that regulates the speed the `epsilon` diminishes toward the minimum. In this setting, it is tuned to reach a minimum value after about 5,000 episodes, which on average should provide the algorithm at least 2 million examples to learn from.
*   `gamma = 0.99` is the reward discount factor with which the Q-value estimation weights the future reward with respect to the present reward, thus allowing the algorithm to be short- or long-sighted, according to what is best in the kind of game being played (in Lunar Lander it is better to be long-sighted because the actual reward will be experienced only when the landing pod lands on the Moon).
*   `learning_rate = 0.0001` is the learning rate for the Adam optimizer to learn the batch of examples.
*   `epochs = 1` is the training epochs used by the neural network in order to fit the batch set of examples.
*   `batch_size = 32` is the size of the batch examples.
*   `memory = Memory(memory_size=250000)` is the size of the memory queue.

Using the preset parameters, you are assured that the present project will work. For different OpenAI environments, you may need to find different optimal parameters.

The initialization will also provide the commands required to define where the TensorBoard logs will be placed (by default, the `experiment` directory), the model for learning how to estimate the immediate next reward, and another model to store the weights for the final reward. In addition, a saver (`tf.train.Saver`) will be initialized, allowing the serialization of the entire session to disk in order to restore it later and use it for playing the real game, not just learning how to play it.

The two mentioned models are initialized in the same session, using different scope names (one will be `q`, the next reward model monitored by the TensorBoard; the other one will be `target_q`). Using two different scope names will allow easy handling of the neuron's coefficients, making it possible to swap them with another method present in the class:

```py
class Agent:
    def __init__(self, nS, nA, experiment_dir):
        # Initializing
        self.nS = nS
        self.nA = nA
        self.epsilon = 1.0  # exploration-exploitation ratio
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9994
        self.gamma = 0.99  # reward decay
        self.learning_rate = 0.0001
        self.epochs = 1  # training epochs
        self.batch_size = 32
        self.memory = Memory(memory_size=250000)

        # Creating estimators
        self.experiment_dir =os.path.abspath\    
                      ("./experiments/{}".format(experiment_dir))
        self.global_step = tf.Variable(0, name='global_step', 
                                                trainable=False)
        self.model = Brain(nS=self.nS, nA=self.nA, scope="q",
                           learning_rate=self.learning_rate,
                           global_step=self.global_step,
                           summaries_dir=self.experiment_dir)
        self.target_model = Brain(nS=self.nS, nA=self.nA, 
                                             scope="target_q",
                             learning_rate=self.learning_rate,
                                 global_step=self.global_step)

        # Adding an op to initialize the variables.
        init_op = tf.global_variables_initializer()
        # Adding ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        # Setting up the session
        self.sess = tf.Session()
        self.sess.run(init_op)
```

The `epsilon` dealing with the share of time devoted exploring new solutions compared to exploiting the knowledge of the network is constantly updated with the `epsilon_update` method, which simply modifies the actual `epsilon` by multiplying it by `epsilon_decay` unless it has already reached its allowed minimum value:

```py
def epsilon_update(self, t):
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
```

The `save_weights` and `load_weights` methods simply allow the session to be saved:

```py
    def save_weights(self, filename):
        """
        Saving the weights of a model
        """
        save_path = self.saver.save(self.sess, 
                                    "%s.ckpt" % filename)
        print("Model saved in file: %s" % save_path)
def load_weights(self, filename):
    """
    Restoring the weights of a model
    """
    self.saver.restore(self.sess, "%s.ckpt" % filename)
    print("Model restored from file")
```

The `set_weights` and  `target_model_update` methods work together to update the target Q network with the weights of the Q network (`set_weights` is a general-purpose, reusable function you can use in your solutions, too). Since we named the two scopes differently, it is easy to enumerate the variables of each network from the list of trainable variables. Once enumerated, the variables are joined in an assignment to be executed by the running session:

```py
    def set_weights(self, model_1, model_2):
        """
        Replicates the model parameters of one 
        estimator to another.
        model_1: Estimator to copy the parameters from
        model_2: Estimator to copy the parameters to
        """
        # Enumerating and sorting the parameters 
        # of the two models
        model_1_params = [t for t in tf.trainable_variables() \
                          if t.name.startswith(model_1.scope)]
        model_2_params = [t for t in tf.trainable_variables() \
                         if t.name.startswith(model_2.scope)]
        model_1_params = sorted(model_1_params, 
                                key=lambda x: x.name)
        model_2_params = sorted(model_2_params, 
                                key=lambda x: x.name)
        # Enumerating the operations to be done
        operations = [coef_2.assign(coef_1) for coef_1, coef_2 \
                      in zip(model_1_params, model_2_params)]
        # Executing the operations to be done
        self.sess.run(operations)
    def target_model_update(self):
        """
        Setting the model weights to the target model's ones
        """
        self.set_weights(self.model, self.target_model)
```

The `act` method is the core of the policy implementation because it will decide, based on `epsilon`, whether to take a random move or go for the best possible one. If it is going for the best possible move, it will ask the trained Q network to provide a reward estimate for each of the possible next moves (represented in a binary way by pushing one of four buttons in the Lunar Lander game) and it will return the move characterized by the maximum predicted reward (a greedy approach to the solution):

```py
    def act(self, s):
        """
        Having the agent act based on learned Q* function
        or by random choice (based on epsilon)
        """
        # Based on epsilon predicting or randomly 
        # choosing the next action
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.nA)
        else:
            # Estimating q for all possible actions
            q = self.model.predict(self.sess, s)[0]
            # Returning the best action
            best_action = np.argmax(q)
            return best_action
```

The `replay` method completes the class. It is a crucial method because it makes learning for the DQN algorithm possible. We are going, therefore, to discuss how it works thoroughly. The first thing that the `replay` method does is to sample a batch (we defined the batch size at initialization) from the memories of previous game episodes (such memories are just the variables containing values about status, action, reward, next status, and a flag variable noticing if the observation is a final status or not). The random sampling allows the model to find the best coefficients in order to learn the *Q* function by a slow adjustment of the network's weights, batch after batch.

Then the method finds out whether the sampling recalled statuses are final or not. Non-final rewards need to be updated in order to represent the reward that you get at the end of the game. This is done by using the target network, which represents a snapshot of the *Q* function network as fixed at the end of the previous learning. The target network is fed with the following status, and the resulting reward is summed, after being discounted by a gamma factor, with the present reward.

Using the present *Q* function may lead to instabilities in the learning process and it may not result in a satisfying *Q* function network. 

```py
    def replay(self):
        # Picking up a random batch from memory
        batch = np.array(random.sample(\
                self.memory.recall_memories(), self.batch_size))
        # Retrieving the sequence of present states
        s = np.vstack(batch[:, 0])
        # Recalling the sequence of actions
        a = np.array(batch[:, 1], dtype=int)
        # Recalling the rewards
        r = np.copy(batch[:, 2])
        # Recalling the sequence of resulting states
        s_p = np.vstack(batch[:, 3])
        # Checking if the reward is relative to 
        # a not terminal state
        status = np.where(batch[:, 4] == False)
        # We use the model to predict the rewards by 
        # our model and the target model
        next_reward = self.model.predict(self.sess, s_p)
        final_reward = self.target_model.predict(self.sess, s_p)

        if len(status[0]) > 0:
            # Non-terminal update rule using the target model
            # If a reward is not from a terminal state, 
            # the reward is just a partial one (r0)
            # We should add the remaining and obtain a 
            # final reward using target predictions
            best_next_action = np.argmax(\
                             next_reward[status, :][0], axis=1)
            # adding the discounted final reward
            r[status] += np.multiply(self.gamma,
                     final_reward[status, best_next_action][0])

        # We replace the expected rewards for actions 
        # when dealing with observed actions and rewards
        expected_reward = self.model.predict(self.sess, s)
        expected_reward[range(self.batch_size), a] = r

        # We re-fit status against predicted/observed rewards
        self.model.fit(self.sess, s, expected_reward,
                       epochs=self.epochs)
```

When the rewards of non-terminal states have been updated, the batch data is fed into the neural network for training.

# Specifying the environment

The last class to be implemented is the `Environment` class. Actually, the environment is provided by the `gym` command, though you need a good wrapper around it in order to have it work with the previous `agent` class. That's exactly what this class does. At initialization, it starts the Lunar Lander game and sets key variables such as `nS`, `nA` (dimensions of state and action), `agent`, and the cumulative reward (useful for testing the solution by providing an average of the last 100 episodes):

```py
class Environment:
    def __init__(self, game="LunarLander-v2"):
        # Initializing
        np.set_printoptions(precision=2)
        self.env = gym.make(game)
        self.env = wrappers.Monitor(self.env, tempfile.mkdtemp(), 
                               force=True, video_callable=False)
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n
        self.agent = Agent(self.nS, self.nA, self.env.spec.id)

        # Cumulative reward
        self.reward_avg = deque(maxlen=100)
```

Then, we prepare the code for methods for `test`, `train`, and `incremental` (incremental training), which are defined as wrappers of the comprehensive `learning` method. 

Using incremental training is a bit tricky and it requires some attention if you do not want to spoil the results you have obtained with your training so far. The trouble is that when we restart the brain has pre-trained coefficients but memory is actually empty (we can call this as a cold restart). Being the memory of the agent empty, it cannot support good learning because of too few and limited examples. Consequently, the quality of the examples being fed is really not perfect for learning (the examples are mostly correlated with each other and very specific to the few newly experienced episodes). The risk of ruining the training can be mitigated using a very low `epsilon` (we suggest set at the minimum, `0.01` ): in this way, the network  will most of the time simply re-learn its own weights because it will suggest for each state the actions it already knows, and its performance shouldn't worsen but oscillate in a stable way until there are enough examples in memory and it will start improving again.

Here is the code for issuing the correct methods for training and testing:

```py
    def test(self):
        self.learn(epsilon=0.0, episodes=100, 
                        trainable=False, incremental=False)

    def train(self, epsilon=1.0, episodes=1000):
        self.learn(epsilon=epsilon, episodes=episodes, 
                        trainable=True, incremental=False)

    def incremental(self, epsilon=0.01, episodes=100):
        self.learn(epsilon=epsilon, episodes=episodes, 
                        trainable=True, incremental=True)
```

The final method is `learn`, arranging all the steps for the agent to interact with and learn from the environment. The method takes the `epsilon` value (thus overriding any previous `epsilon` value the agent had), the number of episodes to run in the environment, whether it is being trained or not (a Boolean flag), and whether the training is continuing from the training of a previous model (another Boolean flag).

In the first block of code, the method loads the previously trained weights of the network for Q value approximation if we want:

1.  to test the network and see how it works;
2.  to carry on some previous training using further examples.

Then the method delves into a nested iteration. The outside iteration is running through the required number of episodes (each episode a Lunar Lander game has taken to its conclusion). Whereas the inner iteration is instead running through a maximum of 1,000 steps making up an episode.

At each time step in the iteration, the neural network is interrogated on the next move. If it is under test, it will always simply provide the answer about the next best move. If it is under training, there is some chance, depending on the value of `epsilon`, that it won't suggest the best move but it will instead propose making a random move.

```py
    def learn(self, epsilon=None, episodes=1000, 
              trainable=True, incremental=False):
        """
        Representing the interaction between the enviroment 
        and the learning agent
        """
        # Restoring weights if required
        if not trainable or (trainable and incremental):
            try:
                print("Loading weights")
                self.agent.load_weights('./weights.h5')
            except:
                print("Exception")
                trainable = True
                incremental = False
                epsilon = 1.0

        # Setting epsilon
        self.agent.epsilon = epsilon
        # Iterating through episodes
        for episode in range(episodes):
            # Initializing a new episode
            episode_reward = 0
            s = self.env.reset()
            # s is put at default values
            s = np.reshape(s, [1, self.nS])

            # Iterating through time frames
            for time_frame in range(1000):
                if not trainable:
                    # If not learning, representing 
                    # the agent on video
                    self.env.render()
                # Deciding on the next action to take
                a = self.agent.act(s)
                # Performing the action and getting feedback
                s_p, r, status, info = self.env.step(a)
                s_p = np.reshape(s_p, [1, self.nS])

                # Adding the reward to the cumulative reward
                episode_reward += r

                # Adding the overall experience to memory
                if trainable:
                    self.agent.memory.add_memory(s, a, r, s_p,
                                                 status)

                # Setting the new state as the current one
                s = s_p

                # Performing experience replay if memory length 
                # is greater than the batch length
                if trainable:
                    if len(self.agent.memory) > \
                           self.agent.batch_size:
                        self.agent.replay()

                # When the episode is completed, 
                # exiting this loop
                if status:
                    if trainable:
                       self.agent.target_model_update()
                    break

            # Exploration vs exploitation
            self.agent.epsilon_update(episode)

            # Running an average of the past 100 episodes
            self.reward_avg.append(episode_reward)
            print("episode: %i score: %.2f avg_score: %.2f"
                  "actions %i epsilon %.2f" % (episode,
                                        episode_reward,
                           np.average(self.reward_avg),
                                            time_frame,
                                               epsilon)
        self.env.close()

        if trainable:
            # Saving the weights for the future
            self.agent.save_weights('./weights.h5')
```

After the move, all the information is gathered (initial state, chosen action, obtained reward, and consequent state) and saved into memory. At this time frame, if the memory is large enough to create a batch for the neural network approximating the *Q* function, then a training session is run. When all the time frames of the episode have been consumed, the weights of the DQN get stored into another network to be used as a stable reference as the DQN network is learning from a new episode.

# Running the reinforcement learning process

Finally, after all the digression on reinforcement learning and DQN and writing down the complete code for the project, you can run it using a script or a Jupyter Notebook, leveraging the `Environment` class that puts all the code functionalities together:

```py
lunar_lander = Environment(game="LunarLander-v2")
```

After instantiating it, you just have to run the `train`, starting from `epsilon=1.0` and setting the goal to `5000` episodes (which corresponds to about 2.2 million examples of chained variables of state, action and reward). The actual code we provided is set to successfully accomplish a fully trained DQN model, though it may take some time, given your GPU's availability and its computing capabilities:

```py
    lunar_lander.train(epsilon=1.0, episodes=5000)
```

In the end, the class will complete the required training, leaving a saved model on disk (which could be run or even reprised anytime). You can even inspect the TensorBoard using a simple command that can be run from a shell:

```py
tensorboard --logdir=./experiments --port 6006
```

The plots will appear on your browser, and they will be available for inspection at the local address `localhost:6006`:

![](img/3007e975-dc2b-4ce9-98b9-b9607a870578.png)

Figure 4: The loss trend along the training, the peaks represent break-thoughts in learning such as at 800k examples
when it started landing safely on the ground.  

The loss plot will reveal that, contrary to other projects, the optimization is still characterized by a decreasing loss, but with many spikes and problems along the way:

The plots represented here are the result of running the project once. Since there is a random component in the process, you may obtain slightly different plots when running the project on your own computer.

![](img/29069c1b-c954-4c9b-8c2c-baae261a3d4a.png)

Figure 5: The trend of maximum q values obtained in a batch session of learning

The same story is told by the maximum predicted *q* value and the average predicted *q* value.  The network improves at the end, though it can slightly retrace its steps and linger on plateaus for a long time:

![](img/4a361ba7-4c97-4b4e-b03f-c64b3a51345a.png)

Figure 6: The trend of average q values obtained in a batch session of learning

Only if you take the average of the last 100 final rewards do you see an incremental path, hinting at a persistent and steady improvement of the DQN network:

![](img/6ce2623c-3338-47c5-8059-45e78cc942fa.png)

Figure 7: The trend of actually obtained scores at the end of each learning episode, it more clearly depicts the growing capabilities of the DQN

Using the same information, from the output, not from the TensorBoard, you'll also figure out that the number of actions changes on average depending on the `epsilon` value. At the beginning, the number of actions required to finish an episode was under 200\. Suddenly, when `epsilon` is `0.5`, the average number of actions tends to grow steadily and reach a peak at about 750 (the landing pod has learned to counteract gravity by using its rockets).

In the end, the network discovers this is a sub-optimal strategy and when `epsilon` turns below `0.3`, the average number of actions for completing an episode drops as well. The DQN in this phase is discovering how to successfully land the pod in a more efficient way:

![](img/9948d955-1382-4c2c-a97d-472505d1ca43.png)

Figure 8: The relationship between the epsilon (the exploration/exploitation rate) and the efficiency of the DQN network,
expressed as a number of moves used to complete an episode

If for any reason, you believe that the network needs more examples and learning, you can reprise the learning using the incremental `method`, keeping in mind that `epsilon` should be very low in this case:

```py
lunar_lander.incremental(episodes=25, epsilon=0.01)
```

After the training, if you need to see the results and know, on average every 100 episodes, how much the DQN can score (the ideal target is a `score >=200`),  you can just run the following command:

```py
 lunar_lander.test()
```

# Acknowledgements

At the conclusion of this project, we would like to indeed thank Peter Skvarenina, whose project "Lunar Lander II" ([https://www.youtube.com/watch?v=yiAmrZuBaYU](https://www.youtube.com/watch?v=yiAmrZuBaYU)) has been the key inspiration for our own project, and for all his suggestions and hints during the making of our own version of the Deep Q-Network.

# Summary

In this project, we have explored what a reinforcement algorithm can manage to achieve in an OpenAI environment, and we have programmed a TensorFlow graph capable of learning how to estimate a final reward in an environment characterized by an agent, states, actions, and consequent rewards. This approach, called DQN, aims to approximate the result from a Bellman equation using a neural network approach. The result is a Lunar Lander game that the software can play successfully at the end of training by reading the game status and deciding on the right actions to be taken at any time. 
