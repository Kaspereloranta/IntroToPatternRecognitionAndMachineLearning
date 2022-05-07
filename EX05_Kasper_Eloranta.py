'''
Course: Data.ml.100 Introduction to pattern recognition and machine learning
Exercise 5

Student name: Kasper Eloranta
Student ID: H274212
E-mail: kasper.eloranta@tuni.fi
'''

# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy
import time

'''
 There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    
     Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations

'''

# Environment
env = gym.make("Taxi-v3")

# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode

num_of_test_episodes = 10
action_size = env.action_space.n
state_size = env.observation_space.n

# Q tables for rewards
Q_reward = -100000*numpy.ones((state_size, action_size)) # All same
#Q_reward = -100000*numpy.random.random((state_size, action_size)) # Random

# More training parameters, (epsilon, exploration/exploitation trade-off)
# Since it is useful if we choose more random actions at the beginning of the
# training to explore the environment. At the end of the training
# system knows the environment better (Q-table is more accurate), so
# we don't have to choose as many random actions, we can just pick the action
# for the certain state from Q-table which has the highest reward.
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

# Training w/ random sampling of actions
# YOU WRITE YOUR CODE HERE
for ep in range(num_of_episodes):

    state = env.reset()
    step = 0
    done = False

    for step in range(num_of_steps):

        random_number = random.uniform(0,1)

        if random_number > epsilon: # Likelihood of this being true increases as the training proceeds.
            action = numpy.argmax(Q_reward[state,:]) # Accuracy of Q-table increases as well when the training proceeds.
        else:
            action = env.action_space.sample()  # Random action

        new_state, reward, done, info = env.step(action)
        # To scale the reward to respond reward values in Q-table.
        reward = reward*100000

        # Updating Q-table
        Q_reward[state,action] = Q_reward[state,action] + alpha*(reward+gamma*numpy.max(Q_reward[new_state, :])
        -Q_reward[state, action])

        state = new_state

        if done: # if true, the passenger has reached his/her destination.
            break

    # To reduce the epsilon as the training proceeds in order to decrease the amount of random actions
    # done in the episodes.
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*numpy.exp(-decay_rate*ep)

num_of_actions = []
total_reward = []

# 10 test runs
for i in range(0,num_of_test_episodes):
    state = env.reset()
    tot_reward = 0
    for t in range(50):
        action = numpy.argmax(Q_reward[state,:])
        state, reward, done, info = env.step(action)
        tot_reward += reward
        env.render()
        time.sleep(1)
        if done:
            print("Total reward %d" %tot_reward)
            num_of_actions.append(t)
            total_reward.append(tot_reward)
            break

#Results
print("Average total reward: ", sum(total_reward) / len(total_reward),
    "Average number of actions: ", sum(num_of_actions) / len(num_of_actions))