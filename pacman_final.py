
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import gc
from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, InputLayer
from keras.callbacks import CSVLogger, TensorBoard
from keras.optimizers import Adam
import keras.backend as K
import gym
import json
from mini_pacman import PacmanGame
from collections import deque


# In[7]:


# set enviornment
plt.rcParams['figure.figsize'] = (9, 9)
with open('test_params.json', 'r') as file:
    read_params = json.load(file)
game_params = read_params['params']
env = PacmanGame(**game_params)

print(game_params)

obs=env.reset()
print(obs)


# In[8]:


# get state
def get_state(obs):
    s = []
    x,y = obs['player']
    s.append(x)
    s.append(y)
    for x, y in obs['monsters']:
        s.append(x)
        s.append(y)
    for x, y in obs['diamonds']:
        s.append(x)
        s.append(y)
    for x, y in obs['walls']:
        s.append(x)
        s.append(y)
    return s


# In[9]:


def create_dqn_model(input_shape, nb_actions):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    # model.add(Dense(units=128, activation='relu'))
    # model.add(Dense(units=16, activation='relu'))
    # model.add(Dense(units=16, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


# In[10]:


obs=env.reset()
input_shape = np.array(get_state(obs)).shape
print(np.array(get_state(obs)).shape)
nb_actions = 9


# In[11]:


online_network = create_dqn_model(input_shape, nb_actions)
# online_network.load_weights('pacman/weights/weights_500000.h5f')


# In[12]:


# print structure of online network
online_network.summary()


# In[13]:


# complie online network
online_network.compile(optimizer=Adam(lr=0.0001), loss='mse')


# In[14]:


# define greedy search
def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.randint(1,9)  # random action
    else:
        return (np.argmax(q_values)+1)        # q-optimal action

replay_memory_maxlen = 1000000
replay_memory = deque([], maxlen=replay_memory_maxlen)

target_network = clone_model(online_network)
target_network.set_weights(online_network.get_weights())


# In[23]:


# Train DQN
obs=env.reset()
n_steps = 100000 # number of times
warmup = 1000 # first iterations after random random initiation before training starts
training_interval = 4
copy_steps = 1500 # number of steps after which weights of
                   # online network copied into target network
gamma = 0.99 # discount rate
batch_size = 64 # size of batch from replay memory
eps_max = 1.0 # parameters of decaying sequence of eps
eps_min = 0.05

eps_decay = 50000

name = 'PacMan'
if not os.path.exists(name):
    os.makedirs(name)

weights_folder = os.path.join(name, 'weights')
if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)

step = 0
iteration = 0
done = True
count = 0
test_reward=0

while step < n_steps:
    if done:
        count += 1
        test_reward += obs['reward']
        obs = env.reset()
    iteration += 1
    state=np.array(get_state(obs))
    q_values = online_network.predict(np.array([state]))[0]
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    action=epsilon_greedy(q_values, epsilon, nb_actions)
    next_obs = env.make_action(action)
    reward, done = next_obs['reward'], next_obs['end_game']
    next_state=np.array(get_state(next_obs))
    replay_memory.append((state, action, reward, next_state, done))
    obs = next_obs

    if iteration >= warmup and iteration % training_interval == 0:
        step += 1
        minibatch = random.sample(replay_memory, batch_size)
        replay_state = np.array([x[0] for x in minibatch]) # one batch size
        replay_action = np.array([x[1] for x in minibatch])
        replay_rewards = np.array([x[2] for x in minibatch])
        replay_next_state = np.array([x[3] for x in minibatch])
        replay_done = np.array([x[4] for x in minibatch], dtype=int)
        target_for_action = replay_rewards + (1-replay_done) * gamma * np.amax(target_network.predict(replay_next_state), axis=1)
        # print("target "+str(target_for_action))
        target = online_network.predict(replay_state)  # targets coincide with predictions ...
        target[np.arange(batch_size), (replay_action-1)] = target_for_action  #...except for targets with actions from replay
        online_network.fit(replay_state, target, epochs=step, verbose=2, initial_epoch=step-1)

        if step % copy_steps == 0:
            target_network.set_weights(online_network.get_weights())

        if count % 20 ==0:
            check_avg_reward = test_reward/20
            print("reward" + str(check_avg_reward))
            test_reward = 0

        if step % eps_decay == 0:
            eps_min = eps_min * 0.6
            eps_max = eps_max * 0.6



# In[24]:


online_network.save_weights(os.path.join(weights_folder, 'weights_{}.h5f'.format(step)))


# In[25]:


def dqn(obs):
    state=np.array(get_state(obs))
    q_values = online_network.predict(np.array([state]))[0]
    action = epsilon_greedy(q_values, 0.00, nb_actions)
    return action


# In[26]:


def preprocess(start_state):
    # make tuples from lists
    start_state['player'] = tuple(start_state['player'])
    start_state['monsters'] = [tuple(m) for m in start_state['monsters']]
    start_state['diamonds'] = [tuple(m) for m in start_state['diamonds']]
    start_state['walls'] = [tuple(m) for m in start_state['walls']]


# In[27]:


def test(strategy=dqn, log_file='train_params.json'):
    with open('test_params.json', 'r') as file:
        read_params = json.load(file)

    game_params = read_params['params']
    test_start_states = read_params['states']
    total_history = []
    total_scores = []

    env = PacmanGame(**game_params)
    for start_state in test_start_states:
        preprocess(start_state)
        episode_history = []
        env.reset()
        env.player = start_state['player']
        env.monsters = start_state['monsters']
        env.diamonds = start_state['diamonds']
        env.walls = start_state['walls']
        assert len(env.monsters) == env.nmonsters and len(env.diamonds) == env.ndiamonds and len(env.walls) == env.nwalls

        obs = env.get_obs()
        episode_history.append(obs)
        while not obs['end_game']:
            action = strategy(obs)
            obs = env.make_action(action)
            episode_history.append(obs)
        total_history.append(episode_history)
        total_scores.append(obs['total_score'])
    mean_score = np.mean(total_scores)
    with open(log_file, 'w') as file:
        json.dump(total_history, file)
    print("Your average score is {}, saved log to '{}'. Do not forget to upload it for submission!".format(mean_score, log_file))
    return mean_score



# In[28]:


test()


# In[29]:


#output json
sc=[]
for i in range(1,20,1):
    a=test(strategy=dqn,log_file='train_pacman{}.json'.format(i))
    sc.append(a)
print(max(sc),np.argmax(sc))

