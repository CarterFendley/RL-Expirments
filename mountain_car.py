import os
import gym
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

#print(env.observation_space.high)
#print(env.observation_space.low)
#print(env.action_space.n)

# [20] * number of deminsions
# [20, 20] for mountain car
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 # in divsion
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

SHOW_EVERY = 2000

log_qtables = True
start_time = datetime.now()
DIRNAME = os.path.dirname(__file__)
SAVE_DIR = os.path.join(DIRNAME, f'data/Run {start_time}/table_data/')
def save(episode, q_table):
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    np.save(os.path.join(SAVE_DIR, f'qtable-{episode}.npy'), q_table)


DISCRETE_O_SPACE_SIZE = [20] * len(env.observation_space.high)
# Window / bucket size for mapping of continous space to descrete
descrete_o_space_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_O_SPACE_SIZE
#print(descrete_o_space_win_size)

'''
Low / High rationalization: The rewards for mountain car are -1 (besides goal)...
can be reasonably sure that the values will fall somewhere in there??
'''
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_O_SPACE_SIZE + [env.action_space.n])) # size = [20, 20, N (actions)]... random table

episode_rewards = []
rewards = {'episode': [], 'average': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / descrete_o_space_win_size
    return tuple(discrete_state.astype(np.int))


for episode in tqdm(range(EPISODES)):
    # Keep track of reward per episode
    total_reward = 0

    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    # Env reset returns a starting state
    discrete_state = get_discrete_state(env.reset())

    done = False

    while not done:
        # Take the most valuable action from the current state
        # 0 left 1 nothing 2 right
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            # Get the current q estimate of the state and the action we took
            current_q = q_table[discrete_state + (action, )]
            # Get the q value of the new state and the best action in that state
            max_future_q = np.max(q_table[new_discrete_state])

            # TODO: Why is this (1-LEARNING_RATE) term here I can't find it anywhere
            # Probably cause the other one causes int overflow
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            #new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            #print(new_q)
            q_table[discrete_state+(action, )] = new_q
        elif new_state[0] >= env.goal_position:
            #print(f'Goal reached on episode {episode}!')
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    episode_rewards.append(total_reward)

    if log_qtables and episode % 10 == 0:
        save(episode, q_table)
    

    # if episode % SHOW_EVERY == 0 
    if not episode % SHOW_EVERY:
        # Get the average reward of SHOW_EVERY number of elements
        average_reward = sum(episode_rewards[-SHOW_EVERY:])/len(episode_rewards[-SHOW_EVERY:])
        max_reward = max(episode_rewards[-SHOW_EVERY:])
        min_reward = min(episode_rewards[-SHOW_EVERY:])



        # Log the current episode
        rewards['episode'].append(episode)
        rewards['average'].append(average_reward) 
        rewards['max'].append(max_reward) # Min of last SHOW_EVERY episodes
        rewards['min'].append(min_reward) # Max of ... ^ ^ 

        print(f'Episode: {episode} average: {average_reward} min: {min_reward} max: {max_reward}')


env.close()

plt.plot(rewards['episode'], rewards['average'], label='avg')
plt.plot(rewards['episode'], rewards['min'], label='min')
plt.plot(rewards['episode'], rewards['max'], label='max')
plt.legend(loc=4)
plt.show()