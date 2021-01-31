import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from tqdm import tqdm

style.use('ggplot')

BOARD_SIZE = 10
EPISODES_LEN = 25000

MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

LEARNING_RATE = 0.1
DISCOUNT = 0.95

EPSILON = 0.9
EPS_DECAY = 0.9998 # Kind random??

SHOW_EVERY = 3000

start_q_table = None # Or load a existing save path

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {
    1: (255, 175, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, BOARD_SIZE)
        self.y = np.random.randint(0, BOARD_SIZE)
    
    def __str__(self):
        return f'{self.x}, {self.y}'
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2) # [-1, 2) ## -1 to 1
        else:
            self.x += x
        
        if not y:
            self.y += np.random.randint(-1, 2) # [-1, 2) ## -1 to 1
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > BOARD_SIZE-1:
            self.x = BOARD_SIZE-1

        if self.y < 0:
            self.y = 0
        elif self.y > BOARD_SIZE-1:
            self.y = BOARD_SIZE-1

q_table = {}

if start_q_table is None:

    # Observations state holds the following information
    # relative_food_pos = (x1, y1) 
    # relative enemy_pos = (x2, y2)
    # Each x or y can be in the range from 0 to BOARD_SIZE - 1
    #
    # Finally,
    # There are 3 actions per state/observation 

    # TODO: Seems hacky but okay
    for x1 in range(-BOARD_SIZE+1, BOARD_SIZE):
        for y1 in range(-BOARD_SIZE+1, BOARD_SIZE):
            for x2 in range(-BOARD_SIZE+1, BOARD_SIZE):
                for y2 in range(-BOARD_SIZE+1, BOARD_SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


win_count = 0
lose_count = 0
episode_rewards = []
for episode in tqdm(range(EPISODES_LEN)):
    player = Blob()
    food = Blob()
    enemy = Blob()

    

    if episode % SHOW_EVERY == 0:
        print('\n\n=======================')
        print(f'EPISODE #{episode}, epsilon: {EPSILON}')
        print('=======================')
        print(f'epsilon: {EPSILON}')
        print(f'Mean (n={SHOW_EVERY}): {np.mean(episode_rewards[-SHOW_EVERY:])}')
        print('=======================')
        print(f'Wins:   {win_count}')
        print(f'Losses: {lose_count}')
        print(f'Draws:  {SHOW_EVERY-(win_count+lose_count)}')
        print('=======================')
        print(f'Win precentage: {round(win_count/SHOW_EVERY, 5)*100}')
        print(f'Lose Precentage: {round(lose_count/SHOW_EVERY, 5)*100}')

        # Reset stats after reset
        win_count = 0
        lose_count = 0

        show = True
    else:
        show = False
    
    episode_reward = 0
    for i in range(200):
        # Calculate the relative positioning and use as the state/obseravation
        observation = (player-food, player-enemy)

        if np.random.random() > EPSILON:
            action = np.argmax(q_table[observation])
        else:
            action = np.random.randint(0,4)
        
        player.action(action)

        # TODO: Randomly move the enemy and food movement

        # Update Q
        if player == enemy:
            lose_count += 1
            reward = -ENEMY_PENALTY
        elif player == food:
            win_count += 1
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
    
        # Preform update to Q
        new_observation = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_observation])
        #print(observation, action)
        current_q = q_table[observation][action]

        # WARNING: Different than tutorial but 90% should work 
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[observation][action] = new_q

        if show:
            # Create a array to store the color of each spot on the board (3 is for R,G,B or B,G,R) 
            env = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
            
            # NOTE: X and Y are goofy here... not 100% which lib needs it in this format
            env[food.y][food.x] = d[FOOD_N] # Food color (dictonary kinda not needed)
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            # NOTE: RGB... might work it might also be BGR 
            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300)) # Scale up image (previously was BOARD_SIZE x BOARD_SIZE)
            cv2.imshow("", np.array(img))

            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY: # End game senarios
                # If at end of game flash frame for 500ms
                # waitKey is just used as a pause / sleep function
                if cv2.waitKey(500) & 0xFF == ord('q'): # NOTE: Second clause is hacky (he said q breaks shit)
                    break
            else:
                # If not at end of game flash frame for 1ms
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Accumulate the total reward after each time step
        episode_reward += reward

        # Break loop if end of episode
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)

    # Decay epsilon
    EPSILON *= EPS_DECAY

# TODO: Very unclear how this operation works
moving_average = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_average))], moving_average)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

if False:
    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)