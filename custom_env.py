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

class Game:

    STATE_ACTIVE = 0
    STATE_WON = 1
    STATE_LOST = 2

    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size

        # Create place holders
        self.player = None
        self.food = None
        self.enemy = None
        
        self._state = None

    def reset(self):
        # Create our players
        self.player = Blob() #TODO: Link board_size to players
        self.food = Blob()
        self.enemy = Blob()

        # Store if the game is over
        self._state = self.STATE_ACTIVE
    
        # Return the starting state
        return self.state()
    

    def step(self, action):
        # Move the player acording to the action
        self.player.action(action)

        # TODO: Maybe move food / enemy randomly (would probably need to add velocity to state )

        # Calculate reward
        reward = None
        if self.player == self.enemy:
            reward = -ENEMY_PENALTY
            self._state = self.STATE_LOST
        elif self.player == self.food:
            reward = FOOD_REWARD
            self._state = self.STATE_WON
        else:
            reward = -MOVE_PENALTY

        return self.state(), reward, self._state

    def done(self):
        return self._state == self.STATE_WON or self._state == self.STATE_LOST

    def state(self):
        return (self.player-self.food, self.player-self.enemy)

    def render(self):
        # Create a array to store the color of each spot on the board (3 is for R,G,B or B,G,R) 
        env = np.zeros((self.board_size, self.board_size, 3), dtype=np.uint8)
        
        # NOTE: X and Y are goofy here... not 100% which lib needs it in this format
        env[self.food.y][self.food.x] = d[FOOD_N] # Food color (dictonary kinda not needed)
        env[self.player.y][self.player.x] = d[PLAYER_N]
        env[self.enemy.y][self.enemy.x] = d[ENEMY_N]

        # NOTE: RGB... might work it might also be BGR 
        img = Image.fromarray(env, "RGB")
        img = img.resize((300,300), resample=Image.BOX) # Scale up image (previously was BOARD_SIZE x BOARD_SIZE)
        cv2.imshow("", np.array(img))

        if self.done(): # End game senarios
            # If at end of game flash frame for 500ms
            # waitKey is just used as a pause / sleep function
            if cv2.waitKey(500) & 0xFF == ord('q'): # NOTE: Second clause is hacky (he said q breaks shit)
                return
        else:
            # If not at end of game flash frame for 1ms
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

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

game = Game()

win_count = 0
lose_count = 0
episode_rewards = []
for episode in tqdm(range(EPISODES_LEN)):
    observation = game.reset()

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
        if np.random.random() > EPSILON:
            action = np.argmax(q_table[observation])
        else:
            action = np.random.randint(0,4)
        
        new_observation, reward, game_state = game.step(action)
    

        new_q = None
        if game_state == Game.STATE_ACTIVE:
            # Preform update to Q
            max_future_q = np.max(q_table[new_observation])
            #print(observation, action)
            current_q = q_table[observation][action]

            # WARNING: Different than tutorial but 90% should work 
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        elif game_state == Game.STATE_LOST:
            lose_count += 1
            new_q = reward
        elif game_state == Game.STATE_WON:
            win_count += 1
            new_q = reward 

        assert new_q is not None, 'Error: Q Estimates should not be NoneType'

        # Update the Q estimate
        q_table[observation][action] = new_q
        # Accumulate the total reward after each time step
        episode_reward += reward

        if show:
            game.render()

        # Break loop if end of episode
        if game.done():
            break
        
        # Update observation for next time step
        observation = new_observation

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