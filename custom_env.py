import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from tqdm import tqdm
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

style.use('ggplot')


import wandb
from config import WANDB_API_KEY
wandb.login(key=WANDB_API_KEY)

hyperparameters = dict(
    BOARD_SIZE = 10,
    EPISODE_COUNT = 12500,
    EPISODE_LEN = 200,
    LEARNING_RATE = 0.1,
    DISCOUNT = 0.9,
    MOVE_PENALTY = 1,
    ENEMY_PENALTY = 300,
    FOOD_REWARD = 25,
    EPSILON_START = 0.9,
    EPS_DECAY = 0.99996, # Kind random??
    SHOW_EVERY = 3000,
)

start_q_table = None # Or load a existing save path

PLAYER_COLOR = (255, 175, 0)
FOOD_COLOR = (0, 255, 0)
ENEMY_COLOR = (0, 0, 255)

class Blob:
    def __init__(self, board_size=10):
        self.board_size = board_size
        self.x = np.random.randint(0, board_size)
        self.y = np.random.randint(0, board_size)
    
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
        else:
            raise ValueError(f'Invalid action type: {choice}')

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
        elif self.x > self.board_size-1:
            self.x = self.board_size-1

        if self.y < 0:
            self.y = 0
        elif self.y > self.board_size-1:
            self.y = self.board_size-1

class Game:

    STATE_ACTIVE = 0
    STATE_WON = 1
    STATE_LOST = 2

    def __init__(self, move_penalty, enemy_penalty, food_reward, board_size=10):
        self.board_size = board_size


        self.move_penalty = move_penalty
        self.enemy_penalty = enemy_penalty
        self.food_reward = food_reward

        # Create place holders
        self.player = None
        self.food = None
        self.enemy = None
        
        self._state = None

    def reset(self):
        # Create our players
        self.player = Blob(board_size=self.board_size) #TODO: Link board_size to players
        self.food = Blob(board_size=self.board_size)
        self.enemy = Blob(board_size=self.board_size)

        # Store if the game is over
        self._state = self.STATE_ACTIVE
    
        # Return the starting state
        return self.state()
    

    def step(self, action):
        # Move the player acording to the action
        self.player.action(action)

        # TODO: Maybe move food / enemy randomly (would probably need to add velocity to state )
        self.enemy.move()

        # Calculate reward
        reward = None
        if self.player == self.enemy:
            reward = -self.enemy_penalty
            self._state = self.STATE_LOST
        elif self.player == self.food:
            reward = self.food_reward
            self._state = self.STATE_WON
        else:
            reward = -self.move_penalty

        return self.state(), reward, self._state

    def done(self):
        return self._state == self.STATE_WON or self._state == self.STATE_LOST

    def state(self):
        return (self.player-self.food, self.player-self.enemy)

    def render(self):
        # Create a array to store the color of each spot on the board (3 is for R,G,B or B,G,R) 
        env = np.zeros((self.board_size, self.board_size, 3), dtype=np.uint8)
        
        # NOTE: X and Y are goofy here... not 100% which lib needs it in this format
        env[self.food.y][self.food.x] = FOOD_COLOR # Food color (dictonary kinda not needed)
        env[self.player.y][self.player.x] = PLAYER_COLOR
        env[self.enemy.y][self.enemy.x] = ENEMY_COLOR

        # NOTE: RGB... might work it might also be BGR 
        img = Image.fromarray(env, "RGB")
        img = img.resize((400,400), resample=Image.BOX) # Scale up image (previously was BOARD_SIZE x BOARD_SIZE)
        cv2.imshow("", np.array(img))

        if self.done(): # End game senarios
            # If at end of game flash frame for 500ms
            # waitKey is just used as a pause / sleep function
            if cv2.waitKey(1000) & 0xFF == ord('q'): # NOTE: Second clause is hacky (he said q breaks shit)
                return
        else:
            # If not at end of game flash frame for 1ms
            if cv2.waitKey(5) & 0xFF == ord('q'):
                return

class RenderThread:
    '''
    Race conditions? Never heard of them :)
    '''
    def __init__(self, q_table, game):
        self.game = game
        self.q_table = q_table

        self._run = False
        self._process = None

    def start(self):
        assert self._process is None

        self._run = True
        self._process = Process(target=self._render, args=(q_table, ))
        self._process.daemon = True # Run without blocking
        self._process.start()

    def stop(self):
        self._run = False
        print('Waiting for thread to rejoin...')
        self._process.join()
        self._process = None

    def update_table(self, q_table):
        self.q_table = q_table

    def _render(self, q_table):
        while self._run:
            # Reset game and store inital state
            observation = self.game.reset()

            for i in range(50):
                # Generate action
                action = np.argmax(self.q_table[observation])

                new_observation, reward, game_state = self.game.step(action)

                # Do the rendering
                self.game.render()

                # Break loop if end of episode
                if self.game.done():
                    break
                
                # Update observation for next time step
                observation = new_observation


def make_qtable(board_size=10):
    table = {}
    if start_q_table is None:

        # Observations state holds the following information
        # relative_food_pos = (x1, y1) 
        # relative enemy_pos = (x2, y2)
        # Each x or y can be in the range from 0 to BOARD_SIZE - 1
        #
        # Finally,
        # There are 3 actions per state/observation 

        # TODO: Seems hacky but okay
        for x1 in range(-board_size+1, board_size):
            for y1 in range(-board_size+1, board_size):
                for x2 in range(-board_size+1, board_size):
                    for y2 in range(-board_size+1, board_size):
                        table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
    else:
        with open(start_q_table, "rb") as f:
            raise 'Will not share state across processes'
            table = pickle.load(f)
    
    return table

def sarsa_pipline(hyperparameters):
    with wandb.init(project='Custom-SARSA', config=hyperparameters):
        config = wandb.config

        q_table, env = make(config)

        train(config, q_table, env)

        render_policy(q_table, env)


def make(config):
    q_table = make_qtable(board_size=config['BOARD_SIZE'])
    game = Game( 
        config['MOVE_PENALTY'],
        config['ENEMY_PENALTY'],
        config['FOOD_REWARD'],
        board_size=config['BOARD_SIZE'],
    )

    return q_table, game

def train(config, q_table, env):
    # Create statistics
    win_count = 0
    lose_count = 0
    episode_rewards = []

    # Create epislon for e-greedy process
    epsilon = config['EPSILON_START']
    lr = config['LEARNING_RATE']
    discount = config['DISCOUNT']
    render = False

    # Run specified number of episodes
    for episode in tqdm(range(config['EPISODE_COUNT'])):
        # Reset set up state for current episode
        action = None
        observation = env.reset()
        episode_reward = 0
        verbose_episode = (episode + 1) % config['SHOW_EVERY'] == 0

        for i in range(config['EPISODE_LEN']):
            # Select action via e-greedy method
            if np.random.random() > epsilon:
                action = np.argmax(q_table[observation])
            else:
                action = np.random.randint(0,4)
            
            # Preform action and get info back
            new_observation, reward, game_state = env.step(action)

            # Accumulate reward
            episode_reward += reward

            # Update Q-Table
            new_q = None
            if game_state == Game.STATE_ACTIVE:
                max_future_q = np.max(q_table[new_observation])
                current_q = q_table[observation][action]

                new_q = (1 - lr) * current_q + lr * (reward + discount * max_future_q)
            elif game_state == Game.STATE_LOST:
                lose_count += 1
                new_q = reward
            elif game_state == Game.STATE_WON:
                win_count += 1
                new_q = reward 

            assert new_q is not None, 'Error: Q Estimates should not be NoneType'
            q_table[observation][action] = new_q

            # Render frame if verbose
            if verbose_episode:
                env.render()
            
            if env.done():
                break
            
            observation = new_observation

        episode_rewards.append(episode_reward)

        # Decay epsilon
        epsilon *= config['EPS_DECAY']

        if verbose_episode:
            batch_mean = np.mean(episode_rewards[-config['SHOW_EVERY']:])
            print('\n\n=======================')
            print(f'EPISODE #{episode}')
            print('=======================')
            print(f'epsilon: {epsilon}')
            print(f'Mean (n={config["SHOW_EVERY"]}): {batch_mean}')
            print('=======================')
            print(f'Wins:   {win_count}')
            print(f'Losses: {lose_count}')
            print(f'Draws:  {config["SHOW_EVERY"]-(win_count+lose_count)}')
            print('=======================')
            print(f'Win precentage: {round(win_count/config["SHOW_EVERY"], 5)*100}')
            print(f'Lose Precentage: {round(lose_count/config["SHOW_EVERY"], 5)*100}')

            wandb.log({
                'episode': episode,
                'epsilon': epsilon,
                'batch_mean': batch_mean,
                'win_precentage': round(win_count/config['SHOW_EVERY'], 5)*100,
                'lose_precentage': round(lose_count/config['SHOW_EVERY'], 5)*100
            })

            # Reset stats after reset
            win_count = 0
            lose_count = 0

def render_policy(q_table, env):
    
    while True:
        action = None
        observation = env.reset()
        for i in range(200):
            action = np.argmax(q_table[observation])
            new_observation, reward, game_state = env.step(action)

            env.render()

            if env.done():
                break

            observation = new_observation



if __name__ == '__main__':
        sarsa_pipline(hyperparameters)

                

        '''
        # TODO: Very unclear how this operation works
        moving_average = np.convolve(episode_rewards, np.ones((config['SHOW_EVERY'],)) / config['SHOW_EVERY'], mode='valid')
        plt.plot([i for i in range(len(moving_average))], moving_average)
        plt.ylabel(f"Reward {config['SHOW_EVERY']}ma")
        plt.xlabel("episode #")
        plt.show()

        if False:
            with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
                pickle.dump(q_table, f)

        #render_process.stop()'''