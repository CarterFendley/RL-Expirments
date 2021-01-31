from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm
import numpy as np
import os

def png(path):

    png_path = os.path.join(path, 'table_png')

    assert os.path.isdir(png_path) != True, 'Directory for pngs already exists; check file structure'

    os.makedirs(png_path)

    style.use('ggplot')


    def get_q_color(value, vals):
        if value == max(vals):
            return "green", 1.0
        else:
            return "red", 0.3


    fig = plt.figure(figsize=(12, 9))

    q_table = None

    for i in tqdm(range(0, 25000, 10)):
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        q_table = np.load(f"{path}/table_data/qtable-{i}.npy")

        for x, x_vals in enumerate(q_table):
            for y, y_vals in enumerate(x_vals):
                ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
                ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
                ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])

                ax1.set_ylabel("Action 0")
                ax2.set_ylabel("Action 1")
                ax3.set_ylabel("Action 2")

        #plt.show()
        plt.savefig(f"{png_path}/{i}.png")
        plt.clf()

if __name__ == '__main__':
    path = input('Save path: ')
    png(path)