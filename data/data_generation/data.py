import os
import matplotlib.pyplot as plt
import math
import numpy as np
import json
import random
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


def create_circle(inner_radius, outer_radius, create_internal_circles=False, visualize=False,
                  n_inner=10000, n_outer=15000):
    """ The method creates a 2-class data-set for training of a neural network. The data will be in the form of
        two circles (inner, outer). If the create_internal_circles option is true, teh method will create smaller
        internal circles of the opposite class in both the inner and the outer circle.

    :param inner_radius: (float) Radius of inner circle
    :param outer_radius: (float) Radius of outer circle
    :param create_internal_circles: (boolean) Boolean variable to control if the generated circles should include
                                              internal sub-circles of the opposite class
    :param visualize: (boolean)
    :param n_inner:
    :param n_outer:
    :return data, labels: (npArray, npArray) data array with x- & y-coordinates
                                             labels with two classes (0, 1)
    """
    dist_between_circles = 0.8  # Distance between inner and outer circles
    sub_circle_1_radius = 0.5   # Radius of sub-circles in inner circle (used if create_internal_circles)
    sub_circle_2_radius = 1.2   # Radius of sub-circles in outer circle (used if create_internal_circles)
    num_1 = n_inner               # Number of samples in inner circle
    num_2 = n_outer               # Number of samples in outer circle
    data = np.zeros([num_1 + num_2, 2])
    labels = np.zeros([num_1 + num_2, 1])

    # ===== inner circle =====
    points = [(0, inner_radius/2), (0, -inner_radius/2), (inner_radius/2, 0), (-inner_radius/2, 0)]
    for i in range(0, num_1):
        r = random.random() * inner_radius  # Random radius within scope oif inner circle
        theta_rand = random.random() * 2 * math.pi
        x = r * math.cos(theta_rand)
        y = r * math.sin(theta_rand)
        data[i, 0] = x
        data[i, 1] = y

        # Labels
        if create_internal_circles:
            dist_to_sub_circles = []
            for point in points:
                dist_to_sub_circles.append(math.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2))

            close_to_circle = False
            for dist in dist_to_sub_circles:
                if dist < sub_circle_1_radius:
                    close_to_circle = True

            if close_to_circle:
                labels[i] = 1
            else:
                labels[i] = 0
        else:
            labels[i] = 0

    # ===== outer circle =====
    mid_radius_outer_circle = inner_radius + dist_between_circles + (outer_radius - (inner_radius + dist_between_circles))/2
    projected_dist = math.sqrt((mid_radius_outer_circle ** 2) / 2)
    points = [(projected_dist, projected_dist),
              (-projected_dist, projected_dist),
              (-projected_dist, -projected_dist),
              (projected_dist, -projected_dist)]
    for i in range(num_1, num_1 + num_2):
        r = random.random() * (outer_radius - inner_radius) + inner_radius + dist_between_circles  # Random radius
        theta_rand = random.random() * 2 * math.pi
        x = r * math.cos(theta_rand)
        y = r * math.sin(theta_rand)
        data[i, 0] = x
        data[i, 1] = y

        # Labels
        if create_internal_circles:
            dist_to_sub_circles = []
            for point in points:
                dist_to_sub_circles.append(math.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2))

            close_to_circle = False
            for dist in dist_to_sub_circles:
                if dist < sub_circle_2_radius:
                    close_to_circle = True
            if close_to_circle:
                labels[i] = 0
            else:
                labels[i] = 1
        else:
            labels[i] = 1
    if visualize:
        visualize_data(data, labels)
    return data, labels


def visualize_data(data, label):
    """ Visualizes 2-dimensional data. The data should be in 2 classes.

    :param data: (npArray) 2-dimensional data
    :param label: (npArray) Labels in 2 classes
    :return:
    """
    col = []
    for i in range(0, len(label)):
        if label[i]:
            col.append('#eb4034')
        else:
            col.append('#34ebd6')
    plt.scatter(data[:, 0], data[:, 1], c=col)
    plt.axis('equal')
    plt.pause(0.001)
    plt.ion()
    plt.show()


def get_xor_data():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])
    return data, labels


def get_square_data():
    train_data = np.random.uniform(-10, 10, 2000)

    train_data = np.append(train_data, np.random.uniform(-20, -15, 2000))
    train_data = np.append(train_data, np.random.uniform(15, 20, 2000))
    train_data = train_data.reshape((train_data.shape[0], 1))
    train_output = train_data ** 2

    test_data = np.linspace(-15, 15, 1000)
    test_data = np.append(test_data, np.random.uniform(30, 40, 1000))
    test_data = test_data.reshape((test_data.shape[0], 1))
    test_output = test_data ** 2
    return  train_data, train_output, test_data, test_output

def get_cube_data():
    train_data = np.random.uniform(-12, 12, 2000)

    train_data = np.append(train_data, np.random.uniform(-20, -15, 2000))
    train_data = np.append(train_data, np.random.uniform(15, 20, 2000))
    train_data = train_data.reshape((train_data.shape[0], 1))
    train_output = train_data ** 3

    test_data = np.linspace(-15, 15, 1000)
    test_data = test_data.reshape((test_data.shape[0], 1))
    test_output = test_data ** 3
    return  train_data, train_output, test_data, test_output

def load_and_parse_dql_data(data_dir: str, ratio: float = 1.0, mode: int = 0, max_samples = None):
    """ Loads and parses episodes from flappy bird games used for training a Deep Q-learning agent.

        state:
            VerticalDistance
            Vertical Velocity
            Distance to Nearest pipe
    :param data_dir: (str) Pointer to location of episodes
    :param ratio: (float) Frame ratio of every episode to be used
    :param mode: (int) Extraction mode
                        0: samples will be randomly selected for each episode to match the ratio
                        1: a ratio of the episode will be selected and in those episodes all samples will be selected
    :param max_samples: (int) Maximum number of samples to extract
    :return: states (np.array), next_states (np.array), rewards (np.array), actions (np.array)
    """
    # Extract all Episode files
    json_files = [file for file in os.listdir(data_dir) if file.endswith('.json') and 'episode' in file]

    states = np.zeros((0, 3))    # State Matrix
    n_states = np.zeros((0, 3))  # Next state Matrix
    rewards = np.zeros((0, 1))   # Reward vector
    actions = np.zeros((0, 1))   # Action vector
    eog = np.zeros((0, 1))       # End of Game vector

    n_episode_to_extract = int(np.floor(len(json_files) * ratio))  # Number of episode to extract
    if mode == 0:
        selected_files = json_files
    else:
        file_idx = random.sample(range(len(json_files)), n_episode_to_extract)
        selected_files = [f for i, f in enumerate(json_files) if i in file_idx]

    for e_file in selected_files:
        file_dir = os.path.join(data_dir, e_file)
        if max_samples is not None and max_samples < len(states):
            break
        with (open(file_dir, 'r') as jf):
            data = json.load(jf)  # Load json file
            nframes = len(data["states"])  # Total number of frames
            nframes_to_extract = int(np.floor(nframes * ratio))  # Number of frames to extract
            selected_frames = list(range(nframes)
                                   ) if mode == 1 else random.sample(range(nframes), nframes_to_extract)

            for iFrame in selected_frames:
                s = data["states"][iFrame]  # State data
                vert_dist = s["Y"] - (s["upperY"] + s["pipeGap"] / 2)  # Vertical offset from between pipe
                f_state = np.array([vert_dist, s["vY"], s["distanceToPipe"]])  # Frame state
                states = np.vstack([states, f_state])                          # Append to states array
                actions = np.vstack([actions, np.array([s["action"]])])        # Append to action array
                rewards = np.vstack([rewards, np.array([s["reward"]])])        # Reward array

                # End of game
                if iFrame >= nframes - 1:
                    eog = np.vstack([eog, np.array([True])])
                else:
                    eog = np.vstack([eog, np.array([False])])
                n_vert_dist = s["next_Y"] - (s["next_upperY"] + s["next_pipeGap"] / 2)
                n_state = np.array([n_vert_dist, s["next_vY"], s["next_distanceToPipe"]])
                n_states = np.vstack([n_states, n_state])

    return {"states": states.transpose(),
            "n_states": n_states.transpose(),
            "actions": actions.transpose(),
            "rewards": rewards.transpose(),
            "eog": eog.transpose()}

    