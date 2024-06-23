import os
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import json
import random

def create_circle(inner_radius, outer_radius, create_internal_circles=False):
    """ The method creates a 2-class data-set for training of a neural network. The data will will be in the form of
        two circles (inner, outer). If the create_internal_circles option is true, teh method will create smaller
        internal circles of the opposite class in both the inner and the outer circle.

    :param inner_radius: (float) Radius of inner circle
    :param outer_radius: (float) Radius of outer circle
    :param create_internal_circles: (boolean) Boolean variable to control if the generated circles should include
                                              internal sub-circles of the opposite class
    :return data, labels: (npArray, npArray) data array with x- & y-coordinates
                                             labels with two classes (0, 1)
    """
    dist_between_circles = 0.8  # Distance between inner and outer circles
    sub_circle_1_radius = 0.5   # Radius of sub-circles in inner circle (used if create_internal_circles)
    sub_circle_2_radius = 1.2   # Radius of sub-circles in outer circle (used if create_internal_circles)
    num_1 = 10000               # Number of samples in inner circle
    num_2 = 15000               # Number of samples in outer circle
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


def load_and_parse_dql_data(data_dir: str, ratio: float = 0.5):
    """ Loads and parses episodes from flappy bird games used for training a Deep Q-learning agent.

        state:
            y
            vy
            ay
            uy
            gap
            dist
    :param data_dir: (str) Pointer to location of episodes
    :param ratio: (float) Frame ratio of every episode to be used
    :return: states (np.array), next_states (np.array), rewards (np.array), actions (np.array)
    """
    # Get list of all episode files
    # for every episode file, determine number of frames and select random frames equal to the ratio
    # Extract state, next_state, reward, action from the selected frames

    # Extract all Episode files
    json_files = [file for file in os.listdir(data_dir) if file.endswith('.json') and 'episode' in file]

    states = np.zeros((0, 6))    # State Matrix
    n_states = np.zeros((0, 6))  # Next state Matrix
    rewards = np.zeros((0, 1))   # Reward vector
    actions = np.zeros((0, 1))   # Action vector
    eog = np.zeros((0, 1))       # End of Game vector

    for e_file in json_files:
        file_dir = os.path.join(data_dir, e_file)
        with open(file_dir, 'r') as jf:
            data = json.load(jf)
            nframes = len(data["states"])
            nframes_to_extract = int(np.floor(nframes * ratio))
            selected_frames = random.sample(range(nframes), nframes_to_extract)

            for iFrame in selected_frames:
                s = data["states"][iFrame]
                f_state = np.array([s["Y"], s["vY"], s["aY"], s["distanceToPipe"], s["pipeGap"], s["upperY"]])
                states = np.vstack([states, f_state])
                actions = np.vstack([actions, np.array([s["action"]])])
                rewards = np.vstack([rewards, np.array([s["reward"]])])

                if iFrame >= nframes - 1:
                    ns = data["states"][iFrame]
                    eog = np.vstack([eog, np.array([True])])
                else:
                    ns = data["states"][iFrame + 1]
                    eog = np.vstack([eog, np.array([False])])
                n_state = np.array([ns["Y"], ns["vY"], ns["aY"], ns["distanceToPipe"], ns["pipeGap"], ns["upperY"]])
                n_states = np.vstack([n_states, n_state])
    return {"states": states.transpose(),
            "n_states": n_states.transpose(),
            "actions": actions.transpose(),
            "rewards": rewards.transpose(),
            "eog": eog.transpose()}

    