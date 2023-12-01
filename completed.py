import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6


#  plot policy with arrows in four direction
#  to understand policy better
def plot_policy_arrows(policy, custom_map):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    n = len(custom_map)
    m = len(custom_map[0])

    if n != 1:
        fig, ax = plt.subplots(n, m, figsize=(m, n))
        for i in range(n):
            for j in range(m):
                ax[i, j].set_xlim([0, 3])
                ax[i, j].set_ylim([0, 3])
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
        for state, subdict in policy.items():
            row = state // m
            col = state % m
            if custom_map_flaten[state] == "S":
                square_fill = plt.Rectangle(
                    (0.5, 0.5),
                    2,
                    2,
                    linewidth=0,
                    edgecolor=None,
                    facecolor="y",
                    alpha=0.5,
                )
                ax[row, col].add_patch(square_fill)
            for direction, value in subdict.items():
                dx, dy = 0, 0
                if direction == 0:
                    dx = -value
                elif direction == 1:
                    dy = -value
                elif direction == 2:
                    dx = value
                else:
                    dy = value
                if value != 0:
                    ax[row, col].arrow(
                        1.5, 1.5, dx, dy, head_width=0.35, head_length=0.25
                    )
            if (
                subdict[0] == 0
                and subdict[1] == 0
                and subdict[2] == 0
                and subdict[3] == 0
            ):
                if custom_map_flaten[state] == "G":
                    color = "g"
                else:
                    color = "r"
                square_fill = plt.Rectangle(
                    (0.5, 0.5),
                    2,
                    2,
                    linewidth=0,
                    edgecolor=None,
                    facecolor=color,
                    alpha=0.5,
                )
                ax[row, col].add_patch(square_fill)
    else:
        fig, ax = plt.subplots(n, m, figsize=(m, n))
        for i in range(n):
            for j in range(m):
                ax[j].set_xlim([0, 3])
                ax[j].set_ylim([0, 3])
                ax[j].set_xticks([])
                ax[j].set_yticks([])
        for state, subdict in policy.items():
            row = state // m
            col = state % m
            if custom_map_flaten[state] == "S":
                square_fill = plt.Rectangle(
                    (0.5, 0.5),
                    2,
                    2,
                    linewidth=0,
                    edgecolor=None,
                    facecolor="y",
                    alpha=0.5,
                )
                ax[col].add_patch(square_fill)
            for direction, value in subdict.items():
                dx, dy = 0, 0
                if direction == 0:
                    dx = -value
                elif direction == 1:
                    dy = -value
                elif direction == 2:
                    dx = value
                else:
                    dy = value
                if value != 0:
                    ax[col].arrow(1.5, 1.5, dx, dy, head_width=0.35, head_length=0.25)
            if (
                subdict[0] == 0
                and subdict[1] == 0
                and subdict[2] == 0
                and subdict[3] == 0
            ):
                if custom_map_flaten[state] == "G":
                    color = "g"
                else:
                    color = "r"
                square_fill = plt.Rectangle(
                    (0.5, 0.5),
                    2,
                    2,
                    linewidth=0,
                    edgecolor=None,
                    facecolor=color,
                    alpha=0.5,
                )
                ax[col].add_patch(square_fill)
    plt.show()


import math

import numpy as np

import numpy as np


def plot_policy_arrows_1dm(policy, custom_map):
    custom_map_flaten = list(custom_map)
    n = 1 if isinstance(custom_map[0], str) else len(custom_map)
    m = len(custom_map[0]) if isinstance(custom_map[0], list) else len(custom_map)

    fig, ax = plt.subplots(n, m, figsize=(8, 8))
    if n == 1:
        ax = np.array([ax])
    for i in range(n):
        for j in range(m):
            ax[i][j].set_xlim([0, 3])
            ax[i][j].set_ylim([0, 3])
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    for state, subdict in policy.items():
        row = 0 if isinstance(custom_map[0], str) else state // m
        col = state if isinstance(custom_map[0], str) else state % m
        if custom_map_flaten[state] == "S":
            square_fill = plt.Rectangle(
                (0.5, 0.5), 2, 2, linewidth=0, edgecolor=None, facecolor="y", alpha=0.5
            )
            ax[row][col].add_patch(square_fill)
        for direction, value in subdict.items():
            dx, dy = 0, 0
            if direction == 0:
                dx = -value
            elif direction == 1:
                dy = -value
            elif direction == 2:
                dx = value
            else:
                dy = value
            if value != 0:
                ax[row][col].arrow(1.5, 1.5, dx, dy, head_width=0.35, head_length=0.25)
        if subdict[0] == 0 and subdict[1] == 0 and subdict[2] == 0 and subdict[3] == 0:
            if custom_map_flaten[state] == "G":
                color = "g"
            else:
                color = "r"
            square_fill = plt.Rectangle(
                (0.5, 0.5),
                2,
                2,
                linewidth=0,
                edgecolor=None,
                facecolor=color,
                alpha=0.5,
            )
            ax[row][col].add_patch(square_fill)
    plt.show()


# plot policy in terminal using best action
# for each state
def plot_policy_terminal(policy, custom_map):
    arr = np.empty((len(custom_map), len(custom_map[0])), dtype=object)
    state = 0
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            subdict = policy[state]
            best_action = max(subdict, key=subdict.get)

            if best_action == 0:
                arr[i, j] = "Lt"  # 0: LEFT
            elif best_action == 1:
                arr[i, j] = "Dn"  # 1: DOWN
            elif best_action == 2:
                arr[i, j] = "Rt"  # 2: RIGHT
            elif best_action == 3:
                arr[i, j] = "UP"  # 3: UP
            else:
                arr[i, j] = "##"
            state += 1
    print(arr)


def plot_state_value(state_value, custom_map):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    rows = len(custom_map)
    cols = len(custom_map[0])
    table = state_value.reshape(rows, cols)

    # Define custom colors
    green = mcolors.to_rgba("green", alpha=0.5)
    blue = mcolors.to_rgba("blue", alpha=0.5)

    fig, ax = plt.subplots()
    im = ax.imshow(table, cmap="Reds")

    # Update specific cells with custom colors
    # ax.add_patch(mpatches.Rectangle(xy=(3-0.5, 2-0.5), width=1, height=1, linewidth=0, facecolor=green))
    # ax.add_patch(mpatches.Rectangle(xy=(1-0.5, 4-0.5), width=1, height=1, linewidth=0, facecolor=blue))

    state = 0
    for i in range(rows):
        for j in range(cols):
            if custom_map_flaten[state] == "H":
                ax.add_patch(
                    mpatches.Rectangle(
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        linewidth=0.1,
                        facecolor=blue,
                    )
                )
            elif custom_map_flaten[state] == "G":
                ax.add_patch(
                    mpatches.Rectangle(
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        linewidth=0,
                        facecolor=green,
                    )
                )

            ax.text(
                j,
                i,
                str(i * cols + j) + "\n" + custom_map_flaten[state],
                ha="center",
                va="center",
            )
            state += 1

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([""] * cols)
    ax.set_yticklabels([""] * rows)

    # Update the colorbar limits to reflect the new colors
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")

    ax.set_title("state value")
    plt.show()


###################################################################


# it gives a randome walk policy w.r.t costum map(type->dict)
def get_init_policy(custom_map):
    policy = {}
    random_sub_dict = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    terminal_sub_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            state = (i * len(custom_map[i])) + j
            if custom_map[i][j] == "H" or custom_map[i][j] == "G":
                # custom_map[i][j] == "S" or
                policy[state] = terminal_sub_dict
            else:
                policy[state] = random_sub_dict

    return policy

def get_policy_dirrection(direction, custom_map): # 'left' 
    policy = {}
    left_sub_dict  = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
    down_sub_dict  = {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0}
    right_sub_dict = {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}
    terminal_sub_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            state = (i * len(custom_map[i])) + j
            if custom_map[i][j] == "H" or custom_map[i][j] == "G":
                # custom_map[i][j] == "S" or
                policy[state] = terminal_sub_dict
            else:
                if(direction == 'left'):
                    policy[state] = left_sub_dict
                elif(direction == 'down'):
                    policy[state] = down_sub_dict
                elif(direction == 'right'):
                    policy[state] = right_sub_dict

    return policy

def act_wrt_prob(probability):
    if random.random() < probability:
        return 1
    else:
        return 0


def get_action_wrt_policy(state, policy):
    action = -1
    while action == -1:
        if act_wrt_prob(policy[state][0]) == 1:
            action = 0
        elif act_wrt_prob(policy[state][1]) == 1:
            action = 1
        elif act_wrt_prob(policy[state][2]) == 1:
            action = 2
        elif act_wrt_prob(policy[state][3]) == 1:
            action = 3
    return action


def do_policy(env, policy, episdoes=5):
    # episdoes = 10
    for ep in range(episdoes):
        n_state = env.reset()[0]
        done = False
        rewards = 0
        moves = 0
        while done is False:
            env.render()
            action = get_action_wrt_policy(n_state, policy)
            n_state, reward, done, imFalse, info = env.step(action)
            rewards += reward
            moves += 1
        print("rewards:", rewards, " - moves:", moves, " - final state:", n_state)


def get_flaten_custom_map(custom_map):
    custom_map_flaten = []
    for row in custom_map:
        for char in row:
            custom_map_flaten.append(char)
    return custom_map_flaten


def dis_state_value(V1, V2):
    Distance = 0
    for i in range(V1.shape[0]):
        Distance += np.abs(V1[i] - V2[i])

    return Distance


###################################################################


    def policy_iteration(
        env, custom_map, max_ittr=30, theta=0.01, discount_factor=0.9, do_print=False
    ):
        nrows = len(custom_map)
        ncols = len(custom_map[0])
        terminals_list = []
        holes_list = []

        for i in range(nrows):
            for j in range(ncols):
                state = (i * len(custom_map[i])) + j
                if custom_map[i][j] == "H" or custom_map[i][j] == "G":
                    terminals_list.append(state)
                    if custom_map[i][j] == "H":
                        holes_list.append(state)

        policy = get_init_policy(custom_map)
        V = np.zeros(env.observation_space.n)

        # discount_factor = 0.9
        # max_ittr = 30
        # theta = 0.01
        P = env.P
        ittr = 0
        policy_stable = False

        while not policy_stable and ittr < max_ittr:
            ittr += 1
            delta = theta + 1
            while delta > theta:
                delta = 0
                for state in range(0, env.observation_space.n):
                    v = V[state]
                    new_V_state = 0
                    for curr_action in policy[state].keys():
                        prob_of_selecting_action = policy[state][curr_action]

                        for idx, transition in enumerate(P[state][curr_action]):
                            (prob_of_transition, s_prime, reward, terminated) = transition
                            new_V_state += (
                                prob_of_selecting_action
                                * prob_of_transition
                                * (reward + discount_factor * V[s_prime])
                            )
                    V[state] = new_V_state

                    delta = max(delta, np.abs(V[state] - v))

            policy_stable = True
            q_values_list = []
            best_action_list = []
            old_policy = copy.deepcopy(policy)
            for s in range(env.observation_space.n):
                if s not in terminals_list:
                    old_action = np.argmax(policy[s])
                    q_values = np.zeros(env.action_space.n)
                    for a in range(env.action_space.n):
                        for prob, next_state, reward, done in P[s][a]:
                            q_values[a] += reward + discount_factor * V[next_state]
                    best_action = np.argmax(q_values)
                    policy[s] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
                    policy[s][best_action] = 1
                    q_values_list.append(q_values)
                    best_action_list.append(best_action)
                    if old_policy[s][old_action] != policy[s][np.argmax(policy[s])]:
                        policy_stable = False

            if do_print:
                V_2 = V.reshape((nrows, ncols))
                print("ittr :", ittr)
                print("v :", np.round(V_2, 2))
                plot_policy_terminal(policy, custom_map)
                print("==============")
        # print("delta:", delta)
        return V, policy


###################################################################


def first_visit_mc_prediction(env, policy, num_episodes, gamma):
    # initilize
    V = np.zeros(env.observation_space.n)
    N = np.zeros(env.observation_space.n)
    Returns = [[] for _ in range(env.observation_space.n)]

    # loop in range num_episodes(for each episode)
    for i_episode in range(num_episodes):
        # generate episode w.r.t policy
        episode = []
        state = env.reset()[0]
        done = False
        while not done:
            action = get_action_wrt_policy(state, policy)
            next_state, reward, done, imFalse, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        # loop for each step of episode , t= T-1, T-2, ..., 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if state not in [x[0] for x in episode[:t]]:
                N[state] += 1
                Returns[state].append(G)
                V[state] = np.mean(Returns[state])

    return V


###################################################################


def every_visit_mc_prediction(env, policy, num_episodes, gamma):
    # initilize
    V = np.zeros(env.observation_space.n)
    N = np.zeros(env.observation_space.n)
    Returns = [[] for _ in range(env.observation_space.n)]

    # loop in range num_episodes(for each episode)
    for i_episode in range(num_episodes):
        # generate episode w.r.t policy
        episode = []
        state = env.reset()[0]
        done = False
        while not done:
            action = get_action_wrt_policy(state, policy)
            next_state, reward, done, imFalse, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        # loop for each step of episode , t= T-1, T-2, ..., 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            N[state] += 1
            Returns[state].append(G)
            V[state] = np.mean(Returns[state])

    return V


###################################################################


def modify_rewards(next_state, custom_map, hole_reward, goal_reward, move_reward):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    state_type = custom_map_flaten[next_state]

    if state_type == "H":
        return hole_reward  # Decrease the reward for falling into a hole
    elif state_type == "G":
        return goal_reward  # Increase the reward for reaching the goal
    else:
        return move_reward  # Decrease the reward for moving


def modify_rewards_P(envP, custom_map, hole_reward, goal_reward, move_reward):
    custom_map_flaten = []
    for row in custom_map:
        for char in row:
            custom_map_flaten.append(char)

    old_envP = copy.deepcopy(envP)
    new_envP = {}

    for state, v1 in old_envP.items():
        inner_dict = {}
        for action, v2 in v1.items():
            inner_list = []
            for tpl in v2:
                (prob_of_transition, s_prime, old_reward, terminated) = tpl
                if custom_map_flaten[s_prime] == "H":
                    new_reward = (
                        hole_reward  # Decrease the reward for falling into a hole
                    )
                elif custom_map_flaten[s_prime] == "G":
                    new_reward = (
                        goal_reward  # Increase the reward for reaching the goal
                    )
                else:
                    new_reward = move_reward  # Decrease the reward for movin
                inner_list.append((prob_of_transition, s_prime, new_reward, terminated))
            inner_dict[action] = inner_list
        new_envP[state] = inner_dict

    return new_envP


class ModifyRewards(gym.Wrapper):
    def __init__(
        self, env, custom_map, hole_reward=-10, goal_reward=10, move_reward=-0.1
    ):
        super().__init__(env)
        self.hole_reward = hole_reward
        self.goal_reward = goal_reward
        self.move_reward = move_reward
        self.custom_map = custom_map
        self.P = modify_rewards_P(
            env.P, custom_map, hole_reward, goal_reward, move_reward
        )

    def step(self, action):
        next_state, reward, done, imFalse, info = self.env.step(action)
        modified_reward = modify_rewards(
            next_state,
            self.custom_map,
            self.hole_reward,
            self.goal_reward,
            self.move_reward,
        )
        return next_state, modified_reward, done, imFalse, info


###################################################################

custom_map_1 = ["HFSFFFFG"]

custom_map_2 = ["SFFFF", "HHHFF", "FFFFH", "FFFFF", "FFFFG"]

custom_map_3 = ["FFFFF", 
                "SHHHF", 
                "FHGHF", 
                "FHHHF", 
                "FFFFF"]


custom_map_4 = ["FFFSFFF", 
                "FHHHHFF", 
                "FFFFFFF", 
                "HFFFFFF", 
                "FGFFFFF"]

custom_map_5 = ["HFSFFFFG"]

custom_map_6 = ["HFSFFFFG"
               ,"HFFFFFFF"
               ,"HFFFFFFF"]

custom_map_7 = ["SFFFF",
                "FFFFH", 
                "HHFFF", 
                "HFFFH", 
                "FFFFG"]

custom_map_8 = ["HFFSFFH",
                "FFFFFFF", 
                "FFFFFFF", 
                "GFFHFFG"]


# custom_map_3 = ["SFFFFFF", "HFFFFFF", "HFFFFFF", "HFFFFFF", "GFFFFFF"]
# custom_map_4 = [
#     "SFFFFFFFFF",
#     "HFFFFFFHFF",
#     "HFFFFFFFFF",
#     "HFFHFFFHFF",
#     "HFFFFHFFFH",
#     "HFFFFFFFFF",
#     "FFFFHFFFFG",
# ]

##############################


#############################
map = custom_map_1
hole_reward_ = -2
goal_reward_ = 50
move_reward_ = -1

env = gym.make("FrozenLake-v1", render_mode="human", desc=map, is_slippery=False)
# env = gym.make("FrozenLake-v1", desc=map, is_slippery=True)
env = ModifyRewards(
    env, custom_map=map, hole_reward=hole_reward_, goal_reward=goal_reward_, move_reward=move_reward_
)
env.reset()
env.render()
V, policy = policy_iteration(
    env, map, theta=0.0001, discount_factor=0.9, do_print=False
)

# plot_state_value(V, map)
# do_policy(env, policy, episdoes=5)
# policy_left = get_policy_dirrection("left", map)
# policy_down = get_policy_dirrection("down", map)
# policy_right = get_policy_dirrection("right", map)

policy = get_init_policy(map)
plot_policy_arrows(policy, map)
# env = gym.make("FrozenLake-v1", render_mode="human", desc=map, is_slippery=False)
# # env = gym.make("FrozenLake-v1", desc=map, is_slippery=True)
# env = ModifyRewards(
#     env, custom_map=map, hole_reward=hole_reward_, goal_reward=goal_reward_, move_reward=move_reward_
# )
# num_episodes = 1000
# gamma = 0.9
# V_F_MC = first_visit_mc_prediction(env, policy_left, num_episodes, gamma)
# V_E_MC = every_visit_mc_prediction(env, policy_left, num_episodes, gamma)
# plot_state_value(V_F_MC, map)
# plot_state_value(V_E_MC, map)


# V_F_MC = first_visit_mc_prediction(env, policy_down, num_episodes, gamma)
# V_E_MC = every_visit_mc_prediction(env, policy_down, num_episodes, gamma)
# plot_state_value(V_F_MC, map)
# plot_state_value(V_E_MC, map)
# V_F_MC = first_visit_mc_prediction(env, policy_right, num_episodes, gamma)
# V_E_MC = every_visit_mc_prediction(env, policy_right, num_episodes, gamma)
# plot_state_value(V_F_MC, map)
# plot_state_value(V_E_MC, map)

# env = gym.make("FrozenLake-v1", desc=map, is_slippery=True)
# env = ModifyRewards(
#     env, custom_map=map, hole_reward=hole_reward_, goal_reward=goal_reward_, move_reward=move_reward_
# )
# env.reset()
# V, policy = policy_iteration(
#     env, map, theta=0.0001, discount_factor=0.9, do_print=False
# )
# plot_policy_arrows(policy, map)

discount_factor_list = [1, 0.9, 0.5, 0.1]
# ###
# policy = get_init_policy(map)
# # print(policy)

# policy_list = []
# V_list = []
# for df in discount_factor_list:

#     V, policy = policy_iteration(
#         env, map, theta=0.0001, discount_factor=df, do_print=False
#     )

#     plot_policy_terminal(policy, map)
#     policy_list.append(policy)
#     V_list.append(V)

# plot_state_value(V, map)
# plot_policy_arrows(policy, map)
# plot_policy_terminal(policy, map)

# num_episodes = 100
# gamma = 0.9
# V_F_MC = first_visit_mc_prediction(env, policy, num_episodes, gamma)
# V_E_MC = every_visit_mc_prediction(env, policy, num_episodes, gamma)


# print(str(dis_state_value(V, V_MC)))
# print(V_MC)
# for i in range (len(discount_factor_list)):
#     # plot_policy_arrows(policy_list[i], map)


#     print(np.round(V_list[i][0:4], 2))
#     plot_policy_arrows(policy_list[i], map)
# plot_state_value(V_list[i], map)
# plot_state_value(V_F_MC, map)
# plot_state_value(V_E_MC, map)
# print(str(dis_state_value(V_F_MC, V_E_MC)))


# print(V)

# do_policy(env, policy, episdoes=10)
# time.sleep(2)


########################################################################


###
# rewards = 0
# for t in range(100):
#     action = env.action_space.sample()
#     next_state, reward, done, imFalse, info = env.step(action)

#     # action = 2
#     # next_state, reward, done, imFalse, info = env.step(action)
#     # rewards += reward
#     # action = 2
#     # next_state, reward, done, imFalse, info = env.step(action)
#     # rewards += reward
#     # action = 2
#     # next_state, reward, done, imFalse, info = env.step(action)
#     # rewards += reward
#     # action = 2
#     # next_state, reward, done, imFalse, info = env.step(action)
#     # rewards += reward
#     # action = 1
#     # next_state, reward, done, imFalse, info = env.step(action)
#     # rewards += reward
#     # action = 1
#     # next_state, reward, done, imFalse, info = env.step(action)
#     # rewards += reward
#     # action = 1
#     # next_state, reward, done, imFalse, info = env.step(action)
#     # rewards += reward
#     # action = 1
#     # next_state, reward, done, imFalse, info = env.step(action)
#     # rewards += reward

#     env.render()
#     rewards += reward
#     if done:
#         break
# print(rewards)
# env.close()


# if __name__ == "__main__":
# map = custom_map_1
# env = gym.make("FrozenLake-v1", render_mode="human", desc=map, is_slippery=True)
# # env = gym.make("FrozenLake-v1", desc=map, is_slippery=True)

# env = ModifyRewards(env, custom_map=map, hole_reward=0, goal_reward=1, move_reward=0)
# env.reset()
# env.render()
# ###
# policy = get_init_policy(map)
# plot_policy_arrows(policy, map)
# # print(policy)

# do_policy(env, policy)
# rewards = 0
# for t in range(100):
#     action = env.action_space.sample()
#     next_state, reward, done, truncated, info = env.step(action)
#     rewards += reward

#     # action = 2
#     # next_state, reward, done, truncated, info = env.step(action)
#     # rewards += reward
#     # action = 1
#     # next_state, reward, done, truncated, info = env.step(action)
#     # rewards += reward
#     # action = 2
#     # next_state, reward, done, truncated, info = env.step(action)
#     # rewards += reward
#     if done:
#         env.reset()
#         # break
#     print(rewards)
time.sleep(2)