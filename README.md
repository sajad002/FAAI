# Reinforcement Learning Project

In this project, we aim to implement three algorithms:
- Policy Iteration
- Monte Carlo (Prediction – First Visit)
- Monte Carlo (Prediction – Every Visit)

Using the OpenAI Gym library in the Frozen Lake environment, we investigate and analyze the impact of reward, discount factor (γ), and determinism in various environments. Several methods for displaying policy values and state values have been implemented for better understanding, and you can utilize them in your project.

## Frozen Lake Environment

The environment consists of navigating through a frozen lake from Start (S) to Goal (G) without falling into any Hole (H) by walking on the Frozen (F) lake. If the agent falls into a hole (H), it must start the traversal again from the starting cell (S). Due to the slippery nature of the frozen lake, the agent may not always move in the intended direction. The agent can move in four directions: left, down, right, and up. If the agent makes a move that crosses the environment boundary, the agent's position does not change. The rewards for reaching each of the states S, G, and F are specified based on the problem under consideration.

For more details, you can refer to the official documentation.

## Section 1 – Implementation of Algorithms

To execute the project, start by installing the Gym library. The implemented code is based on Gym version 0.26.2, but you can use other versions of the Gym library or Gymnasium if needed. Note that if you use a different version, you may need to modify specific parts of the implemented code.

Download the code provided in the repository and add it to your project. As you can see, all the necessary methods for interacting with the Gym library, policy display methods, etc., have already been implemented. You can use these methods to carry out your project.

The initial algorithm to implement is Policy Iteration. This algorithm, with a complete view of the environment, starts with a random policy and iteratively reaches the optimal policy. The algorithm has two main steps: Policy Evaluation and Policy Improvement.

In the first part of this algorithm, using the Bellman equation and the specified policy, it calculates the value of each state.

In the second part, by selecting an action that maximizes the value of that state, it strives to improve its policy. The pseudocode is as follows:

```python
# Implement the Policy Iteration algorithm
# def policy_iteration(environment):
#     ...
