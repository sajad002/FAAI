# FAAI
Fundamentals and Applications of Artificial Intelligence (reinforcement learning in Frozen lake environment using gym library) - spring-2023
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Project Repository</title>
</head>
<body>
    <h1>Project Repository</h1>
    
    <h2>Introduction</h2>
    <p>In this project, we aim to implement three algorithms:</p>
    <ul>
        <li>Policy Iteration</li>
        <li>Monte Carlo (prediction – first visit)</li>
        <li>Monte Carlo (prediction – every visit)</li>
    </ul>
    
    <p>Using the OpenAI Gym library in the Frozen Lake environment, we investigate and analyze the impact of reward, discount factor (γ), and determinism in various environments. To better understand, some methods for displaying policy and state values have been pre-implemented, and you can use them.</p>
    
    <h2>The Frozen Lake Environment</h2>
    <p>The environment consists of navigating a frozen lake from Start(S) to Goal(G) without falling into any Hole(H) by stepping on Frozen(F) tiles. If the agent falls into a hole (H), it must restart from the start position (S). Due to the slippery nature of the frozen lake, the agent may not always move in the intended direction. The agent can move in four directions: left, down, right, and up. If the agent moves in a direction that crosses the environment boundaries, its position remains unchanged. Each of the states S, G, and F has a specific reward associated with it. (For more information, you can refer to the main page of the environment).</p>
    
    <h2>Implementation of Algorithms</h2>
    <h3>1. Policy Iteration</h3>
    <p>The initial algorithm to be implemented is Policy Iteration. This algorithm, given the environment and complete knowledge of the environment, can converge to an optimal policy starting from a random policy. The algorithm consists of two main steps:</p>
    <ol>
        <li>Policy Evaluation</li>
        <li>Policy Improvement</li>
    </ol>
    
    <p>In the first step, using the Bellman equation and the desired policy, the algorithm calculates the value of each state.</p>
    
    <pre><code>// Implementation of the policy_iteration method
def policy_iteration(env, custom_map, max_ittr, theta, discount_factor):
    # Your code here
    pass</code></pre>
    
    <h3>2. First-Visit Monte Carlo Prediction</h3>
    <p>The second algorithm to be implemented is First-Visit Monte Carlo Prediction. The value of a state is equal to the estimation of the expected return starting from that state. A straightforward way to estimate the value of a state from experience is to take the average of the observed returns after visiting that state. If the returns tend towards infinity, the average should converge to the expected value. This idea forms the basis of all Monte Carlo methods.</p>
    
    <pre><code>// Implementation of the first_visit_mc_prediction method
def first_visit_mc_prediction(env, policy, num_episodes, gamma):
    # Your code here
    pass</code></pre>
    
    <h3>3. Every-Visit Monte Carlo Prediction</h3>
    <p>The third algorithm to be implemented is Every-Visit Monte Carlo Prediction. This algorithm is similar to the First-Visit Monte Carlo algorithm, with the difference that the first arrival to a state is not taken as the reference for calculating the value of that state; instead, all visits to that state are considered.</p>
    
    <pre><code>// Implementation of the every_visit_mc_prediction method
def every_visit_mc_prediction(env
