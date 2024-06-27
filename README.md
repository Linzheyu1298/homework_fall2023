Assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

### Assignment 1: Imitation Learning

**Goal:**
Gain familiarity with imitation learning, including direct behavioral cloning and the DAgger algorithm.

**Key Components:**
1. **Analysis:**
   - Problem of imitation learning within a discrete MDP.
   - Show specific theoretical results regarding expected return and state distributions

2. **Editing Code:**
   - Implement parts of behavioral cloning in provided starter code.
   - Code segments include `forward` and `update` functions in `policies/MLP_policy.py`, `sample_trajectory` function in `infrastructure/utils.py`, and `run_training_loop` function in `scripts/run_hw1.py`.

3. **Behavioral Cloning:**
   - Run behavioral cloning (BC) and report results on two tasks.
   - Experiment with hyperparameters affecting the performance of the BC agent.

4. **DAgger:**
   - Modify runtime parameters to run DAgger.
   - Report results on tasks tested with behavioral cloning, including learning curves.

5. **Extra Credit: SwitchDAgger:**
   - Analyze a variant of the DAgger algorithm.
   - Theoretical analysis to bound the cost of the final policy.

### Assignment 2: Policy Gradients

**Goal:**
Experiment with policy gradient and its variants, including variance reduction techniques like reward-to-go and neural network baselines.

**Key Components:**
1. **Applying Policy Gradients:**
   - Compute the gradient of the expected return without discounting.
   - Verify gradient computations.

2. **Variance Reduction:**
   - Compute and analyze the variance of the policy gradient.
   - Implement return-to-go as an advantage estimator and compare variance.

3. **Generalized Advantage Estimation:**
   - Implement and experiment with neural network baselines.
   - Explore hyperparameters and sample efficiency.

4. **Experiments:**
   - Use policy gradients, neural network baseline, and other techniques to train agents on various tasks.
   - Analyze results and compare different methods.

### Assignment 3: Deep Q-Learning

**Goal:**
Implement and evaluate Q-learning for playing Atari games using convolutional neural network architectures.

**Key Components:**
1. **Basic Q-Learning:**
   - Implement the DQN algorithm, including update for the Q-network and a target network  .

2. **Double Q-Learning:**
   - Avoid overestimation bias using double-Q trick.
   - Implement and evaluate double-Q learning on specified tasks .

3. **Experimenting with Hyperparameters:**
   - Analyze the sensitivity of Q-learning to hyperparameters.
   - Run experiments with different hyperparameter settings and analyze the results .

4. **Continuous Actions with Actor-Critic:**
   - Extend DQN to continuous action spaces using actor-critic methods.
   - Implement stabilization techniques for target values  .

### Assignment 4: Model-Based Reinforcement Learning

**Goal:**
Get experience with model-based reinforcement learning (MBRL) by learning a dynamics function to model state transitions and using predictions for action selection.

**Key Components:**
1. **Analysis:**
   - Analyze the effectiveness of a simple count-based model.
   - Theoretical analysis using the Alternative Simulation Lemma  .

2. **Dynamics Model:**
   - Learn a neural network dynamics model to predict state changes.
   - Train the model using supervised learning techniques  .

3. **Action Selection:**
   - Use the learned dynamics model to select actions through optimization.
   - Implement random-shooting and cross-entropy method (CEM) for action selection  .

4. **On-Policy Data Collection:**
   - Implement iterative data collection using the current policy to improve model performance.
   - Use ensembles of models for better predictions  .

5. **MBPO:**
   - Implement a variant of MBPO to leverage the learned model for generating additional samples.
   - Compare model-free SAC, Dyna-style, and MBPO methods.



**TO BE CONTINUE**
