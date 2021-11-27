# Robocode Reinforcement Learning (RL) using Neural Network (NN) to approximate the Q-value function

### In part 2 of assignment, RL is done via lookup table to represent the Q-value function.  State space reduction is used to model the lookup table with a reasonable length.  In part 3 of assignment, NN will be used to train the Q-value function to eliminate the need to store a massive lookup table in memory.

### The following Java files are included in this project:

- **CommonInterface.java** and **NeuralNetInterface.java**

  Interface files to facilitate the transition from LUT to NN.  CommonInterface.java contains the methods that are common to both RL and NN.
- **State.java**

  Model the state of robot in {state, action} pair
  - State 1 : X position {0-800} -> {0.00-8.00}
  - State 2 : Y position {0-600} -> {0.00-6.00}
  - State 3 : Distance to enemy {0-1000} -> {0.00-10.00}
  - State 4 : Energy {0-100} -> {0.0-10.00}

- **Experience.java**

  Model a training vector stored Replay Memory to facilitate NN training
  - Previous state
  - Previous action
  - Current reward
  - Current state

- **MyRobotNN.java**

  Implementation of the robot in Robocode using NN training.
- **CircularQueue.java** and **ReplayMemory**

  Provided code to implement Replay Memory
- **NeuralNet.java**

  NN framework used for robot training in MyRobotNN.java
- **LUTTrain.java**

  A standalone application to use the LUT file from Assignment Part 2 as training data for NN
- **RobotNNTester.java**

  Junit test cases for Test Driven Development (TDD)
