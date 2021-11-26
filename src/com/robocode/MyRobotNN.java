package com.robocode;

import robocode.*;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;

/**
 * Robot using NN to approximate the Q-learning function instead of LUT
 */
public class MyRobotNN extends AdvancedRobot {
    /**
     * Reinforcement Learning parameters
     */
    final double alpha = 0.2;   // Learning rate (0 if no learning)
    //final double alpha = 0.0;   // No learning
    final double gamma = 0.1;   // Discount factor
    final double epsilon = 0.1; // Exploration factor (0 if always greedy)

    /**
     * {State, action} definitions
     * - State 1 : X position {0-800} -> {0.00-8.00}
     * - State 2 : Y position {0-600} -> {0.00-6.00}
     * - State 3 : Distance to enemy {0-1000} -> {0.00-10.00}
     * - State 4 : Energy {0-100} -> {0.0-10.00}
     * - Action : {Circle clockwise, circle anticlockwise, advance, retreat, fire}
     */
    public enum stateAction {a1, a2, a3, a4, a5};
    public enum mode {scan, action};
    public enum policy {on, off};
    policy runPolicy = policy.off;  // Assume off policy

    /**
     * Neural net and battle counters - static so that can retain across rounds
     */
    static int numInputs = 5;
    static int numHidden = 5;
    static double learningRate = 0.2;
    static double momentumTerm = 0.5;
    static public NeuralNet nn = new NeuralNet(
             NeuralNet.ActFnType.BIPOLAR, numInputs, numHidden, learningRate, momentumTerm, -1, 1);

    static int numRounds = 0;
    static int numWins = 0;

    // Win rate counter: winRate[0] = # of wins in rounds 1-100, winRate[1] = # of wins in rounds 101-200, etc
    static int[] winRate = new int[10000];

    /**
     * Create replay memory to train more than 1 sample at a time step
     */
    static int memSize = 10;
    static ReplayMemory<Experience> replayMemory = new ReplayMemory<>(memSize);

    /**
     * Current and previous states (initial value can be any)
     */
    public State currState = new State(0.0, 0.0, 0.0, 0.0);
    public stateAction currStateAction = stateAction.a1;

    public State prevState = new State(0.0, 0.0, 0.0, 0.0);
    public stateAction prevStateAction;

    public mode runMode = mode.scan;

    /**
     * Good/bad instant/terminal reward values
     */
    public final double badInstReward = -0.25;
    public final double goodInstReward = 1.0;
    public final double badTermReward = -0.5;
    public final double goodTermReward = 2.0;
    public double currReward = 0.0;
    public double accumReward = 0.0;

    // rewardRate[0] = accum reward in round 1, rewardRate[1] = accum reward in round 2, etc
    static double[] rewardRate = new double[10000];

    /**
     * State values (non-quantized) obtained from onScannedRobot()
     */
    double xPos = 0.0;
    double yPos = 0.0;
    double dist = 0.0;
    double bearing = 0.0;
    double energy = 0.0;

    int circleDir = 1;  // Clockwise = 1, anti-clockwise = -1

    public void run() {
        /**
         * A battle contains multiple rounds.
         * run() will be called at start of each round.
         * Initialize NN at start of battle (i.e. round 0)
         */

        if (getRoundNum() == 0) {
            nn.initializeWeights();
            nn.zeroWeights();
        }

        // Color my robot
        setColors(Color.blue, Color.red, Color.orange, Color.black, Color.green);

        while (true) {
            switch (runMode) {
                case scan: {
                    currReward = 0;
                    // Perform enemy scan, control will go to onScannedRobot()
                    turnRadarRight(90);
                    break;
                }
                case action: {
                    // Explore or exploit depending on epsilon
                    if (Math.random() <= epsilon) {
                        currStateAction = exploreAction();
                    }
                    else
                        currStateAction = greedyAction(xPos, yPos, dist, energy);

                    /**
                     * These are the macro actions performed by the robot.
                     * Performance may differ depending on how the actions are set.
                     */
                    switch (currStateAction) {
                        case a1: {  // circle clockwise
                            setTurnRight(bearing + 90);
                            setAhead(50 * circleDir);
                            execute();
                            break;
                        }
                        case a2: {  // circle anti-clockwise
                            setTurnLeft(bearing + 90);
                            setAhead(50 * circleDir);
                            execute();
                            break;
                        }
                        case a3: {  // advance
                            setTurnRight(bearing);
                            setAhead(100);
                            execute();
                            break;
                        }
                        case a4: {  // retreat
                            setTurnRight(bearing + 180);
                            setAhead(100);
                            execute();
                            break;
                        }
                        case a5: {  // fire
                            double turn = getHeading() - getGunHeading() + bearing;
                            turnGunRight(normalizeBearing(turn)); // Turn gun towards enemy
                            fire(3);
                            break;
                        }
                        default: {
                            System.out.println("Invalid action = " + currStateAction);
                        }
                    }

                    // Compute Q based on current rewards and update previous Q
                    updatePrevQ();
                    /*
                    double[] x = new double[]{
                            prevState.getXPos(),
                            prevState.getYPos(),
                            prevState.getDist(),
                            prevState.getEnergy(),
                            prevStateAction.ordinal()};

                    replayMemory.add(new Experience(prevState, prevStateAction, currReward, currState));
                    nn.train(x, learnQ(prevState, prevStateAction, currReward, currState));
                    */
                    runMode = mode.scan;    // Switch to scan mode
                    break;
                }
                default: {
                    System.out.println("Invalid runMode = " + runMode);
                }
            }
        }
    }

    /**
     * Normalize the bearing of enemy from robot to the range {-180, 180}.
     * @param angle The input bearing in degrees.
     * @return normalized bearing in the range {-180, 180}.
     */
    double normalizeBearing(double angle) {
        while (angle >  180) angle -= 360;
        while (angle < -180) angle += 360;
        return angle;
    }

    /**
     * Return a random action from current state.
     * @return random action.
     */
    public stateAction exploreAction() {
        int x = new Random().nextInt(stateAction.values().length);
        return stateAction.values()[x];
    }

    /**
     * Return the greedy action with max Q value.
     * @param xPos position in x-axis (actual value).
     * @param yPos position in y-axis (actual value).
     * @param dist distance from enemy (actual value).
     * @param energy energy of my robot (actual value).
     * @return action with max Q value.
     */
    public stateAction greedyAction(double xPos, double yPos, double dist, double energy) {
        // Quantize state values to LUT indices
        int maxQAction = 0;
        double maxQ = 0.0;
        double[] x = new double[]{quantPos(xPos), quantPos(yPos), quantDist(dist), quantEnergy(energy), 0};

        // Locate the greedy action giving the maximum Q value
        for (int i = 0; i < stateAction.values().length; i++) {
            x[4] = i;
            if (nn.outputFor(x) >= maxQ) {
                maxQ = nn.outputFor(x);
                maxQAction = i;
            }
        }

        return stateAction.values()[maxQAction];
    }

    /**
     * return the new Q value based on TD learning.
     * @param reward reward value.
     * @return learned Q value.
     */
    public double learnQ(State prevState, MyRobotNN.stateAction prevAction, double reward, State currState) {
        stateAction bestAction = greedyAction(
                currState.getXPos(),
                currState.getYPos(),
                currState.getDist(),
                currState.getEnergy()
        );

        double[] prevSA = new double[]{
                prevState.getXPos(),
                prevState.getYPos(),
                prevState.getDist(),
                prevState.getEnergy(),
                prevAction.ordinal()
        };

        double[] currSA;    // Current state can be either on or off policy

        if (runPolicy == policy.off) {
            currSA = new double[]{
                    currState.getXPos(),
                    currState.getYPos(),
                    currState.getDist(),
                    currState.getEnergy(),
                    bestAction.ordinal()
            };
        } else {
            currSA = new double[]{
                    currState.getXPos(),
                    currState.getYPos(),
                    currState.getDist(),
                    currState.getEnergy(),
                    currStateAction.ordinal()
            };
        }

        double prevQ = nn.outputFor(prevSA);
        double currQ = nn.outputFor(currSA);

        return prevQ + alpha * (reward + gamma * currQ - prevQ);
    }

    /**
     * Update Q value of the previous state using learned Q value.
     */
    public void updatePrevQ() {
        double[] x = new double[]{
                prevState.getXPos(),
                prevState.getYPos(),
                prevState.getDist(),
                prevState.getEnergy(),
                prevStateAction.ordinal()};

        replayMemory.add(new Experience(prevState, prevStateAction, currReward, currState));
        replayTrain(x);
        //nn.train(x, learnQ(prevState, prevStateAction, currReward, currState));
    }

    /**
     * Train NN using multiple vectors saved in replayMemory
     */
    public void replayTrain(double[] x) {
        //replayMemory.add(new Experience(prevState, prevStateAction, currReward, currState));
        //replayTrain(x);
        int trainSize = Math.min(replayMemory.sizeOf(), memSize);
        Object[] vector = replayMemory.sample(trainSize);

        for (Object e: vector) {
            Experience exp = (Experience) e;
            nn.train(x, learnQ(exp.prevState,
                    exp.prevAction,
                    exp.currReward,
                    exp.currState));
        }
    }

    /**
     * Move away from the wall when hit wall
     */
    public void moveAway() {
        switch (currStateAction) {
            case a1:
            case a2: {
                circleDir = circleDir * -1;
                break;
            }

            case a3:
            case a4:
            case a5: {
                back(20);
                setTurnRight(30);
                setBack(50);
                execute();
                break;
            }
        }
    }

    /**
     * Quantize X and Y position to state value for NN input
     * X : {0..800} -> {0.00, 0.01, 0.02, ..., 7.99, 8.00}
     * Y : {0..600} -> {0.00, 0.01, 0.02, ..., 5.99, 6.00}
     * @param pos position of robot obtained from onScannedRobot() event.
     * @return quantized position value.
     */
    public double quantPos(double pos) {
        final int factor = 100; // quantize factor

        return pos / factor;
    }

    /**
     * Quantize distance to enemy to state value for NN input
     * Distance to enemy : {0..1000} -> {0.00, 0.01, 0.02, ..., 9.99, 10.00}
     * @param dist distance from enemy obtained from onScannedRobot() event.
     * @return quantized distance value.
     */
    public double quantDist(double dist) {
        final int factor = 100; // quantize factor

        return dist / factor;
    }

    /**
     * Quantize self energy to state value for NN input
     * Energy : {0..100} -> {0.0, 0.1, 0.2, ..., 9.99, 10.00}
     * @param energy energy of my robot obtained from onScannedRobot() event.
     * @return quantized energy value.
     */
    public double quantEnergy(double energy) {
        final int factor = 10; // quantize factor
        return energy / factor;
    }

    /**
     * Save winning rate table to log file
     * @param winArr array of winning count.
     */
    public void saveStats(int[] winArr) {
        try {
            File winRatesFile = getDataFile("WinRate.txt");
            PrintStream out = new PrintStream(new RobocodeFileOutputStream(winRatesFile));
            out.format("Win rate, %d/%d = %d\n", numWins, numRounds, numWins*100/numRounds);
            out.format("Every 100 Rounds, Wins,\n");
            for (int i = 0; i < (getRoundNum() + 1) / 100; i++) {
                out.format("%d, %d,\n", i + 1, winArr[i]);
            }
            out.close();
        } catch (IOException exception) {
            exception.printStackTrace();
        }
    }

    /**
     * Save accum reward table to log file
     * @param rewardArr array of winning count.
     */
    public void saveReward(double[] rewardArr ) {
        try {
            File rewardFile = getDataFile("Reward.txt");
            PrintStream out = new PrintStream(new RobocodeFileOutputStream(rewardFile));
            out.format("Round #, Accum reward,\n");
            for (int i = 0; i < getRoundNum() + 1; i++) {
                out.format("%d, %f,\n", i + 1, rewardArr[i]);
            }
            out.close();
        } catch (IOException exception) {
            exception.printStackTrace();
        }
    }

    /**
     * Overridden functions OnXXXX for robocode events
     */
    // Update current state based on scanned values
    public void onScannedRobot(ScannedRobotEvent e) {
        xPos = getX();
        yPos = getY();
        dist = e.getDistance();
        bearing = e.getBearing();
        energy = getEnergy();

        // Update previous state
        prevState.copyState(currState);
        prevStateAction = currStateAction;

        // Update current state
        currState.setXPos(quantPos(xPos));
        currState.setYPos(quantPos(yPos));
        currState.setDist(quantDist(dist));
        currState.setEnergy(quantEnergy(energy));

        // Switch to action mode
        runMode = mode.action;
    }

    // Hit by enemy robot --> bad instant reward
    public void onHitRobot(HitRobotEvent event) {
        currReward = badInstReward;
        accumReward += currReward;
        moveAway();
    }

    // Enemy hit by bullet --> good instant reward
    public void onBulletHit(BulletHitEvent event) {
        currReward = goodInstReward;
        accumReward += currReward;
    }

    // Hit by enemy bullet --> bad instant reward
    public void onHitByBullet(HitByBulletEvent event) {
        currReward = badInstReward;
        accumReward += currReward;
    }

    // Hit wall --> bad instant reward
    public void onHitWall(HitWallEvent e) {
        currReward = badInstReward;
        accumReward += currReward;
        moveAway();
    }

    // Win the round --> good terminal reward
    public void onWin(WinEvent event) {
        numWins++;
        currReward = goodTermReward;
        accumReward += currReward;
        winRate[getRoundNum() / 100]++;

        // Update previous Q before the round ends
        updatePrevQ();
        /*
        double[] x = new double[]{
                prevState.getXPos(),
                prevState.getYPos(),
                prevState.getDist(),
                prevState.getEnergy(),
                prevStateAction.ordinal()};

        nn.train(x, learnQ(prevState, prevStateAction, currReward, currState));

         */
    }

    // Lose the round --> bad terminal reward
    public void onDeath(DeathEvent event) {
        currReward = badTermReward;
        accumReward += currReward;

        // Update previous Q before the round ends
        updatePrevQ();
        /*
        double[] x = new double[]{
                prevState.getXPos(),
                prevState.getYPos(),
                prevState.getDist(),
                prevState.getEnergy(),
                prevStateAction.ordinal()};

        nn.train(x, learnQ(prevState, prevStateAction, currReward, currState));

         */
    }

    // Round ended --> reset reward stats and increase number of rounds for winning statistics calculation
    public void onRoundEnded(RoundEndedEvent e) {
        rewardRate[numRounds] = accumReward;
        accumReward = 0; // reset accum reward for next round
        numRounds++;
    }

    // Battle ended --> save NN weights and battle statistics to file
    public void onBattleEnded(BattleEndedEvent e) {
        System.out.println("Win rate = " + numWins + "/" + numRounds);

        nn.save(getDataFile("NN_weights.txt")); // Save NN weights
        saveStats(winRate);     // Save winning rate
        saveReward(rewardRate); // Save reward rate
    }
}