package com.robocode;

import java.awt.*;
import robocode.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;

public class MyRobotLUT extends AdvancedRobot {
    /**
     * Reinforcement Learning parameters
     */
    final double alpha = 0.7;   // Learning rate (0 if no learning)
    //final double alpha = 0.0;   // No learning
    final double gamma = 0.9;   // Discount factor
    final double epsilon = 0.1; // Exploration factor (0 if always greedy)

    /**
     * {State, action} definitions
     * - State 1 : X position (8)
     * - {1-100, 101-200, 201-300, 301-400, 401-500, 501-600, 601-700, 701-800}
     * - State 2 : Y position (6)
     * - {1-100, 101-200, 201-300, 301-400, 401-500, 501-600}
     * - State 3 : Distance to enemy (4)
     * - {1-250, 251-500, 501-750, 751-1000}
     * - State 4 : Energy (4)
     *   - {0, 1-33, 34-66, 67-100}
     * - Action (5):
     * - {Circle clockwise, circle anticlockwise, advance, retreat, fire}
     */
    public enum stateXPos {x1, x2, x3, x4, x5, x6, x7, x8};
    public enum stateYPos {y1, y2, y3, y4, y5, y6};
    public enum stateDist {d1, d2, d3, d4};
    public enum stateEnergy {e1, e2, e3, e4};
    public enum stateAction {a1, a2, a3, a4, a5};
    public enum mode {scan, action};

    public enum policy {on, off};
    policy runPolicy = policy.off;  // Assume off policy

    /**
     * Lookup Table and battle counters - static so that can retain across rounds
     */
    static boolean randomQ = false;

    static public LUT lut = new LUT(
            stateXPos.values().length,
            stateYPos.values().length,
            stateDist.values().length,
            stateEnergy.values().length,
            stateAction.values().length,
            randomQ);

    static int numRounds = 0;
    static int numWins = 0;
    static boolean startBattle = true;

    // Win rate counter: winRate[0] = # of wins in rounds 1-100, winRate[1] = # of wins in rounds 101-200, etc
    static int[] winRate = new int[10000];

    /**
     * Current and previous states (initial value can be any)
     */
    public stateXPos currStateXPos = stateXPos.x1;
    public stateYPos currStateYPos = stateYPos.y1;
    public stateDist currStateDist = stateDist.d1;
    public stateEnergy currStateEnergy = stateEnergy.e1;
    public stateAction currStateAction = stateAction.a1;

    public stateXPos prevStateXPos;
    public stateYPos prevStateYPos;
    public stateDist prevStateDist;
    public stateEnergy prevStateEnergy;
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
         * Only load the LUT file at start of battle (instead of start of each round).
         */
        if (startBattle) {
            try {
                lut.load(getDataFile("luttest.txt"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        lut.save(getDataFile("luttest.txt"));
        startBattle = false;    // startBattle is static so that will not load LUT again in next round

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
                            turnGunRight(normalizeBearing(turn));
                            fire(3);
                            break;
                        }
                        default: {
                            System.out.println("Invalid action = " + currStateAction);
                        }
                    }

                    // Compute Q based on current rewards and update previous Q
                    double[] x = new double[]{
                            prevStateXPos.ordinal(),
                            prevStateYPos.ordinal(),
                            prevStateDist.ordinal(),
                            prevStateEnergy.ordinal(),
                            prevStateAction.ordinal()};

                    lut.train(x, learnQ(currReward));
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
     * return the greedy action with max Q value.
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
            if (lut.outputFor(x) >= maxQ) {
                maxQ = lut.outputFor(x);
                maxQAction = i;
            }
        }

        return stateAction.values()[maxQAction];
    }

    /**
     * return the greedy action with max Q value.
     * @param s1 position in x-axis (enum index).
     * @param s2 position in y-axis (enum index).
     * @param s3 distance from enemy (enum index).
     * @param s4 energy of my robot (enum index).
     * @return action with max Q value.
     */
    public stateAction greedyAction(int s1, int s2, int s3, int s4) {
        // Quantize state values to LUT indices
        int maxQAction = 0;
        double maxQ = 0.0;
        double[] x = new double[]{s1, s2, s3, s4, 0};

        // Locate the greedy action giving the maximum Q value
        for (int i = 0; i < stateAction.values().length; i++) {
            x[4] = i;
            if (lut.outputFor(x) >= maxQ) {
                maxQ = lut.outputFor(x);
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
    public double learnQ(double reward) {
        stateAction bestAction = greedyAction(
                currStateXPos.ordinal(),
                currStateYPos.ordinal(),
                currStateDist.ordinal(),
                currStateEnergy.ordinal()
        );

        double[] prevSA = new double[]{
                prevStateXPos.ordinal(),
                prevStateYPos.ordinal(),
                prevStateDist.ordinal(),
                prevStateEnergy.ordinal(),
                prevStateAction.ordinal()
        };

        double[] currSA;    // Current state can be either on or off policy

        if (runPolicy == policy.off) {
            currSA = new double[]{
                    currStateXPos.ordinal(),
                    currStateYPos.ordinal(),
                    currStateDist.ordinal(),
                    currStateEnergy.ordinal(),
                    bestAction.ordinal()
            };
        } else {
            currSA = new double[]{
                    currStateXPos.ordinal(),
                    currStateYPos.ordinal(),
                    currStateDist.ordinal(),
                    currStateEnergy.ordinal(),
                    currStateAction.ordinal()
            };
        }

        double prevQ = lut.outputFor(prevSA);
        double currQ = lut.outputFor(currSA);

        return prevQ + alpha * (reward + gamma * currQ - prevQ);
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
     * Quantize X and Y positions to state index
     * X : {0..799} -> {0, 1, 2, 3, 4, 5, 6, 7}
     * Y : {0..599} -> {0, 1, 2, 3, 4, 5}
     * @param pos position of robot obtained from onScannedRobot() event.
     * @return quantized position index.
     */
    public int quantPos(double pos) {
        final int factor = 100; // quantize factor

        return (int) pos / factor;
    }

    /**
     * Quantize distance to enemy to state index
     * Distance to enemy : {0..999} -> {0, 1, 2, 3}
     * @param dist distance from enemy obtained from onScannedRobot() event.
     * @return quantized distance index.
     */
    public int quantDist(double dist) {
        final int factor = 250; // quantize factor

        return (int) dist / factor;
    }

    /**
     * Quantize enemy bearing to state index
     * Enemy bearing : {-180..179} -> {0, 1, 2, 3}
     * @param bearing bearing of enemy from robot obtained from onScannedRobot() event.
     * @return quantized bearing index.
     */
    public int quantBear(double bearing) {
        final int factor = 90; // quantize factor

        return (int) (bearing + 180) / factor;
    }

    /**
     * Quantize self energy to state index
     * Energy : {0..100} -> {0, 1, 2, 3}
     * @param energy energy of my robot obtained from onScannedRobot() event.
     * @return quantized energy index.
     */
    public int quantEnergy(double energy) {
        if (energy == 0) return 0;
        if (energy <= 33) return 1;
        if (energy <= 66) return 2;
        return 3;
    }

    /**
     * Save LUT table to log file
     * @param winArr array of winning count.
     */
    public void saveStats(int[] winArr ) {
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
        prevStateXPos = currStateXPos;
        prevStateYPos = currStateYPos;
        prevStateDist = currStateDist;
        prevStateEnergy = currStateEnergy;
        prevStateAction = currStateAction;

        // Update current state
        currStateXPos = stateXPos.values()[quantPos(xPos)];
        currStateYPos = stateYPos.values()[quantPos(yPos)];
        currStateDist = stateDist.values()[quantDist(dist)];
        currStateEnergy = stateEnergy.values()[quantEnergy(energy)];

        // Switch to action mode
        runMode = mode.action;
    }

    // Hit by enemy robot --> bad instant reward
    public void onHitRobot(HitRobotEvent event) {
        currReward = badInstReward;
        moveAway();
    }

    // Enemy hit by bullet --> good instant reward
    public void onBulletHit(BulletHitEvent event) {
        currReward = goodInstReward;
    }

    // Hit by enemy bullet --> bad instant reward
    public void onHitByBullet(HitByBulletEvent event) {
        currReward = badInstReward;
    }

    // Hit wall --> bad instant reward
    public void onHitWall(HitWallEvent e) {
        currReward = badInstReward;
        moveAway();
    }

    // Win the round --> good terminal reward
    public void onWin(WinEvent event) {
        numWins++;
        currReward = goodTermReward;
        winRate[getRoundNum() / 100]++;

        // Update previous Q before the round ends
        double[] x = new double[]{
                prevStateXPos.ordinal(),
                prevStateYPos.ordinal(),
                prevStateDist.ordinal(),
                prevStateEnergy.ordinal(),
                prevStateAction.ordinal()};

        lut.train(x, learnQ(currReward));
    }

    // Lose the round --> bad terminal reward
    public void onDeath(DeathEvent event) {
        currReward = badTermReward;

        // Update previous Q before the round ends
        double[] x = new double[]{
                prevStateXPos.ordinal(),
                prevStateYPos.ordinal(),
                prevStateDist.ordinal(),
                prevStateEnergy.ordinal(),
                prevStateAction.ordinal()};

        lut.train(x, learnQ(currReward));
    }

    // Round ended --> increase number of rounds for winning statistics calculation
    public void onRoundEnded(RoundEndedEvent e) {
        numRounds++;
    }

    // Battle ended --> save LUT and battle statistics to file
    public void onBattleEnded(BattleEndedEvent e) {
        System.out.println("Win rate = " + numWins + "/" + numRounds);

        // At end of battle, save LUT to file
        lut.save(getDataFile("luttest.txt"));
        saveStats(winRate);
    }
}