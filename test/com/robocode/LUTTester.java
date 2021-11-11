package com.robocode;

import org.junit.Assert;
import org.junit.Test;
import robocode.BulletHitEvent;
import robocode.HitByBulletEvent;

import java.io.FileWriter;
import java.util.Arrays;

/** Test Driven Development (TDD) approach is used where the software is made as modular as possible via Java methods.
 * The test cases of each method are written in JUnit that drives the actual coding of the method.
 * This ensures each method is tested thoroughly in terms of expected functionality and code coverage.
 */
public class LUTTester {
    // Test LUT constructor
    @Test
    public void testLUT() {
        LUT lut = new LUT(8, 6, 4, 4, 5, true);

        double [] x = {0, 0, 0, 0, 0};
        double [] y = {7, 5, 3, 3, 4};
        Assert.assertEquals(0, lut.getAccessCnt(x));
        Assert.assertEquals(0, lut.getAccessCnt(y));
        Assert.assertTrue("Initial Q should be between 0 and 1", lut.outputFor(y) >= 0.0 && lut.outputFor(y) <= 1.0);
    }

    // Test train() and outputFor()
    @Test
    public void testTrain() {
        LUT lut = new LUT(8, 6, 4, 4, 5, false);

        double [] x = {0, 1, 2, 3, 3};

        Assert.assertEquals(0.0, lut.outputFor(x), 0.005);
        lut.train(x, 2.43);
        Assert.assertEquals(2.43, lut.outputFor(x), 0.005);
    }

    @Test
    public void testExploreAction() {
        MyRobotLUT robot = new MyRobotLUT();

        MyRobotLUT.stateAction rndAction = robot.exploreAction();
        //System.out.println("Random action = " + rndAction);

        Assert.assertTrue("Explored action should be {a1, a2, a3, a4, a5}",
                Arrays.asList(MyRobotLUT.stateAction.values()).contains(rndAction));
    }

    @Test
    public void testQuantPos() {
        MyRobotLUT robot = new MyRobotLUT();

        Assert.assertEquals(0, robot.quantPos(0));
        Assert.assertEquals(0, robot.quantPos(50));
        Assert.assertEquals(1, robot.quantPos(100));
        Assert.assertEquals(1, robot.quantPos(199));
        Assert.assertEquals(2, robot.quantPos(250));
        Assert.assertEquals(7, robot.quantPos(799));
    }

    @Test
    public void testQuantDist() {
        MyRobotLUT robot = new MyRobotLUT();

        Assert.assertEquals(0, robot.quantDist(0));
        Assert.assertEquals(0, robot.quantDist(249));
        Assert.assertEquals(1, robot.quantDist(250));
        Assert.assertEquals(1, robot.quantDist(300));
        Assert.assertEquals(2, robot.quantDist(501));
        Assert.assertEquals(3, robot.quantDist(799));
        Assert.assertEquals(3, robot.quantDist(999));
    }

    @Test
    public void testQuantBear() {
        MyRobotLUT robot = new MyRobotLUT();

        Assert.assertEquals(0, robot.quantBear(-180));
        Assert.assertEquals(0, robot.quantBear(-130));
        Assert.assertEquals(1, robot.quantBear(-90));
        Assert.assertEquals(1, robot.quantBear(-30));
        Assert.assertEquals(2, robot.quantBear(0));
        Assert.assertEquals(2, robot.quantBear(89));
        Assert.assertEquals(3, robot.quantBear(119));
    }

    @Test
    public void testOnBulletHit() {
        MyRobotLUT robot = new MyRobotLUT();
        BulletHitEvent e1 = null;
        HitByBulletEvent e2 = null;

        robot.currReward = 0;
        robot.onBulletHit(e1);
        Assert.assertEquals(1.0, robot.currReward, 0.005);
        robot.onHitByBullet(e2);
        Assert.assertEquals(0.75, robot.currReward, 0.005);
        robot.onHitByBullet(e2);
        robot.onHitByBullet(e2);
        Assert.assertEquals(0.25, robot.currReward, 0.005);
        robot.onHitByBullet(e2);
        robot.onHitByBullet(e2);
        Assert.assertEquals(-0.25, robot.currReward, 0.005);
    }

    @Test
    public void testGreedyAction() {
        MyRobotLUT robot = new MyRobotLUT();

        double [] a1 = {0, 1, 2, 3, 0};
        double [] a2 = {0, 1, 2, 3, 1};
        double [] a3 = {0, 1, 2, 3, 2};
        double [] a4 = {0, 1, 2, 3, 3};
        double [] a5 = {0, 1, 2, 3, 4};

        robot.lut.train(a1, -0.1);
        robot.lut.train(a2, 0.7);   // Max action
        robot.lut.train(a3, 0.3);
        robot.lut.train(a4, 0.1);
        robot.lut.train(a5, -0.4);

        Assert.assertEquals(MyRobotLUT.stateAction.a2, robot.greedyAction(0, 1, 2, 3));

        robot.lut.train(a1, 1);
        robot.lut.train(a2, 2);
        robot.lut.train(a3, 3);
        robot.lut.train(a4, 4);
        robot.lut.train(a5, 5); // Max action

        Assert.assertEquals(MyRobotLUT.stateAction.a5, robot.greedyAction(0, 1, 2, 3));
        Assert.assertEquals(MyRobotLUT.stateAction.a5, robot.greedyAction(1.0, 150, 700, 170));
    }

    @Test
    public void testLearnQ() {
        //LUT lut = new LUT(8, 6, 4, 4, 5, true);
        double [] c1 = {0, 1, 2, 3, 0}; // Current state
        double [] c2 = {0, 1, 2, 3, 1};
        double [] c3 = {0, 1, 2, 3, 2};
        double [] c4 = {0, 1, 2, 3, 3};
        double [] c5 = {0, 1, 2, 3, 4};
        double [] p = {6, 4, 1, 2, 3};  // Previous state

        MyRobotLUT robot = new MyRobotLUT();

        robot.lut.train(c1, 0.2);
        robot.lut.train(c2, 0.4);
        robot.lut.train(c3, 0.6);
        robot.lut.train(c4, 0.8);
        robot.lut.train(c5, 1.0);   // Max action
        robot.lut.train(p, 0.7);

        robot.currStateXPos = MyRobotLUT.stateXPos.x1;
        robot.currStateYPos = MyRobotLUT.stateYPos.y2;
        robot.currStateDist = MyRobotLUT.stateDist.d3;
        robot.currStateBear = MyRobotLUT.stateBear.b4;

        robot.prevStateXPos = MyRobotLUT.stateXPos.x7;
        robot.prevStateYPos = MyRobotLUT.stateYPos.y5;
        robot.prevStateDist = MyRobotLUT.stateDist.d2;
        robot.prevStateBear = MyRobotLUT.stateBear.b3;
        robot.prevStateAction = MyRobotLUT.stateAction.a4;

        Assert.assertEquals(1.68, robot.learnQ(1.2), 0.005);
        // prevQ + alpha * (reward + gamma * currQ - prevQ)
        // 0.7   + 0.7   x (1.2    + 0.9   x 1.0   - 0.7) = 1.68
    }
}