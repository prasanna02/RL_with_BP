package com.robocode;

import static org.junit.Assert.*;
import org.junit.Test;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;

/** Test Driven Development (TDD) approach is used where the software is made as modular as possible via Java methods.
 * The test cases of each method are written in JUnit that drives the actual coding of the method.
 * This ensures each method is tested thoroughly in terms of expected functionality and code coverage.
 */
public class RobotNNTester {
    // Test State constructor and get/set methods
    @Test
    public void testState() {
        State s = new State(2.3, 3.8, 9.8, 8.1);

        System.out.println(s);
        assertEquals(2.3, s.getXPos(), 0.005);
        assertEquals(3.8, s.getYPos(), 0.005);
        assertEquals(9.8, s.getDist(), 0.005);
        assertEquals(8.1, s.getEnergy(), 0.005);

        s.setXPos(2.4);
        s.setYPos(3.9);
        s.setDist(9.9);
        s.setEnergy(8.2);

        assertEquals(2.4, s.getXPos(), 0.005);
        assertEquals(3.9, s.getYPos(), 0.005);
        assertEquals(9.9, s.getDist(), 0.005);
        assertEquals(8.2, s.getEnergy(), 0.005);

        State s1 = new State(0, 0, 0, 0);

        s1.copyState(s);
        assertEquals(2.4, s1.getXPos(), 0.005);
        assertEquals(3.9, s1.getYPos(), 0.005);
        assertEquals(9.9, s1.getDist(), 0.005);
        assertEquals(8.2, s1.getEnergy(), 0.005);

        s.setXPos(2.5);
        assertEquals(2.4, s1.getXPos(), 0.005);
        s1.setXPos(2.6);
        assertEquals(2.6, s1.getXPos(), 0.005);
    }

    // Test loading weights from file and saving weights to file
    @Test
    public void testLoadSave() {
        NeuralNet nn = new NeuralNet(
                NeuralNet.ActFnType.BIPOLAR, 5, 5, 0.2, 0.5, -1, 1);

        File file1 = new File("NN_weights.txt");
        File file2 = new File("NN_weights_2.txt");

        // Read file 1 into NN
        try {
            nn.load(file1);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Save NN weights into file 2
        FileWriter weightFile = nn.createFile("NN_weights_2.txt");
        nn.save(weightFile);
        nn.closeFile(weightFile);

        // Files 1 & 2 should be equal in contents
        try {
            byte[] file1Bytes = Files.readAllBytes(file1.toPath());
            byte[] file2Bytes = Files.readAllBytes(file1.toPath());

            assertTrue(Arrays.equals(file1Bytes, file2Bytes));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Test replayMemory operations on saving Experience
    @Test
    public void testReplayMemory() {
        State s1 = new State(1,2,3,4);
        MyRobotNN.stateAction a1 = MyRobotNN.stateAction.a1;
        double r1 = 0.1;
        State s2 = new State(1.1,2.1,3.1,4.1);

        State s3 = new State(2,3,4,5);
        MyRobotNN.stateAction a2 = MyRobotNN.stateAction.a2;
        double r2 = 0.2;
        State s4 = new State(2.1,3.1,4.1,5.1);

        State s5 = new State(3,4,5,6);
        MyRobotNN.stateAction a3 = MyRobotNN.stateAction.a3;
        double r3 = 0.3;
        State s6 = new State(3.1,4.1,5.1,6.1);

        State s7 = new State(4,5,6,7);
        MyRobotNN.stateAction a4 = MyRobotNN.stateAction.a4;
        double r4 = 0.4;
        State s8 = new State(4.1,5.1,6.1,7.1);

        State s9 = new State(5,6,7,8);
        MyRobotNN.stateAction a5 = MyRobotNN.stateAction.a5;
        double r5 = 0.5;
        State s10 = new State(5.1,6.1,7.1,8.1);

        Experience exp1 = new Experience(s1, a1, r1, s2);
        Experience exp2 = new Experience(s3, a2, r2, s4);
        Experience exp3 = new Experience(s5, a3, r3, s6);
        Experience exp4 = new Experience(s7, a4, r4, s8);
        Experience exp5 = new Experience(s9, a5, r5, s10);

        // Initial size should be 0
        ReplayMemory<Experience> replayMemory = new ReplayMemory<>(3);
        assertEquals(0, replayMemory.sizeOf());

        // Size should be 2 after adding two elements
        replayMemory.add(exp1);
        replayMemory.add(exp2);
        assertEquals(2, replayMemory.sizeOf());

        // Get a sample size of 1 and verify
        Object[] vector = replayMemory.sample(1);
        assertEquals(0.2, ((Experience) vector[0]).currReward, 0.005);

        // Size should be 3 (max) after adding 3 more elements
        replayMemory.add(exp3);
        replayMemory.add(exp4);
        replayMemory.add(exp5);
        assertEquals(3, replayMemory.sizeOf());

        // Get a sample size of 2 (should be last 2 added) and verify
        vector = replayMemory.sample(2);
        assertEquals(0.4, ((Experience) vector[0]).currReward, 0.005);
        assertEquals(0.5, ((Experience) vector[1]).currReward, 0.005);
        assertEquals(7, ((Experience) vector[0]).prevState.getEnergy(), 0.005);
        assertEquals(5.1, ((Experience) vector[1]).currState.getXPos(), 0.005);
    }
}