package com.robocode;

import robocode.*;

import java.io.*;

/**
 * Lookup table for Robocode Reinforcement Learning
 * THis LUT table maps {states, action} to Q value
 * State-action representation (4 states + 1 action):
 * - State 1 : X position (8)
 *   - {1-100, 101-200, 201-300, 301-400, 401-500, 501-600, 601-700, 701-800}
 * - State 2 : Y position (6)
 *   - {1-100, 101-200, 201-300, 301-400, 401-500, 501-600}
 * - State 3 : Distance to enemy (4)
 *   - {1-250, 251-500, 501-750, 751-1000}
 * - State 4 : Bearing (4)
 *   - {1-90, 91-180, 181-270, 271-360}
 * - Action (5):
 *   - {Circle clockwise, circle anticlockwise, advance, retreat, fire}
 * Total number of entries in LUT = 8 x 6 x 4 x 4 x 5 = 3840
 */

public class LUT implements CommonInterface {
    private double [][][][][] lut;      // Q value
    private int [][][][][] accessCnt;   // Access count
    private int state1Dim;              // Dimension for state 1
    private int state2Dim;              // Dimension for state 2
    private int state3Dim;              // Dimension for state 3
    private int state4Dim;              // Dimension for state 4
    private int actionDim;              // Dimension for action
    private boolean randomQ;  // Random or zero initial Q

    // Constructor
    public LUT (int state1Dim, int state2Dim, int state3Dim, int state4Dim, int actionDim, boolean randomQ) {
        this.state1Dim = state1Dim;
        this.state2Dim = state2Dim;
        this.state3Dim = state3Dim;
        this.state4Dim = state4Dim;
        this.actionDim = actionDim;
        this.randomQ = randomQ;

        lut = new double [state1Dim][state2Dim][state3Dim][state4Dim][actionDim];
        accessCnt = new int [state1Dim][state2Dim][state3Dim][state4Dim][actionDim];

        this.initLUT();
    }

    /**
     * Initialize the lut array to random number between {0, 1}.
     * Initialize the accessCnt array to 0.
     */
    public void initLUT () {
        for (int a = 0; a < state1Dim; a++) {
            for (int b = 0; b < state2Dim; b++) {
                for (int c = 0; c < state3Dim; c++) {
                    for (int d = 0; d < state4Dim; d++) {
                        for (int e = 0; e < actionDim; e++) {
                            if (randomQ)
                                lut[a][b][c][d][e] = Math.random();
                            else
                                lut[a][b][c][d][e] = 0.0;
                            accessCnt[a][b][c][d][e] = 0;
                        }

                    }
                }
            }
        }
    }

    /**
     * Return access count of a {state, action} entry.
     * @param x The {state, action} vector.
     * @return access count of the corresponding {state, action} LUT entry.
     */
    public int getAccessCnt (double [] x) {
        return accessCnt[(int)x[0]][(int)x[1]][(int)x[2]][(int)x[3]][(int)x[4]];
    }

    /**
     * Return Q-value of a {state, action} entry, i.e. Q(s, a).
     * @param x The {state, action} vector.
     * @return Q-value of the corresponding {state, action} LUT entry.
     */
    @Override
    public double outputFor (double [] x) {
        return lut[(int)x[0]][(int)x[1]][(int)x[2]][(int)x[3]][(int)x[4]];
    }

    /**
     * Write the current LUT to output file.
     * @param filename Target output file.
    */
    @Override
    public void save(File filename) {
        PrintStream w = null;
        try {
            w = new PrintStream(new RobocodeFileOutputStream(filename));
            for (int a = 0; a < state1Dim; a++) {
                for (int b = 0; b < state2Dim; b++) {
                    for (int c = 0; c < state3Dim; c++) {
                        for (int d = 0; d < state4Dim; d++) {
                            for (int e = 0; e < actionDim; e++) {
                                w.println(a + "" + b + "" + c + "" + d + "" + e + "\t" +
                                        lut[a][b][c][d][e] + "\t" +
                                        accessCnt[a][b][c][d][e]);
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            w.flush();
            w.close();
        }
    }

    /**
     * Read the saved LUT file into the LUT.
     * @param filename Saved LUT filename.
     */
    @Override
    public void load(File filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line = reader.readLine();
        try {
            int zz = 0;
            while (line != null) {
                String splitLine[] = line.split("\t");
                //System.out.println(splitLine[0]);
                //System.out.println(splitLine[1]);
                //System.out.println(splitLine[2]);
                int a = Character.getNumericValue(splitLine[0].charAt(0));
                int b = Character.getNumericValue(splitLine[0].charAt(1));
                int c = Character.getNumericValue(splitLine[0].charAt(2));
                int d = Character.getNumericValue(splitLine[0].charAt(3));
                int e = Character.getNumericValue(splitLine[0].charAt(4));
                lut[a][b][c][d][e] = Double.valueOf(splitLine[1]);
                accessCnt[a][b][c][d][e] = Integer.valueOf(splitLine[2]);
                line = reader.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            reader.close();
        }
    }

    /**
     * Learn the Q-value of {state, action} vector x from argValue.
     * @param x The {state, action} vector.
     * @param target Target value to be learned.
     */
    @Override
    public void train(double[] x, double target) {
        int a = (int)x[0];
        int b = (int)x[1];
        int c = (int)x[2];
        int d = (int)x[3];
        int e = (int)x[4];

        lut[a][b][c][d][e] = target;
        accessCnt[a][b][c][d][e]++;
    }
}
