package com.robocode;

/**
 * State of robot
 * - State 1 : X position {0-800} -> {0.00-8.00}
 * - State 2 : Y position {0-600} -> {0.00-6.00}
 * - State 3 : Distance to enemy {0-1000} -> {0.00-10.00}
 * - State 4 : Energy {0-100} -> {0.0-10.00}
 */
public class State {
    public double xPos;
    public double yPos;
    public double dist;
    public double energy;

    // Constructor
    public State (double xPos, double yPos, double dist, double energy) {
        this.xPos = xPos;
        this.yPos = yPos;
        this.dist = dist;
        this.energy = energy;
    }

    // Get methods
    public double getXPos() {
        return xPos;
    }
    public double getYPos() { return yPos; }
    public double getDist() { return dist; }
    public double getEnergy() {
        return energy;
    }

    // Set methods
    public void setXPos(double xPos) {
        this.xPos = xPos;
    }
    public void setYPos(double yPos) { this.yPos = yPos; }
    public void setDist(double dist) { this.dist = dist; }
    public void setEnergy(double energy) {
        this.energy = energy;
    }

    // Copy method
    public void copyState(State s) {
        this.xPos = s.xPos;
        this.yPos = s.yPos;
        this.dist = s.dist;
        this.energy = s.energy;
    }

    // Convert to string
    public String toString() {
        return xPos + ", " + yPos + ", " + dist + ", " + energy;
    }
}
