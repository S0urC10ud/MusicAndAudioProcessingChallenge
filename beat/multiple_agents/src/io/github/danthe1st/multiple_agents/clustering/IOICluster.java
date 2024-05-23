package io.github.danthe1st.multiple_agents.clustering;

public class IOICluster {
	private int size;
	private double intervalSum;
	private int score;
	
	public void addInterval(double interval) {
		intervalSum += interval;
		size++;
	}
	
	public void addCluster(IOICluster other) {
		size += other.size;
		intervalSum += other.intervalSum;
	}
	
	public double getClusterInterval() {
		return intervalSum / size;
	}
	
	public int getSize() {
		return size;
	}
	
	public void addToScore(int increment) {
		score += increment;
	}
	
	public int getScore() {
		return score;
	}
}
