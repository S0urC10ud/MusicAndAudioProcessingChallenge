package io.github.danthe1st.multiple_agents.beat_tracking;

import java.util.ArrayList;
import java.util.List;

public class Agent {
	private double beatInterval;
	private double prediction;
	private List<Double> history = new ArrayList<>();
	private double score;
	
	public Agent(double beatInterval, double onset, double onsetScore) {
		this.beatInterval = beatInterval;
		this.prediction = onset + beatInterval;
		this.history = new ArrayList<>();
		this.history.add(onset);
		this.score = onsetScore;
	}
	
	public Agent(Agent agent) {
		beatInterval = agent.beatInterval;
		prediction = agent.prediction;
		history = new ArrayList<>(agent.history);
		score = agent.score;
	}
	
	public double getLastAction() {
		return history.getLast();
	}
	
	public double getPrediction() {
		return prediction;
	}
	
	public void setPrediction(double prediction) {
		this.prediction = prediction;
		history.add(prediction);
	}
	
	public void increasePrediction() {
		prediction += beatInterval;
	}
	
	public void increaseBeatInterval(double increment) {
		beatInterval += increment;
	}
	
	public void increaseScore(double increment) {
		score += increment;
	}
	
	public double getScore() {
		return score;
	}
	
	public double[] getHistory() {
		return history
			.stream()
			.mapToDouble(Double::doubleValue)
			.toArray();
	}
	
	public double getPreTolerance() {
		return beatInterval / 5;// TODO from paper: beatInterval / 5
	}
	
	public double getPostTolerance() {
		// return (2*beatInterval)/5;
		return beatInterval * 2.3;
//		return beatInterval * 2.6;// TODO in paper: 2 * getPreTolerance()
	}
	
//	public boolean isEquivalent(Agent other) {
//		return (Math.abs(beatInterval - other.beatInterval) < 10. / 1000) &&
//				(Math.abs(prediction - other.prediction) < 20. / 1000);
//	}
	
	public double getBeatInterval() {
		return beatInterval;
	}
}
