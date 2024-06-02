package io.github.danthe1st.multiple_agents.beat_tracking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

import io.github.danthe1st.multiple_agents.OnsetInformation;

public class BeatTracking {
	private static final double TIMEOUT = 5.6;
	private static final double STARTUP_PERIOD = 10;
	private static final double TOLERANCE_INNER = 40. / 1000;
	private static final double CORRELATION_FACTOR = 35;
	
	private List<Agent> agents = new ArrayList<>();
	private final double[] onsets;
	private final double[] odfValues;
	
	public static double[] trackBeats(double[] tempoHypothesis, OnsetInformation piece) {
		BeatTracking tracking = new BeatTracking(tempoHypothesis, piece);
		tracking.perform();
		Agent bestAgent = tracking.getBestAgent();
		System.out.println("interval at end: " + bestAgent.getBeatInterval());
		return bestAgent.getHistory();
	}
	
	private BeatTracking(double[] tempoHypothesis, OnsetInformation piece) {
		this.odfValues = piece.odfValues();
		this.onsets = piece.onsetTimes();
		for(double hypothesis : tempoHypothesis){
			for(int i = 0; i < onsets.length && onsets[i] < STARTUP_PERIOD; i++){
				agents.add(new Agent(hypothesis, onsets[i], odfValues[i]));
			}
		}
		System.out.println("start with " + agents.size() + " agents");
	}
	
	private void perform() {
		for(int i = 0; i < onsets.length; i++){
			double onsetTime = onsets[i];
			final double lastOnsetTime;
			if(i > 10){
				lastOnsetTime = onsets[i - 10];
			}else{
				lastOnsetTime = 0;
			}
			double odfValue = odfValues[i];
			
			List<Integer> agentsToRemove = Collections.synchronizedList(new ArrayList<>());
			List<Agent> newAgents = Collections.synchronizedList(new ArrayList<>());
			IntStream.range(0, agents.size())
				.mapToObj(j -> new AgentInfo(j, agents.get(j)))
//				.parallel()
				.forEach(
						info -> processEventWithAgent(info.agent, onsetTime, lastOnsetTime, odfValue, agentsToRemove, newAgents, info.agentId)
			);
			agents.addAll(newAgents);
			clearDuplicateAgents(agentsToRemove);
		}
	}
	
	record AgentInfo(int agentId, Agent agent) {
	}

	private void processEventWithAgent(Agent agent, double onsetTime, final double lastOnsetTime, double odfValue,
			List<Integer> agentsToRemove, List<Agent> newAgents, int agentId) {
		// if there are few acceptable onsets, ignore the timeouts
		if(onsetTime - agent.getLastAction() > TIMEOUT && (Math.abs(onsetTime - lastOnsetTime) < TIMEOUT / 5)){
			agentsToRemove.add(agentId);
		}else{
//			int i = 0;
			while(agent.getPrediction() + agent.getPostTolerance() < onsetTime){
//				if(i++ != 0){//interpolation
//					agent.setLastAction(onsetTime);
//				}
				agent.increasePrediction();
//				agent.setLastAction(onsetTime);
			}
			if(isInTolerance(onsetTime, agent)){
				if(Math.abs(agent.getPrediction()-onsetTime)>TOLERANCE_INNER) {
					newAgents.add(new Agent(agent));
				}
				double error = onsetTime - agent.getPrediction();
				double relativeError = error / agent.getBeatInterval();
//				if(error > 2){
//					agentsToRemove.add(agentId);// if an error of 2s is tolerated, we just remove the agent
//				}
				agent.increaseBeatInterval(error / CORRELATION_FACTOR);
				agent.setPrediction(onsetTime);
				agent.increasePrediction();
				agent.increaseScore((1 - relativeError / 2) * odfValue);
			}
		}
	}

	private boolean isInTolerance(double onsetTime, Agent agent) {
		return agent.getPrediction() + agent.getPreTolerance() <= onsetTime &&
				onsetTime <= agent.getPrediction() + agent.getPostTolerance();
	}
	
	private void clearDuplicateAgents(List<Integer> knownAgentsToRemove) {
		Comparator<Agent> cmp = Comparator.comparingDouble(Agent::getBeatInterval);
		Collections.sort(agents, cmp);
		boolean[] toDelete = new boolean[agents.size()];
		for(Integer integer : knownAgentsToRemove){
			toDelete[integer] = true;
		}
		IntStream.range(0, agents.size()).forEach(i -> {
			checkForDuplicates(toDelete, i);
		});
		agents = findNewAgentsWithDeletionBitmap(toDelete);
	}

	private List<Agent> findNewAgentsWithDeletionBitmap(boolean[] toDelete) {
		List<Agent> newAgents = new ArrayList<>();
		for(int i = 0; i < agents.size(); i++){
			Agent agent = agents.get(i);
			if(!toDelete[i]){
				newAgents.add(agent);
			}
		}
		
//		int numDeleted = agents.size() - newAgents.size();
//		System.out.println("deleted " + numDeleted + "; " + newAgents.size() + " left");
		
		return newAgents;
	}

	private void checkForDuplicates(boolean[] toDelete, int firstAgentId) {
		if(toDelete[firstAgentId]){
			return;
		}
		Agent firstAgent = agents.get(firstAgentId);
		
		for(int j = firstAgentId + 1; j < agents.size(); j++){
			if(toDelete[j]){
				continue;
			}
			Agent secondAgent = agents.get(j);
			
			// Due to sorting the agents, we can skip further agents if these agents are sufficiently different
			if(secondAgent.getBeatInterval() - firstAgent.getBeatInterval() > 10. / 1000){
				return;
			}
			if(Math.abs(firstAgent.getPrediction() - secondAgent.getPrediction()) < 20. / 1000){
				if(firstAgent.getScore() <= secondAgent.getScore()){
					toDelete[firstAgentId] = true;
					return;
				}
				toDelete[j] = true;
			}
		}
	}
	
	private Agent getBestAgent() {
		return agents
			.stream()
			.max(Comparator.comparingDouble(Agent::getScore))
			.orElseThrow(() -> new IllegalStateException("no best agent found"));
	}
}
