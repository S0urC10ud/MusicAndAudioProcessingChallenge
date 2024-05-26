package io.github.danthe1st.multiple_agents;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import io.github.danthe1st.multiple_agents.beat_tracking.BeatTracking;
import io.github.danthe1st.multiple_agents.clustering.Clustering;
import io.github.danthe1st.multiple_agents.clustering.IOICluster;

public class MultipleAgents {
	
	private static final int TOP_K_HYPOTHESIS_BEAT_TRACKING = 10;
	private static final double MAX_CLUSTER_WIDTH_BEAT_TRACKING = 25. / 1000;
//	private static final double MAX_CLUSTER_WIDTH_TEMPO_ESTIMATION = 20. / 1000;
	private static final double MAX_CLUSTER_WIDTH_TEMPO_ESTIMATION = 21. / 1000;
	
	public static void main(String[] args) throws IOException {
		
		try(ServerSocket serv = new ServerSocket(1337)){
			while(!Thread.currentThread().isInterrupted()){
				try(Socket connection = serv.accept()){
					handleConnection(connection);
				}catch(Exception e){
					e.printStackTrace();
				}
			}
		}
	}
	
	private static void handleConnection(Socket connection) throws IOException {
		try(DataInputStream is = new DataInputStream(new BufferedInputStream(connection.getInputStream()));
				DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(connection.getOutputStream()))){
			IO.readOnsets(
					is, piece -> beatDetection(dos, piece), onsets -> tempoEstimation(dos, onsets)
			);
		}
	}

	private static void tempoEstimation(DataOutputStream dos, double[] onsets) throws IOException {
//		double[] hypothesis = Clustering.getClusters(onsets, MAX_CLUSTER_WIDTH_TEMPO_ESTIMATION)
//			.stream()
//			.sorted(Comparator.comparingInt(IOICluster::getScore).reversed())
//			.mapToDouble(c -> 60 / c.getClusterInterval())
//			.filter(bpm -> bpm >= 60 && bpm <= 200)
//			.limit(2)
//			.toArray();
//		IO.sendArray(dos, hypothesis);
		
		double bestHypothesis = Clustering.getClusters(onsets, MAX_CLUSTER_WIDTH_TEMPO_ESTIMATION)
			.stream()
			.map(c -> new TempoInfo(60 / c.getClusterInterval(), c.getScore()))
			.filter(c -> c.bpm() >= 60 && c.bpm() <= 200)
			.max(Comparator.comparingInt(TempoInfo::score))
			.orElseThrow().bpm();
		
//		double[] hypothesis = Clustering.getClusters(onsets, MAX_CLUSTER_WIDTH_TEMPO_ESTIMATION)
//		.stream()
//		.sorted(Comparator.comparingInt(IOICluster::getScore).reversed())
//		.mapToDouble(c -> 60 / c.getClusterInterval())
//		.filter(bpm -> bpm >= 60 && bpm <= 200)
//		.sorted()
//		.toArray();
//		double bestHypothesis = hypothesis[hypothesis.length / 2];
		
		double secondHypothesis;
		if(bestHypothesis > 115){
			secondHypothesis = bestHypothesis / 2;
		}else{
			secondHypothesis = bestHypothesis * 2;
		}
		
		IO.sendArray(dos, new double[] { bestHypothesis, secondHypothesis });
	}
	
	private record TempoInfo(double bpm, int score) {
		
	}
	
	private static void beatDetection(DataOutputStream dos, OnsetInformation piece) throws IOException {
		double[] beats;
		try{
			System.out.println("clustering...");
			double[] tempoHypothesis = getIntervalHypothesis(piece.onsetTimes(), TOP_K_HYPOTHESIS_BEAT_TRACKING);
			
//			System.out.println("start with " + tempoHypothesis.length + " hypothesis");
			System.out.println("average tempo hypothesis: " + Arrays.stream(tempoHypothesis).average().orElseThrow());
			System.out.println("min hypothesis: " + Arrays.stream(tempoHypothesis).min().orElseThrow());
			
			beats = BeatTracking.trackBeats(tempoHypothesis, piece);
		}catch(Exception e){
			e.printStackTrace();
			beats = new double[0];
		}
		IO.sendArray(dos, beats);
	}
	
	private static double[] getIntervalHypothesis(double[] onsetTimes, int numberOfHypothesis) {
		List<IOICluster> clusters = Clustering.getClusters(onsetTimes, MAX_CLUSTER_WIDTH_BEAT_TRACKING);
		
		return clusters
			.stream()
			.filter(h -> h.getClusterInterval() < 3)
			.sorted(Comparator.comparingInt(IOICluster::getScore).reversed())
			.limit(numberOfHypothesis)
			.mapToDouble(IOICluster::getClusterInterval)
			.toArray();
	}
	
}
