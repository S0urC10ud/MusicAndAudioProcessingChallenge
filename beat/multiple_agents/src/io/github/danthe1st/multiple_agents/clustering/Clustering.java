package io.github.danthe1st.multiple_agents.clustering;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

public class Clustering {
	
	private static final double MAX_CLUSTER_WIDTH = 25. / 1000;
	
	private List<IOICluster> clusters = new ArrayList<>();
	
	public static List<IOICluster> getClusters(double[] onsets) {
		Clustering clustering = new Clustering();
		clustering.createClusters(onsets);
		clustering.mergeClusters();
		clustering.calculateScores();
		return clustering.clusters;
	}
	

	private Clustering() {
	}
	
	
	private void createClusters(double[] onsets) {
		if(!clusters.isEmpty()){
			throw new IllegalStateException("clusters should initially be empty");
		}
		
		for(int i = 0; i < onsets.length; i++){
			for(int j = i + 1; j < onsets.length; j++){
				double distance = Math.abs(onsets[i] - onsets[j]);
				IOICluster foundCluster = findBestClusterForInterval(distance);
				if(foundCluster == null){
					foundCluster = new IOICluster();
					clusters.add(foundCluster);
				}
				foundCluster.addInterval(distance);
			}
		}
	}
	
	private IOICluster findBestClusterForInterval(double distance) {
		IOICluster bestCluster = null;
		double bestClusterDifference = Double.POSITIVE_INFINITY;
		for(IOICluster cluster : clusters){
			double difference = Math.abs(cluster.getClusterInterval() - distance);
			if(difference < bestClusterDifference && difference < MAX_CLUSTER_WIDTH){
				bestClusterDifference = difference;
				bestCluster = cluster;
			}
		}
		return bestCluster;
	}
	
	private void mergeClusters() {
		BitSet deletedClusters = new BitSet(clusters.size());
		for(int i = 0; i < clusters.size(); i++){
			if(deletedClusters.get(i)) {
				continue;
			}
			IOICluster firstCluster = clusters.get(i);
			for(int j = 0; j < clusters.size(); j++){
				if(deletedClusters.get(j) || i == j){
					continue;
				}
				IOICluster secondCluster = clusters.get(j);
				if(Math.abs(firstCluster.getClusterInterval() - secondCluster.getClusterInterval()) < MAX_CLUSTER_WIDTH){
					firstCluster.addCluster(secondCluster);
					deletedClusters.set(j);
				}
			}
		}
		List<IOICluster> newClusters = new ArrayList<>();
		for(int i = 0; i < clusters.size(); i++){
			if(!deletedClusters.get(i)){
				newClusters.add(clusters.get(i));
			}
		}
		clusters = newClusters;
	}
	
	private void calculateScores() {
		for(IOICluster firstCluster : clusters){
			for(IOICluster secondCluster : clusters){
				int n;
				for(n = 1; n * firstCluster.getClusterInterval() < secondCluster.getClusterInterval() - MAX_CLUSTER_WIDTH; n++){
					// just for calculating n
				}
				if(Math.abs(n * firstCluster.getClusterInterval() - secondCluster.getClusterInterval()) < MAX_CLUSTER_WIDTH){
					secondCluster.addToScore(calculateRawScoreFromIntegerMultiple(n) * firstCluster.getSize());
				}
			}
		}
	}
	
	private int calculateRawScoreFromIntegerMultiple(int n) {
		if(n <= 4){
			return 6 - n;
		}
		if(n <= 8){
			return 1;
		}
		return 0;
	}
}
