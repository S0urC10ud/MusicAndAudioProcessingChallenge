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
	
	private static final int TOP_K_HYPOTHESIS = 10;
	
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
			IO.readOnsets(is, piece -> {
				beatDetection(dos, piece);
			});
		}
	}

	private static void beatDetection(DataOutputStream dos, OnsetInformation piece) throws IOException {
		double[] beats;
		try{
			System.out.println("clustering...");
			List<IOICluster> clusters = Clustering.getClusters(piece.onsetTimes());
			
			double[] tempoHypothesis = clusters
				.stream()
				.filter(h -> h.getClusterInterval() < 3)
				.sorted(Comparator.comparingInt(IOICluster::getScore).reversed())
				.limit(TOP_K_HYPOTHESIS)
				.mapToDouble(IOICluster::getClusterInterval)
				.toArray();
			
//			System.out.println("start with " + tempoHypothesis.length + " hypothesis");
			System.out.println("average tempo hypothesis: " + Arrays.stream(tempoHypothesis).average().orElseThrow());
			System.out.println("min hypothesis: " + Arrays.stream(tempoHypothesis).min().orElseThrow());
			
			beats = BeatTracking.trackBeats(tempoHypothesis, piece);
		}catch(Exception e){
			e.printStackTrace();
			beats = new double[0];
		}
		IO.sendBeats(dos, beats);
	}
	
}
