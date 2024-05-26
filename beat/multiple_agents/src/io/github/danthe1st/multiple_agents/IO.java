package io.github.danthe1st.multiple_agents;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class IO {
	
	@FunctionalInterface
	public interface IOConsumer<T> {
		void accept(T o) throws IOException;
	}
	
	public static void readOnsets(DataInputStream dis, IOConsumer<OnsetInformation> beatTracker, IOConsumer<double[]> tempoEstimator) throws IOException {
		while(true){
			switch(dis.read()) {
			case -1 -> {
				System.out.println("connection closed by other party");
				return;
			}
			case 0 -> processBeatTracking(dis, beatTracker);
			case 1 -> processTempoEstimation(dis, tempoEstimator);
			default ->
				throw new IllegalArgumentException("Unexpected value: " + dis.read());
			}
			
		}
	}
	
	private static void processTempoEstimation(DataInputStream dis, IOConsumer<double[]> listener) throws IOException {
		int numberOfOnsets = dis.readInt();
		double[] onsets = readDoubleArray(dis, numberOfOnsets);
		byte sanityByte = dis.readByte();
		if(sanityByte != 0){
			throw new IllegalStateException("Sanity byte wrong!");
		}
		listener.accept(onsets);
	}
	
	private static void processBeatTracking(DataInputStream dis, IOConsumer<OnsetInformation> listener) throws IOException {
		int numberOfOnsets = dis.readInt();
		double[] onsets = readDoubleArray(dis, numberOfOnsets);
		double[] odfValues = readDoubleArray(dis, numberOfOnsets);
		byte sanityByte = dis.readByte();
		if(sanityByte != 0){
			throw new IllegalStateException("Sanity byte wrong!");
		}
		listener.accept(new OnsetInformation(onsets, odfValues));
	}

	private static double[] readDoubleArray(DataInputStream dis, int size) throws IOException {
		double[] data = new double[size];
		for(int i = 0; i < data.length; i++){
			data[i] = dis.readDouble();
		}
		return data;
	}
	
	public static void sendArray(DataOutputStream dos, double[] beats) throws IOException {
		dos.writeInt(beats.length);
		for(double beat : beats){
			dos.writeDouble(beat);
		}
		dos.writeByte(0);
		dos.flush();
	}
	
	private IO() {
		// prevent instantiation
	}
}
