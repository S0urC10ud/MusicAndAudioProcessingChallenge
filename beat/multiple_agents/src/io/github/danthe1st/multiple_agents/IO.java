package io.github.danthe1st.multiple_agents;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;

public class IO {
	
	@FunctionalInterface
	public interface IOConsumer<T> {
		void accept(T o) throws IOException;
	}
	
	public static void readOnsets(DataInputStream dis, IOConsumer<OnsetInformation> listener) throws IOException {
		while(true){
			int numberOfOnsets;
			try{
				numberOfOnsets = dis.readInt();
			}catch(EOFException e){
				System.out.println("connection closed by other party");
				return;
			}
			double[] onsets = readDoubleArray(dis, numberOfOnsets);
			double[] odfValues = readDoubleArray(dis, numberOfOnsets);
			byte sanityByte = dis.readByte();
			if(sanityByte != 0){
				throw new IllegalStateException("Sanity byte wrong!");
			}
			listener.accept(new OnsetInformation(onsets, odfValues));
		}
	}

	private static double[] readDoubleArray(DataInputStream dis, int size) throws IOException {
		double[] data = new double[size];
		for(int i = 0; i < data.length; i++){
			data[i] = dis.readDouble();
		}
		return data;
	}
	
	public static void sendBeats(DataOutputStream dos, double[] beats) throws IOException {
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
