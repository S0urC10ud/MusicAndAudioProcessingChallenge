package io.github.danthe1st.jsonmerge;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.json.JSONObject;

public class JsonMerger {
	
	private static final Path TARGET_PATH = Path.of("/../merged.json");
	private static final Path ONSET_PATH = Path.of("/../train_cnn.json");
	private static final Path BEAT_PATH = Path.of("/../train.json");
	private static final Path TEMPO_PATH = Path.of("/../train.json");
	
	public static void main(String[] args) throws IOException {
		JSONObject onsetData = read(ONSET_PATH);
		JSONObject beatData = read(BEAT_PATH);
		JSONObject tempoData = read(TEMPO_PATH);
		
		if(!onsetData.keySet().equals(beatData.keySet()) || !onsetData.keySet().equals(tempoData.keySet())){
			throw new IllegalStateException("keys don't match");
		}
		
		JSONObject result = new JSONObject();
		for(String key : onsetData.keySet()){
			JSONObject target = new JSONObject();
			copy(onsetData, key, "onsets", target);
			copy(beatData, key, "beats", target);
			copy(tempoData, key, "tempo", target);
			result.put(key, target);
		}
		
		Files.writeString(TARGET_PATH, result.toString());
	}
	
	private static JSONObject read(Path path) throws IOException {
		try(Stream<String> lines = Files.lines(path)){
			return new JSONObject(lines.collect(Collectors.joining("\n")));
		}
	}
	
	private static void copy(JSONObject origin, String originKey, String identifier, JSONObject target) {
		target.put(identifier, origin.getJSONObject(originKey).getJSONArray(identifier));
	}
}
