package io.github.danthe1st.multiple_agents;

import java.util.Objects;

public record OnsetInformation(double[] onsetTimes, double[] odfValues) {// NOSONAR this is a record
	public OnsetInformation {
		Objects.requireNonNull(onsetTimes);
		Objects.requireNonNull(odfValues);
		if(onsetTimes.length != odfValues.length){
			throw new IllegalArgumentException();
		}
	}
}
