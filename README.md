# Audio-and-Music-Processing Team V
In this document we provide the necessary info to run the individual implementations.


### CNN
For evaluation, the file `detector_cnn.py` can be used with the default `detector.py`-usage. For training, the directory `onsets/cnn` should be examined where `train.py` holds the final (best) hyperparameters. If some additional hyperparameter-tuning has to be done, `train_sweep_final.py` in this directory can be examined. The pre-trained models are `cnn_output_cpu.dmp` and `cnn_output.dmp` in the root directory respectively for cuda.

### Wavebeat
Regarding the inference, `detector_wavebeat.py` can be utilized (again with the default usage - for example `python detector_wavebeat.py data/test test_wavbeat_preds.pd`). For the training, the directory `beat/wavbeat` should be examined where pytorch lightning is used for the training loop. 

For training, the first two lines of `model.py` have to be adjusted as follows (but then inference will not work):
```py
from loss import BCFELoss
from eval import evaluate, find_beats
```

The default lines are:
```py
from beat.wavebeat.loss import BCFELoss
from beat.wavebeat.eval import evaluate, find_beats
```

The file `train_multiproc.py` is the recommended script to perform the training which performed trainsfer-learning from the pre-trained model.

When encountering issues, please contact [martin.dallinger@outlook.com](mailto:martin.dallinger@outlook.com).

### Multiple Agents
In order to use the multiple agents approach for beat detection or tempo estimation, first run the Java program located in `beat/multiple_agents` in order to listen on port 1337. The main class is `io.github.danthe1st.multiple_agents.MultipleAgents`.

The Java program can be compiled and run by executing the following commands in `beat/multiple_agents`:

```bash
javac -d bin/ $(find src -name '*.java')
java -cp bin/ io.github.danthe1st.multiple_agents.MultipleAgents
```

It is recommended to use Java 21.
On Windows, it would be necessary to list all Java files for compilation:
```bash
javac -d bin src/io/github/danthe1st/multiple_agents/IO.java src/io/github/danthe1st/multiple_agents/clustering/IOICluster.java src/io/github/danthe1st/multiple_agents/clustering/Clustering.java src/io/github/danthe1st/multiple_agents/MultipleAgents.java src/io/github/danthe1st/multiple_agents/beat_tracking/BeatTracking.java src/io/github/danthe1st/multiple_agents/beat_tracking/Agent.java src/io/github/danthe1st/multiple_agents/OnsetInformation.java
java -cp bin/ io.github.danthe1st.multiple_agents.MultipleAgents
```

## How to get predictions
There are 2 detector scripts which only differe in how onsets and onset-detection-function are obtained: the first file called `detector.py` uses the superflux approach for onset-detection while the second file called `detector_cnn.py` uses a CNN-based approach. Other than that both files use the our best approaches for both beat detection and tempo estimation. However other methods for these tasks are also implemented in this code-base. In order to use them, simply comment-in or comment-out the approaches you want to use.  
**Important:** If you intend to use Multiple Agents (which is the default for beat tracking), please read the section above. If you don't intend to use them (or don't want to install Java), just comment out its import *and* function-call.

As the two detector files are based on the the provided template, they both require two inputs: one for a directory of `.wav` and another for the output JSON file

```
$ ./detector.py train/ train.json
$ ./detector_cnn.py train/ train_cnn.json
```


## Merging JSON files
In order to merge JSON files of multiple approaches together, a Java program in the `jsonmerge` directory can be used. This program requires [Maven 3](https://maven.apache.org/) and Java 21.

Once these requirements are satisfied (make sure to set the `JAVA_HOME` to a Java 21 JDK), run the following command in the `jsonmerge` directory:
```bash
mvn compile exec:java -Dexec.mainClass=io.github.danthe1st.jsonmerge.JsonMerger
```

This merges the onsets from `train_cnn.json` with the beats and tempi from `train.json` into a file named `merged.json`.