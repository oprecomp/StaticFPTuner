# StaticFPTuner

StaticFPTuner is a python module for statically tuning the precision of
floating point variables.

It takes as input the name of the application whose variables need to be tuned,
the input set to be fed to the application and the desired maximum error ratio.
The error ratio measures the difference between the output
generated with maximum variable precision and the output obtained at tuned
precision; smaller ratios are preferable.

StaticFPTuner takes a desired error ratio as input and assigns the minimal
number of bit to each benchmark FP variable while ensuring that the
corresponding error is smaller than the desired one.

## Building StaticFPTuner

Requires python > 3.6
Python modules required:
* Empirical Model Learning library (EML)
    - download EML at https://github.com/emlopt/emllib
    - unzip it in <EML_download_destination_folder>
    - mv <EML_downloaded_unzipped> <StaticFPTuner_dir>/eml
    - this exact path is required the correct functioning of StaticFPTuner
        (this behaviour can be modified by changing the value of the variable 
        <eml_path> in tune_variable_precision.py)
* Tensorflow 1.x
    - instructions at https://www.tensorflow.org/install
    - e.g. pip install tensorflow (CPU only)
* keras 
    - instructions at https://keras.io/
* numpy
* scikit-learn
* pandas
* networkX
* yaml

Requires also:
* ILOG CPLEX Optimization Studio > 12 
    - installation instructions:
      https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.studio.help/Optimization_Studio/topics/PLUGINS_ROOT/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/setup_synopsis.html

## Base usage

* cd <StaticFPTuner_dir> 
* python3 tune_variable_precision.py <benchmark> <target_error> <input_set_id>
    - <benchmark>: the application whose variable precisions are to be tuned
    (supported benchmarks can be found in <StaticFPTuner_dir>/benchmarks/)
    - <target_error>: desired bound on the error -- StaticFPTuner expects the
      exponent of the desired error ratio. For instance, assume the user wants to
bound the error obtained with the tuned precision to be lower than 0.001; this
value can be expressed in the form: 1^-{exp}, e.g. 0.001 = 1^{-3}; to impose
this bound the value passed to StaticFPTuner as <target_error> must be 3 (the
negative of the exponent)
    - <input_set_id>: input set for the benchmark; an integer in the range 
    [0, 29], where each values indexes a specific input set
