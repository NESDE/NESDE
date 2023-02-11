# NESDE Algorithm implementation

## Installation:
Within the project directory:

pip install -e .

## Experiments:
### SDE synthetic data:
The experimets located in the jupyter notebook at: './Experiments/Synthetic_Data/run_experiment.ipynb'

To generate data, run the script:'./Experiments/Controlled_SDE/SDE_datasets_gen.py --<name>'

Where <name> is either 'efficiency'/'ood'/'regular'

### Blood Coagulation and Vancomycin Dosing:
To generate the data, use an SQL server with the MIMIC-IV database (available at https://physionet.org/content/mimiciv/1.0/, although need to get a permission)

Then follow the instructions at the README file within: './Experiments/Blood_Coagulation/Data' or './Experiments/Vancomycin_Dosing/Data'

The NESDE and LSTM baseline implementations are located at: './Experiments/Blood_Coagulation' and './Experiments/Vancomycin_Dosing'

### Simulators:
#### look for the examples' at: './Simulator_Suite/Blood_Coagulation_Simulator', './Simulator_Suite/Vancomycin_Dosing_Simulator'


