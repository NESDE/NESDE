# NESDE Algorithm implementation

## Installation:
Within the project directory:

pip install -e .

## Experiments:
### SDE synthetic data:
The experimets located in the jupyter notebook at: './Experiments/Controlled_SDE/NESDE_vs_LSTM.ipynb'

To generate data, run the script:'./Experiments/Controlled_SDE/SDE_gen.py'

### Blood Coagulation forecasting:
To generate the data, use an SQL server with the MIMIC-IV database (available at https://physionet.org/content/mimiciv/1.0/, although need to get a permission)

Then follow the instructions at the README file within: './Experiments/Blood_Coagulation/Data'

The NESDE and LSTM baseline implementations are located at: './Experiments/Blood_Coagulation'

### Blood coagulation simulator:
#### look for the example at: './Experiments/Blood_Coagulation_Simulator


