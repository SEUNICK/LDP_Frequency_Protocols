# LDP_Frequency_Protocols
This repository lists our implementations of LDP frequency protocols.

Related Paper: 'Further Study on Frequency Estimation under Local Differential Privacy' accepted by USENIX Security 2025

## Code
- main.py: main entrance for experiment (with multiprocessing to save time and memory) and save results in the results folder
- runtime.py: get the aggregation runtime (without multiprocessing) of every frequency protocol and save results in the results folder
- drawxx.py: draw the experimental results into figures and save them in the draw folder
- yellow_tripdata_2024-03.parquet: the taxi dataset from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- fp: source code folder of LDP frequency protocols
  - original protocols
    - GRR
    - OUE
    - OLH
    - SS
  - proposed protocols
    - RUE
    - RLH
    - RWS

## Environment
I mainly used Python 3.10.4 with NumPy to run experiments and Matplotlib to draw the results. PyArrow is used to read the taxi dataset from TLC Trip Record Data. Although not tested, the code should be compatible with any recent version. In case your local environment does not work, just run `pip install -r requirements.txt`. The code is compatible with any commodity computer with a minimum of 16GB of RAM and requires around 300MB of disk space. The memory requirement mainly comes from the runtime.py and more RAM is required if the customized domain size in the runtime experiment is larger than 4096. Multi-core CPUs can significantly speed up the experiments. I run the experiments on a PC with AMD Ryzen 9 7950X and 64GB memory.

## Knowing compatibility issues 
NumPy 2.0 is a major release and I find there are some compatibility issues betweent the newest PyArrow and NumPy, especilly for the pyarrow.parquet.read_table function. To avoid potential issues, I fixed numpy==1.26.4 in the requirements.txt.

## Run experiments
1. custom the parameters in the main.py and runtime.py as you wish
   - run_repeat_time: default 100 but very time-consuming. You can reduce it to a small number (e.g. 10), but the results for small domain d would fluctuate more.
   - run_cpu_count: default 0 which means using all logical cores in multiprocessing. If you want to do something else while experimenting, it can be set to a number less than the number of logical cores. Besides, it should be less than or equal to 61 on Windows. This is a Windows specific limit, tied to the MAXIMUM_WAIT_OBJECTS limit of WaitForMultipleObjects.
   - run_protocol_list: frequency protocol to run experiments
   - scan d parameters:
     - run_epsilon: default 4
     - run_d_range: default from 2 to 4096
   - scan epsilon parameters:
     - run_d: default 128
     - run_epsilon_range: default from 1 to 5 with step size 0.25
2. run main.py
3. run runtime.py - Runtime and memory costs are linearly related to data size and domain size, so the taxi dataset requires about 36 times as much runtime and RAM as the synthetic dataset. For quick verification or if memory is limited, users can run this experiment only on the synthetic datasets. The range of run_d_range can also be reduced to speed up the experiment and reduce the RAM requirements.
4. run each drawxx.py to draw the results