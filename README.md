This is the open-source repository for the paper "Pairwise Neural Cross-Correlation on Raw I/Q", currently under review at MILCOM 2025.

To run the code, first generate a dataset by running

`python3 generate_data.py`

Then you can run all experiments in the paper on Linux by running

`bash run_cc.sh`

The code makes use of Pytorch, Numpy, Scipy, Scikit-Learn, and Matplotlib dependencies.

If you find this code useful, please cite the associated technical report:

```
@inproceedings{dominguez2025,
author="Dominguez, Miguel",
title="Pairwise Neural Cross-Correlation on Raw I/Q",
booktitle="Under Review at MILCOM 2025",
address="Los Angeles, USA",
month= oct,
year="2025"}
```