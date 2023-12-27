# TransFusE DTI

**E**fficient **D**rug-**T**arget **I**nteractions Prediction Framework via **Trans**ferable Knowledge **Fus**ing



## Abstract

Drug-target integrations (DTI) prediction is a niche in drug discovery, streamlining the search for potential drugs. Computer-aided drug discovery (CADD) has gained traction for its precise predictions, efficiency, and adaptability across various situations. Yet, the computational demands of current top CADD models hinder their practical use due to heavy resource needs.

In this research, we introduce TransFusE DTI, an effective framework for predicting DTIs that leverages pre-trained knowledge to construct models that optimize predictive accuracy while minimizing computational demands. The encoder uses a pre-extracted embedding vector from ProtBERT to reduce computational load and adapts a smaller ProtBERT model. It also includes target-related functional text to boost predictive accuracy. We evaluate the performance of TransFusE DTI using three widely-recognized benchmark datasets: BIOSNAP, DAVIS, and BindingDB, and compare its results to prior studies.

Our results demonstrate that TransFusE DTI exhibits superior predictive performance on the BIOSNAP and BindingDB datasets. Notably, the model's parameter count is only 60% of that of the previous top-performing model by [Kang et al. (2022)](https://www.mdpi.com/1999-4923/14/8/1710), and it operates efficiently with a learning rate of 26%. Furthermore, the model's video memory requirement is 11.2 GB, rendering it suitable for use on general-purpose Graphics Processing Units (GPUs). 



### Graphical abstract

## ![figure_1](/Users/jonghyunlee/Workspace/TransFusE-DTI/fig/figure_1.jpeg)



### Conceptual diagram

![figure_2](/Users/jonghyunlee/Workspace/TransFusE-DTI/fig/figure_2.jpeg)



## Installation

```bash
pip install -r requirments.txt
```



## Run experiments

```bash
python ./run.py -c config/EXAMPLE_CONFIG_FILE.yaml
```



## Datasets

- Original datasets: [MolTrans GitHub Repo](https://github.com/kexinhuang12345/MolTrans/tree/master/dataset)
- Our datasets: [Zenodo]()
- Target sequence embeddings from pre-trained ProtBERT encoder: [Zenodo]()



## Performances

| Dataset   | Method        | AUROC         | AUPRC         | Sensitivity   | Specificity   |
| --------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| BIOSNAP   | GNN-CPI       | 0.880 ± 0.007 | 0.891 ± 0.004 | 0.781 ± 0.014 | 0.820 ± 0.012 |
|           | DeepDTI       | 0.877 ± 0.005 | 0.877 ± 0.006 | 0.790 ± 0.027 | 0.846 ± 0.017 |
|           | DeepDTA       | 0.877 ± 0.005 | 0.884 ± 0.006 | 0.782 ± 0.015 | 0.825 ± 0.012 |
|           | DeepConv-DTI  | 0.884 ± 0.002 | 0.890 ± 0.005 | 0.771 ± 0.023 | 0.833 ± 0.016 |
|           | MolTrans      | 0.896 ± 0.002 | 0.902 ± 0.004 | 0.776 ± 0.032 | 0.852 ± 0.014 |
|           | Kang et al.   | 0.914 ± 0.006 | 0.900 ± 0.007 | 0.862 ±0.025  | 0.847 ± 0.007 |
|           | TransFusE DTI | 0.916 ± 0.005 | 0.915 ± 0.005 | 0.847 ± 0.010 | 0.846 ± 0.018 |
| DAVIS     | GNN-CPI       | 0.841 ± 0.012 | 0.270 ± 0.020 | 0.697 ± 0.047 | 0.843 ± 0.039 |
|           | DeepDTI       | 0.862 ± 0.002 | 0.232 ± 0.006 | 0.752 ± 0.015 | 0.854 ± 0.012 |
|           | DeepDTA       | 0.881 ± 0.007 | 0.303 ± 0.044 | 0.765 ± 0.045 | 0.866 ± 0.020 |
|           | DeepConv-DTI  | 0.885 ± 0.008 | 0.300 ± 0.039 | 0.755 ± 0.040 | 0.881 ± 0.024 |
|           | MolTrans      | 0.908 ± 0.002 | 0.405 ± 0.016 | 0.801 ± 0.022 | 0.877 ± 0.013 |
|           | Kang et al.   | 0.920 ± 0.002 | 0.395 ± 0.007 | 0.824 ± 0.026 | 0.889 ± 0.015 |
|           | TransFusE DTI | 0.883 ± 0.010 | 0.357 ± 0.025 | 0.847 ± 0.035 | 0.776 ± 0.061 |
| BindingDB | GNN-CPI       | 0.888 ± 0.002 | 0.558 ± 0.015 | 0.742 ± 0.013 | 0.897 ± 0.011 |
|           | DeepDTI       | 0.909 ± 0.003 | 0.614 ± 0.015 | 0.770 ± 0.028 | 0.915 ± 0.021 |
|           | DeepDTA       | 0.901 ± 0.004 | 0.579 ± 0.015 | 0.755 ± 0.015 | 0.904 ± 0.011 |
|           | DeepConv-DTI  | 0.845 ± 0.002 | 0.430 ± 0.005 | 0.652 ± 0.024 | 0.896 ± 0.023 |
|           | MolTrans      | 0.914 ± 0.003 | 0.623 ± 0.012 | 0.781 ± 0.035 | 0.916 ± 0.016 |
|           | Kang et al.   | 0.922 ± 0.001 | 0.623 ± 0.010 | 0.814 ± 0.025 | 0.916 ± 0.016 |
|           | TransFusE DTI | 0.911 ± 0.005 | 0.636 ± 0.013 | 0.890 ± 0.019 | 0.787 ± 0.010 |