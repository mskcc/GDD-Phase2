# GDD-Phase2
This is the Second Generation of Genome Derived Diagnosis AI Project



## Prerequisites
* Conda
```
curl -Ok https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda create --name gddP2 python=3.8
```
* python
* numpy
```
conda install numpy
```
* pandas
```
conda install pandas
```
* [scikit-learn](https://scikit-learn.org/stable/index.html)
```
conda install -c conda-forge scikit-learn
```

* [PyTorch](https://pytorch.org/)
```
conda install pytorch torchvision torchaudio -c pytorch
```
* [Imbalanced-Learn Library](https://imbalanced-learn.org/stable/index.html)
```
conda install -c conda-forge imbalanced-learn
```
* [Scopt Library](https://scikit-optimize.github.io/stable/index.html)
```
conda install -c conda-forge scikit-optimize
```

## Setting up

```
git clone https://github.com/mskcc/GDD-Phase2.git

cd GDD-Phase2/

git checkout feature_ss_hpc

mkdir -p ./Data/FeatureTable

```


## LSF commands

```
bjobs

bpeek jobid

bjobs -p3 -l <jobid>

bhosts

bmgroup

bqueues
```
