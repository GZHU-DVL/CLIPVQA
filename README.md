# CLIPVQA
CLIPVQA: Video Quality Assessment via CLIP 

This is an official implementation of CLIPVQA, a new framework adapting language-image foundation models to video quality assessment.
# Environment Setup
To set up the environment, you can easily run the following command:
```
conda create -n CLIPVQA python=3.7
conda activate CLIPVQA
pip install -r requirements.txt
```

Install Apex as follows
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
