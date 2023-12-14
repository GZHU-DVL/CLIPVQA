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
VideoQualitylanguage

Since that our method employs semantic information in text labels, rather than traditional MOS label, it is necessary to provide a textual description. For example, we provide the text description of video quality assessment dataset in the file `labels/labels.csv`. Here is the format:
```Shell
id,name
0, Excellent
1, Good
2, Fair
3, Poor
4, Bad
5, Terrible
```
The `id` indicates the  quality class, while the `name` denotes the text description.
