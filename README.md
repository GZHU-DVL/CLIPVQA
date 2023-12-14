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

# Train
The config files lie in `configs`. For example, to train CLIPVQA-B/16 with 32 frames on KoNViD-1k on 1 GPUs, you can run
```
python -m torch.distributed.launch --nproc_per_node=1 \ 
main.py -cfg configs/k400/16_32.yaml --output /PATH/TO/OUTPUT --accumulation-steps 4
```

# Test
For example, to test the CLIPVQA-B/16 with 32 frames on KoNViD-1k, you can run
```
python -m torch.distributed.launch --nproc_per_node=1 main.py \
-cfg configs/k400/32_8.yaml --output /PATH/TO/OUTPUT --only_test --resume /PATH/TO/CKPT \
--opts TEST.NUM_CLIP 1 TEST.NUM_CROP 1
```

# Acknowledgements
Parts of the codes are borrowed from [mmaction2](https://github.com/open-mmlab/mmaction2), [Swin](https://github.com/microsoft/Swin-Transformer) and [CLIP](https://github.com/openai/CLIP). Sincere thanks to their wonderful works.
