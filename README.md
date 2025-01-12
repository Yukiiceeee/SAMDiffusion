# 



# SAMDiffusion
SAMDiffusion: Semantic Segmentation with Diffusion Model and Segmentation Anything Model

### Conda env installation

```sh
conda create -n SAMDiffusion python=3.8

conda activate SAMDiffusion
```

```
install pydensecrf https://github.com/lucasb-eyer/pydensecrf
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

pip install -r requirements.txt
```

### 1. Prepare SAM2

Refer to [facebookresearch/sam2](https://github.com/facebookresearch/sam2) for the recommended folder structure.  
Place the `sam2` folder in the project directory, prepare the model files, and organize them as follows:

```bash
.
├── SAMDiffusion
│   ├── sam2_optimization.py
│   ├── sam2_optimization_multi.py
│   └── ...
└── sam2
    ├── checkpoints
    ├── configs
    └── ...

```

### 2. Data and mask generation
```
# generating data and attention map witn stable diffusion (Before generating the data, you need to modify the "hunggingface key" in the "Stable_Diffusion" codes to your own key. )
sh ./generate/VOC_multiple_data_generation.sh
```

### 3. Training model with clear data
Following the segmentation model framework from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation),  
we adopt the standard training configuration.

### 4. Our Synthetic Dataset
We are providing synthetic data here with [Baidu Drive](https://pan.baidu.com/s/1LW6ZlRv8w_83oq_xyUiY1g?pwd=abcd) (password: abcd). Feel free to use it.
