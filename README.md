# Variable Rate Learned Wavelet Video Coding with Temporal Layer Adaptivity

This repository contains training and inference code for the paper "Variable rate learned wavelet video coding using temporal layer adaptivity". 
The paper is accepted for ICIP2025 and available on [Arxiv](https://arxiv.org/abs/2410.15873).

## Installation

Setup conda environment and install Pytorch:
```bash
conda create -n $ENV_NAME python=3.8
conda activate $ENV_NAME

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

Build C++ code for bitstream generation (same entropy coder as [DCVC-DC](https://github.com/microsoft/DCVC/tree/main/DCVC-DC)). <br>
In folder pMCTF, run the following:
```bash
mkdir build
cd build
conda activate $ENV_NAME
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
make -j
```

## Usage

### Training
The training data set for both the image and video coding model is the Vimeo-Septuplet data set available [here](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip). <br>
Image coder pWave++:
```bash
python train_pWave.py -d /home/data/vimeo_septuplet --cuda --epochs 31 --seed 0
```

Video coder pMCTF-L:
```bash
python train_pMCTF_L.py -d /home/data/vimeo_septuplet --cuda --seed 0 --num_me_stages 3 --iframe_path checkpoints/pwave++/state_epoch30.pth.tar
```
For training the video model, a pretrained image coder and a checkpoint for the optical flow network is required. The checkpoints used in the paper can be downloaded below.
### Evaluation
For evaluation, setup path to UVG data set in [configs/dataset_config.json](configs/dataset_config.json). The sequences in YUV format are available [here](https://ultravideo.fi/dataset.html).
Run the following command for evaluation:
```bash
 python test_pMCTF_flex.py --model_path checkpoints/pMCTF-L/pMCTF_L_epoch28.pth.tar --test_config ./configs/dataset_config.json --cuda 1 --write_stream 1 --force_intra_period 16 --force_frame_num 96  --ds_name UVG --skip_decoding --verbose 3 --two_stage_me --num_me_stages 4 --q_index_num 6
```
- --force_intra_period: 16 (test GOP size (2, 4, 8, 16, ...))
- --q_index_num: 6 (evaluate 6 rate-distortion points within training quantization index range)
- --write_stream: 1 (bitstream writing enabled)

test_pMCTF_CA.py can be called using the same parameters as above for a content-adaptive GOP choice, where _force_intra_period_ is the maximum GOP size.
### Models
Download pretrained models here: [Google Drive](https://drive.google.com/drive/folders/1-Opac8I7bH5JZfXRsXzYbyhovQ5mTXtj?usp=drive_link)
## Acknowledgement 

- The implementation is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [DCVC-DC](https://github.com/microsoft/DCVC/tree/main/DCVC-DC). <br>
- The pretrained SpyNet model is from Simon Niklaus' re-implementation [pytorch-spynet](https://github.com/sniklaus/pytorch-spynet).
## Citation

If you use this project, please cite the relevant original publications for the
models and datasets, and cite this project as:

```bash
@InProceedings{Meyer2024,
	title={Variable Rate Learned Wavelet Video Coding using Temporal Layer Adaptivity},
	author={Anna Meyer and Andr{\'e} Kaup},
	year={2024},
	journal={arXiv preprint arXiv:2410.15873},
}
```


