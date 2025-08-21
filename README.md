# FFHQ-Realign

This repository provides **a preprocessing pipeline for the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset)**,  
where **face realignment** is applied to improve the consistency of in-the-wild images.  

It is built on top of the [DECA](https://github.com/yfeng95/DECA) framework,  
with the following main modifications:

- `preprocess_ffhq.py`: restructured to function as a **dataset preprocessing script**.  
- `decalib/utils/face_alignment.py`: a newly added module providing **face alignment utilities**.  

The goal of this project is to **re-estimate bounding boxes and re-crop images based on neutral facial landmarks**,  
producing a more consistent alignment of unconstrained face images.


## Installation

* Python 3.7 (numpy, skimage, scipy, opencv)  
* PyTorch >= 1.6 (pytorch3d)  
* face-alignment (Optional for detecting face)  
  You can run 
  ```bash
  pip install -r requirements.txt
  ```
  For visualization, we use our rasterizer that uses pytorch JIT Compiling Extensions. If there occurs a compiling error, you can install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) instead and set --rasterizer_type=pytorch3d when running the preprocessing script.


## Data Preparation

Before you continue, you must register at [FLAME](https://flame.is.tue.mpg.de/) and agree to the license.

### 1. Download FLAME data
```bash
mkdir -p ./data

# Enter your FLAME username/password
USERNAME="<your_username>"
PASSWORD="<your_password>"

wget --post-data "username=$USERNAME&password=$PASSWORD" \
  "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1" \
  -O ./data/FLAME2020.zip --no-check-certificate --continue

unzip -o ./data/FLAME2020.zip -d ./data/FLAME2020
mv ./data/FLAME2020/generic_model.pkl ./data
```

### 2. Download DECA model
```bash
pip install gdown
gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O ./data/deca_model.tar
```


## Usage
```bash
python preprocess_ffhq.py -i ffhq/in-the-wild-images -s ffhq/realigned --sample_size 1024
``` 


## License

This repository follows the [DECA License](https://github.com/yfeng95/DECA/tree/master?tab=readme-ov-file#license).


## Acknowledgements

This repo is based on DECA, please cite the original paper if you use this code:
```
@inproceedings{DECA:Siggraph2021,
  title={Learning an Animatable Detailed {3D} Face Model from In-The-Wild Images},
  author={Feng, Yao and Feng, Haiwen and Black, Michael J. and Bolkart, Timo},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH)}, 
  volume = {40}, 
  number = {8}, 
  year = {2021}, 
  url = {https://doi.org/10.1145/3450626.3459936} 
}
```
