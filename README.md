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
  Or use virtual environment by runing 
  ```bash
  bash install_conda.sh
  ```
  For visualization, we use our rasterizer that uses pytorch JIT Compiling Extensions. If there occurs a compiling error, you can install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) instead and set --rasterizer_type=pytorch3d when running the demos.


## Usage

1. Prepare data   
    run script: 
    ```bash
    bash fetch_data.sh
    ```
    <!-- or manually download data form [FLAME 2020 model](https://flame.is.tue.mpg.de/download.php) and [DECA trained model](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view?usp=sharing), and put them in ./data  -->  
    (Optional for Albedo)   
    follow the instructions for the [Albedo model](https://github.com/TimoBolkart/BFM_to_FLAME) to get 'FLAME_albedo_from_BFM.npz', put it into ./data

2. Run preprocessing
    ```bash
    python preprocess_ffhq.py -i ffhq/in-the-wild-images -s ffhq/realigned --sample_size 512
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
