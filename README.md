# Event-based Image Deblurring with Dynamic Motion Awareness

This repository is about the [**Event-based Image Deblurring with Dynamic Motion Awareness**](https://arxiv.org/abs/2208.11398) 

by Patricia Vitoria, Stamatios Georgoulis, Stepan Tulyakov, Alfredo Bochicchio, Julius Erbach and  Yuanyou Li.

[<img width="900" src="figures/resultsGOPRO.png?raw=true">](figures/resultsGOPRO.png?raw=true)



## Prerequisits 
Linux 

Python 3.8.8

Basicsr


## Getting Started



### Requirements
```
conda create -n env python=3.8.8
conda activate env
pip install -r requirements.txt
```


  ### Testing
To test the network you can either run the code
```
python demo.py
```
Images are saved to `./results/`


You can change the results directory with 

```
python demo.py --results_dir DIR_TO_SAVE_RESULTS
```

### Checkpoints

Checkpoitns are [available here](https://drive.google.com/drive/folders/11ozMUSctpSfxJcKgwn7cT_tL-yWIcZV3?usp=drive_link).

### Citation
A pdf of the paper is [available here](https://arxiv.org/pdf/2208.11398.pdf). If you this code please cite this publication as follows:

```bibtex
@article{vitoria2022event,
  title={Event-based Image Deblurring with Dynamic Motion Awareness},
  author={Vitoria, Patricia and Georgoulis, Stamatios and Tulyakov, Stepan and Bochicchio, Alfredo and Erbach, Julius and Li, Yuanyou},
  journal={arXiv preprint arXiv:2208.11398},
  year={2022}
}

```

### License
This project is under the Apache 2.0 license, and it is based on [BasicSR](https://github.com/xinntao/BasicSR) which is under the Apache 2.0 license. 
