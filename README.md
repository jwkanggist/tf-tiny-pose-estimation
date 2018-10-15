Readme for  Tiny pose estimation
==================================
- Final update: 2018 Oct 
- All right reserved @ Jaewook Kang 2018


## About
The aim of this repository is to introduce a tiny pose estimation in Tensorflow.


```bash
- data_loader.py    : Preparing and feeding the dataset in batchwise by using tf.data
- model_builder.py  : Building a model in tensorflow computational graph.
- model_config.py   : Specifying a configulation for the model 
- trainer.py        : Training the model by importing the dataloader and the model_builer
- train_config.py   : Including a configulation for the training
- eval.py           : Evaluating the model with respect to test dataset by loading a ckpt

```

## Installation

### Compiler/Interface Dependencies
- Tensorflow >=1.9
- Python2 <= 2.7.12
- Python3 <= 3.6.0
- opencv-python >= 3.4.2
- pycocotools   == 2.0.0
- Cython        == 0.28.4
- tensorpack    == 0.8.0
- Tf plot       == 0.2.0.dev0 



### Git Clone
```bash
git clone https://github.com/jwkanggist/tf-tiny-pose-estimation
# cd tf-tiny-pose-estimation/
git init

pip install -r requirement.txt
./sh_scripts/install_tensorflow_gpu.sh
```


### Pycocotools (WIN)
> For OSX and Ubuntu, we can install pycocotool by pip


* MFC++
- https://blog.naver.com/swkim4610/221335020498
 
* Pycocotools
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
related git: https://github.com/philferriere/cocoapi




## How to Run
Training
```bash
python ./tf_module/trainer.py
```

Inference
```bash
python ./tfmodule/eval.py
```

## Components

```bash
./tfmodule/
├── dataset
│   └── ai_challenger
│
├── export
│   ├── train_setup_log
│   └── tf_logs
│
├── coco_dataload_modules
│   ├── testcodes
│   │   └── test_dataloader.py
│   │
│   ├── dataset_augment.py
│   └── dataset_prepare.py
│
├── data_loader.py
├── eval.py
├── model_builder.py
├── model_config.py
├── train_config.py
└── trainer.py
```


### Related Materials
- [Jaewook Kang, Tensorflow Practical Project Configuration](https://docs.google.com/presentation/d/1zyubZQKQ3tQvQppp_7ljPnWXwCNmf3UDMQhP2GBn7ng/edit#slide=id.p1)


# Feedback 
- Issues: report issues, bugs, and request new features
- Pull request
- Email: jwkang10@gmail.com

# License
- Apach License 2.0


# Authors information 
- Jaewook Kang Ph.D.
- Personal website: [link](https://sites.google.com/site/jwkang10/)
- Facebook : [link](https://www.facebook.com/jwkkang)
- Linkedin : [link](https://www.linkedin.com/in/jaewook-kang-3a4217b9/)


