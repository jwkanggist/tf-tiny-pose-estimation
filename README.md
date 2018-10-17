Readme for  Tiny pose estimation
==================================
- Final update: 2018 Oct 
- All right reserved @ Jaewook Kang 2018


## About
The aim of this repository is to introduce a tiny pose estimation tutorial.
This pose estimation model is based on single hourglass model.
- [Alejandro Newell, Kaiyu Yang, Jia Deng, "
Stacked Hourglass Networks for Human Pose Estimation," ECCV 2016.](https://arxiv.org/abs/1603.06937)
We implement the pose estimation model in [Tensorflow](https://tensorflow.org).

#### Keywords
- Tensorflow
- Single hourglass model
- Human pose estimation
- Inverted bottleneck (Mobilenet v2)


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


### Pycocotools Installation (Only for Win)
> For OSX and Ubuntu, we can install pycocotool by pip
- Step 1) [Download and install `Microsoft Build Tools 2015`](https://www.microsoft.com/ko-kr/download/details.aspx?id=48159)
- Step 2) pip install `cocoapi` repository
- Step 3) Make `pycocotools`
```bash
git clone https://github.com/cocodataset/cocoapi
cd PythonAPI
make
```



## How to Run
1) Downloading dataset
> Downloading dataset from the AI challenger website and place the dataset on `./dataset`.
- [AI challenge dataset link](https://challenger.ai/datasets/)

2) Training
```bash
python ./tf_module/trainer.py
```

3) Monitoring by Tensorboard
```bash
tensorboard --logdir ./export/tf_logs
```

## Components

```bash
./tfmodules/
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

#### Log Data
- `./export/train_setup_log/`: We log and store setup of each training run in this folder. 
- `./export/tf_logs/run-yyyymmddHHmmss/train/`: We log `tensorboard summary` of each training run in this folder.
- `./export/tf_logs/run-yyyymmddHHmmss/valid/`: We log `tensorboard summary` of each validation run in this folder.
- `./export/tf_logs/run-yyyymmddHHmmss/pb_and_ckpt/`: We save `ckpt` and `pb` files resulting from each training run.


## Related Materials
- [Jaewook Kang, Tensorflow Practical Project Configuration](https://docs.google.com/presentation/d/1zyubZQKQ3tQvQppp_7ljPnWXwCNmf3UDMQhP2GBn7ng/edit#slide=id.p1)


## Feedback 
- Issues: report issues, bugs, and request new features
- Pull request
- Email: jwkang10@gmail.com

## License
- Apach License 2.0


## Authors information 
- Jaewook Kang Ph.D.
- Personal website: [link](https://sites.google.com/site/jwkang10/)
- Facebook : [link](https://www.facebook.com/jwkkang)
- Linkedin : [link](https://www.linkedin.com/in/jaewook-kang-3a4217b9/)


