Readme for  Tiny pose estimation
==================================
- Final update: 2018 Oct 
- All right reserved @ Jaewook Kang 2018


## About
The aim of this repository is to introduce an exemplary TF project in practice.
We show the example with respect to [MNIST](http://yann.lecun.com/exdb/mnist/) classification by Lenet5.
This project is composed of several `.py` files,
each of which undertakes single role and responsibility 
according to the OOP philosophy.

```bash
- data_loader.py    : Preparing and feeding the dataset in batchwise by using tf.data
- model_builder.py  : Building a model in tensorflow computational graph.
- model_config.py   : Specifying a configulation for the model 
- trainer.py        : Training the model by importing the dataloader and the model_builer
- train_config.py   : Including a configulation for the training
- eval.py           : Evaluating the model with respect to test dataset by loading a ckpt

```


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
├── data
│   └── mnist
├── export
│   └── tf_logs
├── data_loader.py
├── eval.py
├── model_builder.py
├── model_config.py
├── testcodes
│   └── test_dataloader.py
├── train_config.py
└── trainer.py
```

### Compiler/Interface Dependencies
- Tensorflow >=1.9
- Python2 <= 2.7.12
- Python3 <= 3.6.0


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


