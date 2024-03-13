# pytorch-be-your-own-teacher
An pytorch implementation of paper 'Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation', https://arxiv.org/abs/1905.08094

## Introduction
We provide code of training ResNet18 and ResNet50 with multiple breanches on CIFAR100 dataset. 

## MyUnderstanding
总的损失函数由三部分构成：
+ 标签与各个阶段的分类器中softmax层的输出做损失。

+ 最深层的分类器中的softmax层前的输出logits与其他各个分类器的softmax层的输出做损失。

+ 最深层分类器里的特征图与其他分类器里的特征图做损失，bottleneck用来将浅层的特征图的形状与最深层分类器里的特征图对齐。

本文件提供继续训练的功能：
+ 通过添加 --resume 参数，用户可以在命令行中指定一个检查点文件的路径，用于恢复之前训练的模型状态。如果用户不指定 --resume 参数，则默认值为空字符串。
python train.py train multi_resnet18_kd --resume ./save_checkpoints/multi_resnet18_kd/checkpoint_latest.pth.tar

## MyExperiments
相比于原来文件的改进（原来的文件地址：https://github.com/luanyunteng/pytorch-be-your-own-teacher）：
+ 自己在train.py文件的基础上进行修改，修改后的文件名为train_nodistiller.py:
  此文件不采用知识蒸馏的形式训练resnet模型
+ 将医院数据集搬入实验里,医院数据集的结构大致如下：<br>
  hospital_data <br>
  ├── 1 <br>
  ├── 2 <br>
  ├── 3 <br>
  ├── 4 <br>
  ├── 5 <br>
  ├── 6 <br>
  ├── 7 <br>
  ├── 8 <br>
  首先需要通过split_data.py文件将医院分类好的数据集分为两份，train文件夹和val文件夹的结构大致如下：<br>
  ├── test <br>
│   ├── 1 <br>
│   ├── 2 <br>
│   ├── 3 <br>
│   ├── 4 <br>
│   ├── 5 <br>
│   ├── 6 <br>
│   ├── 7 <br>
│   └── 8 <br>
└── train <br>
    ├── 1 <br>
    ├── 2 <br>
    ├── 3 <br>
    ├── 4 <br>
    ├── 5 <br>
    ├── 6 <br>
    ├── 7 <br>
    └── 8 <br>
+ 实验结果
  只进行了一次实验： <br>
  + 将CIFAR数据集放入resnet18模型里，在只有100epochs的情况下，使用了知识蒸馏的准确率最高达到了77.34%；相比于没有使用知识蒸馏的准确率最高达到了76.46%
  + 将医院数据集放入resnet18模型里，在只有200epochs的情况下，使用了知识蒸馏的准确率最高达到了61.382%
  + 将医院数据集放入resnet50模型里，即在控制台输入python train.py train multi_resnet50_kd，首先报错：
RuntimeError: CUDA out of memory. Tried to allocate 196.00 MiB (GPU 0; 23.65 GiB total capacity; 21.89 GiB already allocated; 2.06 MiB free; 22.21 GiB reserved in total by PyTorch) <br>
后来将batch_size改为8，再将显卡换为autodl的2元/h的，就可以初步运行了，但是运行的时候老是报错： <br>
Traceback (most recent call last): <br>
  File "train.py", line 443, in <module> 
    main()
  File "train.py", line 91, in main
    run_training(args)
  File "train.py", line 200, in run_training
    loss = criterion(output, target)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/loss.py", line 961, in forward
    return F.cross_entropy(input, target, weight=self.weight,
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/functional.py", line 2468, in cross_entropy
    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/functional.py", line 2260, in nll_loss
    if input.size(0) != target.size(0): <br>
IndexError: dimension specified as 0 but tensor has no dimensions <br>
我至今还未解决！
## Dependencies:

+ Ubuntu 18.04.5 LTS
+ Python 3.6.9
+ PyTorch 1.7.1
+ torchvision 0.8.2 
+ numpy 1.19.2 
+ tensorboardX 2.1

Note: this is my machine environment, and the other version of software may also works.

## Train an ResNet on CIFAR-100:

```sh
# resnet18
python train.py train multi_resnet18_kd \
                             --data-dir /PATH/TO/CIFAR100 

# resnet50
python train.py train multi_resnet18_kd \
                             --data-dir /PATH/TO/CIFAR100 
```

## Load an ResNet on CIFAR-100 and test it:
```sh
# resnet18
python train.py test multi_resnet18_kd \
                             --data-dir /PATH/TO/CIFAR100 \
                             --resume /PATH/TO/checkpoint

# resnet50
python train.py test multi_resnet50_kd \
                             --data-dir /PATH/TO/CIFAR100 \
                             --resume /PATH/TO/checkpoint
```

## Result on Resnet18

As the original paper does not tell us the hyper-parameters, I just use the follow setting. If you find better hyper-parmeters, you could tell me in the issues. Moreover, we do not know the side-branch architecture details. So the accuracy is lower than the original paper.
I will fine-tune the hyper-parameters.

21.01.12) At the same settings, with different library version, the performance of last classifier is better than paper result.
However, hyperparameter searching is still needed.

 ```
alpha = 0.1
temperature = 3
beta = 1e-6
```
|   Method   | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 |
|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Original   |    **67.85**       |74.57          |       **78.23**   |     78.64      |
| Ours       | 66.570         | **74.690**        | 78.040        | **78.730**         |

## Result on Resnet50

 ```
alpha = 0.1
temperature = 3
beta = 1e-6
```
|   Method   | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 |
|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Original   |    68.23       |74.21          |       75.23   |     80.56      |
| Ours       |    **71.840**       |   **77.750**     |   **80.140**      |     **80.640**     |
   
