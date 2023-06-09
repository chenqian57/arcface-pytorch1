## Arcface：人脸识别模型在Pytorch当中的实现
---

## 目录
1. [仓库更新 Top News](#仓库更新)
3. [性能情况 Performance](#性能情况)
4. [所需环境 Environment](#所需环境)
5. [注意事项 Attention](#注意事项)
6. [文件下载 Download](#文件下载)
7. [预测步骤 How2predict](#预测步骤)
8. [训练步骤 How2train](#训练步骤)
9. [参考资料 Reference](#Reference)

## Top News
**`2022-03`**:**创建仓库，支持不同模型训练，支持大量可调整参数，支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整、新增图片裁剪。**  


## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | accuracy | Validation rate |
| :-----: | :-----: | :------: | :------: | :------: | :------: |
| CASIA-WebFace | [arcface_mobilenet.pth](https://github.com/bubbliiiing/arcface-pytorch/releases/download/v1.0/arcface_mobilenet.pth) | LFW | 112x112 | 99.11% |  0.95033+-0.02152 @ FAR=0.00133 |
| CASIA-WebFace | [arcface_mobilefacenet.pth](https://github.com/bubbliiiing/arcface-pytorch/releases/download/v1.0/arcface_mobilefacenet.pth) | LFW | 112x112 | 98.78% | 0.91100+-0.01745 @ FAR=0.00100 |
| CASIA-WebFace | [arcface_iresnet50.pth](https://github.com/bubbliiiing/arcface-pytorch/releases/download/v1.0/arcface_iresnet50.pth) | LFW | 112x112 | 98.93% | 0.93100+-0.01422 @ FAR=0.00133 |

（arcface_mobilenet的准确度相比其它较高是因为使用了backbone的预训练权重，正在努力调参中。）

## 所需环境
pytorch

## 文件下载
已经训练好的权值可以在百度网盘下载。    
链接: https://pan.baidu.com/s/1ElJlfmMwOGX699MsgLY8qA 提取码: z3rq   


## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，可直接运行predict.py输入：
```python
img\1_001.jpg
img\1_002.jpg
```  
2. 也可以在百度网盘下载权值，放入model_data，修改arcface.py文件的model_path后，输入：
```python
img\1_001.jpg
img\1_002.jpg
```  

## 训练步骤
1. 本文使用如下格式进行训练。
```
|-datasets
    |-people0
        |-123.jpg
        |-234.jpg
    |-people1
        |-345.jpg
        |-456.jpg
    |-...
```  
2. 下载好数据集，将训练用的CASIA-WebFaces数据集以及评估用的LFW数据集，解压后放在根目录。
3. 在训练前利用txt_annotation.py文件生成对应的cls_train.txt。  
4. 利用train.py训练模型，训练前，根据自己的需要选择backbone，model_path和backbone一定要对应。
5. 运行train.py即可开始训练。

## 评估步骤
1. 下载好评估数据集，将评估用的LFW数据集，解压后放在根目录
2. 在eval_LFW.py设置使用的主干特征提取网络和网络权值。
3. 运行eval_LFW.py来进行模型准确率评估。

## Reference
https://github.com/deepinsight/insightface  
https://github.com/timesler/facenet-pytorch   

