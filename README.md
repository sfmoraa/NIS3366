运行方式参考scripts内文件夹。

# 环境配置

1. 本工具需要在有英伟达显卡和驱动的设备上运行
2. 安装cuda,cudnn，其中cuda中需包含nvidia compute和nvidia system工具
3. 安装python环境 `conda env create -f environment.yml`


# 具体应用
## Sarcasm

需要从huggingface上下载bert-base-uncased的model.onnx、pytorch_model.bin和tf_model.h5三个文件，放入applications/Sarcasm/bert-base-uncased文价夹中

然后运行

```
bash ./scripts/ncu_metric.sh applications/Sarcasm/inference.py applications/Sarcasm/ncu.csv --option normal

bash ./scripts/nsys_metric.sh applications/Sarcasm/inference.py --options fusion
```

## Avmnist

从  [raw avmnist dataset](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing)下载数据集，然后解压缩到applications/Avmnist/ 目录下

其余同上


## Medical-Segmentation

下载原始数据集[dataset](https://www.med.upenn.edu/sbia/brats2018/data.html)，然后修改applicaions/Medical-Segmentation/inference.py 中的 "src_path = " 为其路径。

其余同上

## Medical-VQA

下载原始数据集 [raw medica_vqa dataset](https://zenodo.org/record/6784358)，解压后修改
applications/Medical-VQA/config/idrid_regions/single/default_baseline.yaml 中的 path路径为对应路径。

其余同上

## MUjoCo-Push

下载数据集 [gentle_push_10.hdf5](https://drive.google.com/file/d/1qmBCfsAGu8eew-CQFmV1svodl9VJa6fX/view) , [gentle_push_300.hdf5](https://drive.google.com/file/d/18dr1z0N__yFiP_DAKxy-Hs9Vy_AsaW6Q/view) , [gentle_push_1000.hdf5](https://drive.google.com/file/d/1JTgmq1KPRK9HYi8BgvljKg5MPqT_N4cR/view),
修改datasets/gentle_push/data_loader.py 中的 145行 return _load_trajectories($PATH, **dataset_args)

其余同上


