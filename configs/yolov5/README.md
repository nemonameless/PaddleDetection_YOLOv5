# YOLOv5

## Model Zoo
### YOLOv5 on COCO

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 |推理时间(fps) |   mAP  |   AP50  |   下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :---------: | :-----: |:-----: | :-------------: | :-----: |
| YOLOv5-n        |  640     |    8      |   300e    |     ----    |  28.0  | 45.7 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_n_300e_coco.pdparams) | [配置文件](./yolov5_n_300e_coco.yml) |
| YOLOv5-s        |  640     |    8      |   300e    |     ----    |  37.0  | 55.9 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams) | [配置文件](./yolov5_s_300e_coco.yml) |
| YOLOv5-m        |  640     |    8      |   300e    |     ----    |  45.3  | 63.8 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_m_300e_coco.pdparams) | [配置文件](./yolov5_m_300e_coco.yml) |
| YOLOv5-l        |  640     |    8      |   300e    |     ----    |  48.6  | 66.9 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_l_300e_coco.pdparams) | [配置文件](./yolov5_l_300e_coco.yml) |
| YOLOv5-x        |  640     |    8      |   300e    |     ----    |  50.6  | 68.7 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_x_300e_coco.pdparams) | [配置文件](./yolov5_x_300e_coco.yml) |

**注意:**
  - YOLOv5模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
  - YOLOv5模型训练过程中默认使用8 GPUs进行混合精度训练，默认单卡batch_size为8，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率；


## 使用教程

### 1. 训练
执行以下指令使用混合精度训练YOLOv5
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolov5/yolov5_s_300e_coco.yml --amp --eval
```
**注意:**
`--amp`表示混合精度FP16训练，`--eval`表示边训边验证。

### 2. 评估
执行以下命令在单个GPU上评估COCO val2017数据集
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams
```

### 3. 推理
使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。
```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams --infer_dir=demo
```

### 4. 部署

#### 4.1. 导出模型
YOLOv5在GPU上推理部署或benchmark测速等需要通过`tools/export_model.py`导出模型。
运行以下的命令进行导出：
```bash
python tools/export_model.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams
```

#### 4.2. Python部署
`deploy/python/infer.py`使用上述导出后的Paddle Inference模型用于推理和benchnark测速，如果设置了`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

```bash
# Python部署推理单张图片
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu

# 推理文件夹下的所有图片
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_dir=demo/ --device=gpu

# benchmark测速
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True

# tensorRT-FP32测速
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --trt_max_shape=640 --trt_min_shape=640 --trt_opt_shape=640 --run_mode=trt_fp32

# tensorRT-FP16测速
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --trt_max_shape=640 --trt_min_shape=640 --trt_opt_shape=640 --run_mode=trt_fp16
```

#### 4.2. C++部署
`deploy/cpp/build/main`使用上述导出后的Paddle Inference模型用于C++推理部署, 首先按照[docs](../../deploy/cpp/docs)编译安装环境。
```bash
# C++部署推理单张图片
./deploy/cpp/build/main --model_dir=output_inference/yolov5_s_300e_coco/ --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=GPU --threshold=0.5 --output_dir=cpp_infer_output/yolov5_s_300e_coco
```
