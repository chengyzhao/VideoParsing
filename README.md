# 2022年春计算机网络(实验班)Lab2：多媒体内容生成与传播

小组成员：王梓昱、李浩雨、赵橙阳、林昊苇



在本项目中，我们选择的是“动作类”track，预期目标为：根据输入的多人横屏舞蹈长视频，输出单人直拍短视频片段。

本算法基于华中科技大学、香港大学、字节跳动等机构联合提出的[ByteTrack](https://github.com/ifzhang/ByteTrack)算法，及其在[DanceTrack](https://github.com/DanceTrack/DanceTrack)数据集上的预训练模型（由[DanceTrack](https://github.com/DanceTrack/DanceTrack)开源）。代码在[ByteTrack](https://github.com/ifzhang/ByteTrack)的基础上进行修改和扩充。



## 环境配置及Track安装

以下流程在Linux上测试成功。

Step1. 安装[ByteTrack](https://github.com/ifzhang/ByteTrack).

```shell
conda create -n dance python=3.7
source activate dance

cd tracker/
pip install -r requirements.txt
python setup.py develop
```

Step2. 安装依赖包[pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. 安装依赖包[cython_bbox](https://github.com/samson-wang/cython_bbox).
```shell
pip install cython_bbox
```



## 下载预训练模型

预训练模型由[DanceTrack](https://github.com/DanceTrack/DanceTrack)开源，下载[bytetrack_model.pth.tar](https://drive.google.com/drive/u/0/folders/1v1hiIrgH0b5-ZavRAVa1MJS8iGr-Pn5U)，并将其移动到./pretrained/文件夹下。



## 代码运行
在./config.yaml文件中修改对应的配置项，之后运行：

```shell
python segment.py
```



## Acknowledgment

本算法的大部分代码借鉴了[ByteTrack](https://github.com/ifzhang/ByteTrack)，[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)，[FairMOT](https://github.com/ifzhang/FairMOT)，[TransTrack](https://github.com/PeizeSun/TransTrack)和 [JDE-Cpp](https://github.com/samylee/Towards-Realtime-MOT-Cpp)的已有工作，预训练模型采用了[DanceTrack](https://github.com/DanceTrack/DanceTrack)的开源模型。正是在上述优秀工作的基础上，我们的算法效果才能有较为不错的表现。在此对上述精彩且具有启发意义的工作及其作者表示衷心的敬意与感谢。


