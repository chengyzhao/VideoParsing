tracking codes from ByteTrack [https://github.com/ifzhang/ByteTrack.git]

pretrained model from DanceTrack [https://github.com/DanceTrack/DanceTrack/tree/main/ByteTrack]

## Tracker Installation
### Installing on the host machine
Step1. Install ByteTrack.
```shell
conda create -n dance python=3.7
source activate dance

cd tracker/
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
python setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip install cython_bbox
```

