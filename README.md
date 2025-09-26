# 飞行器视觉技术课程作业（此次课程自动布置,不可能不做）

## 任务描述

给定图片

<img src="assets/test.png" alt="img" width="500" height="400">

1. 图像灰度化、图像二值化处理、图像进行傅里叶变换
2. 图像去噪（背景为黑，小丑、气球、牵引绳标记为白色）
3. 使用自拟尺寸的滑框，计算自拟图像的局部均值、局部一阶矩、局部二阶矩、局信息熵，将计算值归一化到0-255的整数，输出灰度图像
4. 计算图像的LBP特征、Hog特征、harr特征
5. 解救小丑
6. 解开牵引绳

## 实现
使用opencv-python库实现图像处理，pyside6实现图形界面
## 环境搭建

下载/fork仓库，进入目录

```powershell
git clone ....
#进入项目文件
cd clown-vision
```

**uv方式**

```powershell
#创建虚拟环境
uv venv --python 3.12
#激活环境
.venv\Scripts\activate
#安装依赖
uv add -r requirements.txt
```

**pip+venv方式**

```powershell
#创建虚拟环境
python -m venv .venv
#激活环境
.venv\Scripts\activate
#安装依赖
pip install -r requirements.txt
```

**conda方式**

```powershell
#创建环境
conda create -n [env_name] python=3.12
#激活环境
conda activate [env_name] 
#安装依赖
conda install -n [env_name] -c conda-forge [package_name]
```



项目框架（clown_vision下包含图像处理函数、ui）

```ruby
clown-vision/
├── clown_vision/
│   ├── __init__.py
│   ├── preprocessing.py       # 灰度化、二值化、傅里叶变换
│   ├── denoising.py           # 去噪
│   ├── local_features.py      # 局部均值/矩/熵计算
│   ├── descriptors.py         # LBP/HOG/Haar 特征
│   ├── rescue.py              # 解救小丑（只保留气球和线）
│   ├── untangle.py            # 解开三条线并着色
│   ├── ui.py                  # PySide6 UI界面
│   └── utils.py               # 工具函数（显示、保存等）
├── assets/
│   └── test.png               # 测试图像
├── tests/
│   └── test_processing.py     # 测试文件
├── scripts/
│   └── run_ui.py              # 运行UI的脚本
├── pyproject.toml            # 项目配置和依赖管理
├── requirements.txt          # 依赖列表
└── README.md                 # 项目说明文档
```
