# tensorflow 2.0
    
	this repo is for learning how to use tensorflow 2.0 

# 在windows系统上安装tensorflow 

## 先检查以下前置条件是否满足：

Check your Windows system, should be Windows 7 or later, and 64-bit operating systems.

    TensorFlow was built and tested on 64-bit laptop/desktop operating system.

Download and install the version 3.6 of Python (建议使用 3.6 的版本) for Windows.

    Should select to download 64bit version of Python

# 安装tensorflow 2.0 的 GPU 版本

## 安装CUDA 10.0:

先检查自己CUDA版本：

	>>> nvcc --version
	
如果不是10.0的版本，则需要先下载并安装好。

下载安装好CUDA10.0后，有可能发现系统找不到了显卡，这时候重新安装显卡的驱动即可。

## 安装CuDNN 7.6.0:

如果CuDNN 的版本不是7.6.0，在运行时可能会碰到下面的问题：

    Loaded runtime CuDNN library: 7.4.1 but source was compiled with: 7.6.0.  
    CuDNN library major and minor version needs to match or have higher minor version in case of CuDNN 7.0 or later version. 
	
去下面的地址下载正确的版本：

	https://developer.nvidia.com/rdp/cudnn-archive
	
然后只需要把下载后的压缩文件解压缩，分别将cuda/include、cuda/lib、cuda/bin三个目录中的内容拷贝到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0 对应的include、lib、bin目录下即可。


## 安装tensorflow-gpu：

	pip install --upgrade tensorflow-gpu

## 可能碰到的问题：

安装Tensorflow–GPU版本时如果一直出现如下问题:

	“ Could not load dynamic library ‘cudart64_100.dll’; dlerror: cudart64_100.dll not found”
	
则需要检查和安装CUDA10.0 & CuDNN7.6.0
