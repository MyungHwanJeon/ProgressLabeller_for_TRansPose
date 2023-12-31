FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
LABEL Author="Myung-Hwan Jeon"
LABEL Title="Progresslabeller for TRansPose dataset"

# Environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PATH "$PATH:/bin/2.93/python/bin/"
ENV BLENDER_PATH "/bin/2.93"
ENV BLENDERPIP "/bin/2.93/python/bin/pip3"
ENV BLENDERPY "/bin/2.93/python/bin/python3.9"
ENV HW="GPU"
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH

# Install dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
	vim \ 
	gedit \ 	
	wget \ 
	libopenexr-dev \ 
	bzip2 \ 
	build-essential \ 
	zlib1g-dev \ 
	libxmu-dev \ 
	libxi-dev \ 
	libxxf86vm-dev \ 
	libfontconfig1 \ 
	libxrender1 \ 
	libgl1-mesa-glx \ 
	freeglut3-dev \
	xz-utils \
	tar \
	curl \
	git \
	cmake build-essential \
	pkg-config \
	libjpeg-dev libtiff5-dev libpng-dev \
	ffmpeg libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
	libv4l-dev v4l-utils \
	libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev  \
	libgtk2.0-dev \
	mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev \
	libatlas-base-dev gfortran libeigen3-dev \
	python3-dev python3-numpy \
	zip \
    libboost-all-dev libglew-dev libssl-dev \
    lsb \
    ninja-build gcc g++ 

## Install RealSense2
RUN mkdir -p /etc/apt/keyrings \
    && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null \
    && echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | tee /etc/apt/sources.list.d/librealsense.list \
    && apt-get update \
    && apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg
    
## Install OpenCV-4.4.0
RUN mkdir /workspace \
    && mkdir /workspace/opencv \
    && cd /workspace/opencv \
    && wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip \
    && unzip opencv.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip \
    && unzip opencv_contrib.zip \
    && cd opencv-4.4.0 \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=OFF -D WITH_IPP=OFF -D WITH_1394=OFF -D BUILD_WITH_DEBUG_INFO=OFF -D BUILD_DOCS=OFF -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=OFF -D BUILD_PACKAGE=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D WITH_QT=OFF -D WITH_GTK=ON -D WITH_OPENGL=ON -D BUILD_opencv_python3=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.4.0/modules -D WITH_V4L=ON  -D WITH_FFMPEG=ON -D WITH_XINE=ON -D OPENCV_ENABLE_NONFREE=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D OPENCV_SKIP_PYTHON_LOADER=ON -D OPENCV_GENERATE_PKGCONFIG=ON ../ \
    && time make -j$(nproc) \
    && make install \
    && ldconfig        
    
# Install ORB-SLAM3    
RUN cd /workspace \
    && git clone https://github.com/ZerenYu/ORB_SLAM3.git \
    && cd ORB_SLAM3 \
    && chmod +x build.sh \
    && ./build.sh

# Download and install Blender
RUN cd /workspace \
    && wget https://mirror.clarkson.edu/blender/release/Blender2.93/blender-2.93.0-linux-x64.tar.xz \
	&& tar -xvf blender-2.93.0-linux-x64.tar.xz --strip-components=1 -C /bin
ENV PROGRESSLABELLER_BLENDER_PATH "/bin"

# Download the Python source since it is not bundled with Blender
RUN cd /workspace \
    && wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz \
	&& tar -xzf Python-3.9.0.tgz \
	&& cp -r Python-3.9.0/Include/* $BLENDER_PATH/python/include/python3.9/

# Blender comes with a super outdated version of numpy (which is needed for matplotlib / opencv) so override it with a modern one
RUN rm -rf ${BLENDER_PATH}/python/lib/python3.9/site-packages/numpy

# Must first ensurepip to install Blender pip3 and then new numpy
RUN ${BLENDERPY} -m ensurepip && ${BLENDERPIP} install --upgrade pip

## Install Python Packages
RUN ${BLENDERPIP} install \
pyglet==1.5.27 \
numpy==1.23.5 \
open3d==0.15.2 \
Pillow==9.2.0  \
scipy==1.8.0  \
pyyaml==6.0  \
tqdm==4.64.1  \
trimesh==3.15.2  \
pyrender==0.1.45  \
pyntcloud==0.3.1 \
numba==0.56.2 \
pycuda==2022.1 \
pybind11==2.10.0 \
scikit-image==0.18.3 \
opencv-python==4.6.0.66 \
opencv-contrib-python==4.6.0.66 \
PyOpenGL==3.1.0 \
PyOpenGL-accelerate==3.1.7 \
setuptools==59.8.0 \    
matplotlib==3.7.1 \
imageio==2.27.0 \
plotly==5.13.1 \
pybind11-global==2.10.1 \
--target $PROGRESSLABELLER_BLENDER_PATH/2.93/python/lib/python3.9/site-packages

## Install Pytorch
RUN $PROGRESSLABELLER_BLENDER_PATH/2.93/python/bin/python3.9 -m pip install  torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118 --target $PROGRESSLABELLER_BLENDER_PATH/2.93/python/lib/python3.9/site-packages 

## Install Pytorch3d
RUN cd /workspace \
    && git clone https://github.com/facebookresearch/pytorch3d.git \
    && cd /workspace/pytorch3d \
    && git checkout v0.7.2 \
    && FORCE_CUDA=1 ${BLENDERPIP} install -e .

## Install ProgressLabeller
RUN cd /workspace \
    && git clone https://github.com/MyungHwanJeon/ProgressLabeller.git    
ENV PROGRESSLABELLER_PATH "/workspace/ProgressLabeller"

## Install ProgressLabeller for ORB-SLAM3
ENV ORB3_SOURCE_DIR "/workspace/ORB_SLAM3"
RUN cd $PROGRESSLABELLER_PATH/kernel/orb_slam3 \
    && tar -xf ../orb_slam/ORBvoc.txt.tar.gz \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make
    
## Install ProgressLabeller for TRansPose    
RUN cd /workspace \
    && git clone https://github.com/MyungHwanJeon/ProgressLabeller_for_TRansPose.git
ENV PROGRESSLABELLER_TRANSPOSE_PATH "/workspace/ProgressLabeller_for_TRansPose"

# Set the working directory
WORKDIR ${PROGRESSLABELLER_TRANSPOSE_PATH}
