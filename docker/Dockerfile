 
From nvidia/cuda:11.1.0-devel-ubuntu20.04

LABEL author="unknownue <unknownue@outlook.com>" version="1.0"
 
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

ARG USER_ID=1000
ARG GROUP_ID=1001
ARG DOCKER_USER=unknownue
ARG DOCKER_PASSWORD=password

ENV PATH="/home/$DOCKER_USER/.local/bin:${PATH}"

# ADD mirror-ubuntu2004.txt /etc/apt/sources.list

# base installation
RUN apt update && \
    apt install -y ca-certificates curl sudo wget git && \
    apt install -y python3 python3-pip && \
    apt install -y python3-tk && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt autoremove && apt clean && \
    rm -r /var/lib/apt/lists/*

# Docker user -------------------------------------------------------------------
# See also http://gbraad.nl/blog/non-root/user/inside-a-docker-container.html
RUN adduser --disabled-password --gecos '' $DOCKER_USER && \
    adduser $DOCKER_USER sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER $DOCKER_USER

# Packages -----------------------------------------------------------------
RUN pip install --upgrade pip
# RUN pip install --no-cache-dir --user pqi && pqi use aliyun

# Install from stable build
RUN pip install --no-cache-dir --user torch==1.8.0 torchvision==0.9.0

RUN pip install --no-cache-dir --user pytorch_lightning==1.2.2 tensorboardX h5py imageio matplotlib ipython omegaconf
RUN pip install --no-cache-dir --user open3d==0.10.0 && sudo apt update && sudo apt install -y libgl1-mesa-glx

# Install some special package
RUN pip install --user --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
RUN sudo wget -P /usr/bin https://github.com/unlimblue/KNN_CUDA/raw/master/ninja && sudo chmod +x /usr/bin/ninja
RUN pip install --user "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# PytorchEMD
# RUN export FORCE_CUDA="1" && pip install --user "git+git://github.com/daerduoCarey/PyTorchEMD.git"
# RUN cd metric/PytorchEMD/ && python setup.py install --user && cp build/lib.linux-x86_64-3.6/emd_cuda.cpython-36m-x86_64-linux-gnu.so .

# Powerline helps to distinguish command and program output
RUN sudo wget https://raw.githubusercontent.com/bcmarinacci/powerline-shell/master/powerline-shell.bash && \
    sudo mv powerline-shell.bash ~/.powerline-shell.bash && \
    sudo printf "source ~/.powerline-shell.bash\n" >> ~/.bashrc

# CMD ["bash"]
RUN python -m torch.utils.collect_env
