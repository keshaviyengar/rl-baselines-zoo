FROM ros:melodic-ros-core

RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1 openssh-server
ENV CODE_DIR /root/code
ENV VENV $CODE_DIR/venv

ADD stable-baselines/ $CODE_DIR/stable-baselines/
ADD pypcd/ $CODE_DIR/pypcd/
ADD gym-ctr-ros/ $CODE_DIR/gym-ctr-ros/

RUN \
    pip install virtualenv && \
    virtualenv $VENV --python=python3 && \
    . $VENV/bin/activate && \
    cd $CODE_DIR && \
    pip install --upgrade pip && \
    pip install codacy-coverage && \
    pip install scipy && \
    pip install tqdm && \
    pip install joblib && \
    pip install zmq && \
    pip install dill && \
    pip install progressbar2 && \
    pip install mpi4py && \
    pip install cloudpickle && \
    pip install tensorflow==1.5.0 && \
    pip install click && \
    pip install opencv-python && \
    pip install numpy && \
    pip install pandas && \
    pip install pytest==3.5.1 && \
    pip install pytest-cov && \
    pip install pytest-env && \
    pip install pytest-xdist && \
    pip install matplotlib && \
    pip install seaborn && \
    pip install glob2 && \
    pip install gym[atari,classic_control]>=0.12.4 && \
    pip install python-gnupg && \
    pip install pycrypto && \
    pip install pybullet && \
    pip install optuna && \
    pip install rospkg && \
    pip install -e stable-baselines/ && \
    pip install -e pypcd/ && \
    pip install -e gym-ctr-ros/


ENV PATH=$VENV/bin:$PATH

ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

CMD /bin/bash
