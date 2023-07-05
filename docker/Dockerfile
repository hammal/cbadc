# A dockerfile describing a container with all necessary dependencies
# to use the full cbadc feature suit.

# Use the official python image as a parent image
FROM ubuntu:latest

# Install dependencies and update
RUN apt-get update && apt-get upgrade; \
    apt-get install \
    git \
    curl \
    libx11-dev \
    python3.10 \
    python3-pip \
    libqt5core5a \
    libxft2 \
    libxss1 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxaw7 \
    libcairo2 \
    qtbase5-dev \
    build-essential \
    libxaw7-dev \
    automake \
    libtool \
    gperf \
    flex \
    bison \
    libxml2 \
    libxml2-dev \
    libxml-libxml-perl \
    libgd-perl \
    libreadline-dev \
    autotools-dev \
    autoconf \
    texinfo \
    # readline \
    # editline \
    # tcl/tk \
    gawk \
    adms \
    xorg \
    xserver-xorg-input-evdev  \
    xserver-xorg-input-all \
    openssh-server \
    vim \
    # xserver \
    # ssh \
    -y; \
    apt-get autoremove -y; \
    apt-get clean;

# Add cbadc user
RUN useradd -ms /bin/bash cbadc
# make sudo user
RUN echo "cbadc:cbadc" | chpasswd && adduser cbadc sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN echo  "X11UseLocalhost no\n" >> /etc/ssh/sshd_config

# expose ssh port
EXPOSE 22

# nointeractive
ARG DEBIAN_FRONTEND=noninteractive

# CBADC
WORKDIR /ngspice
# USER cbadc
# ADD ./cbadc /cbadc/
# RUN python -m pip install .;

# Install ADMS compiler
RUN git clone https://github.com/Qucs/ADMS.git; \
    cd ADMS; \
    sh bootstrap.sh; \
    ./configure; \
    make install;
# Install NGSpice
RUN git clone https://github.com/ngspice/ngspice.git; \
    cd ngspice; \
    ./autogen.sh --adms; \
    ./configure --with-x --enable-xspice --disable-debug --with-readline=yes --enable-openmp; \
    # # ./configure --with-x --enable-xspice --disable-debug --enable-cider --with-readline=yes --enable-openmp; \
    make clean; \
    make; \
    make install;

ADD . /cbadc

RUN

# USER cbadc
# RUN mkdir /home/cbadc/.ssh

# ADD CMOS_inverter.cir ./

# ENTRYPOINT sudo service ssh restart && tail -f /dev/null
