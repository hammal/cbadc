# install dependencies
apt update && apt install -y \
    git \
    curl \
    build-essential \
    bison \
    flex  \
    autoconf \
    automake \
    libtool \
    libx11-dev \
    libxaw7-dev \
    libxmu-dev \
    libxi-dev \
    libxcursor-dev \
    libxext-dev \
    libxft-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxpm-dev \
    libxrender-dev \
    libxt-dev \
    libxv-dev \
    libxss-dev \
    libxtst-dev \
    libxkbfile-dev \
    libxkbcommon-d \
    adms \
    xorg \
    fftw-3 \
    xserver-xorg-input-evdev  \
    xserver-xorg-input-all \
apt autoremove -y
apt clean -y
# Install ngspice
git clone git://git.code.sf.net/p/ngspice/ngspice
cd ngspice
./compile_linux.sh
cd ..
rm -rf ngspice
