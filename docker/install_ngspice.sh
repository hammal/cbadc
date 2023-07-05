# Hampus Malmberg, 2023-03-26

# install ngspice dependencies
apt-get update && apt-get install -y \
    bison \
    flex \
    build-essential \
    autoconf \
    automake \
    libtool \
    libxaw7-dev \
    libreadline-dev \
    git

# Retrive the ngspice source code from sourceforge
git clone git://git.code.sf.net/p/ngspice/ngspice
cd ngspice
./autogen.sh
./configure --enable-xspice --enable-cider --disable-debug --with-readline=yes --enable-openmp
cat config.log
make clean
make
make install

cd ..
rm -rf ngspice
