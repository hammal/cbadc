FROM gitpod/workspace-full

ADD . /cbadc

WORKDIR /cbadc

RUN sudo apt update && sudo apt upgrade -y
RUN sudo apt install python3.9 
RUN python -m pip install --upgrade pip
RUN python -m pip install -r docs/requirements.txt; python -m pip install -r requirements.txts
