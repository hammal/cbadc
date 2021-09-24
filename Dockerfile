FROM gitpod/workspace-full

RUN sudo apt update && sudo apt upgrade -y
RUN sudo apt install python3.9 -y
RUN python -m pip install --upgrade pip
