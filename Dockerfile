FROM python:3.9

ADD . /cbadc

WORKDIR /cbadc

RUN python -m pip install --upgrade pip . -r docs/requirements.txt