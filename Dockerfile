FROM python:3.9

ADD . /cbadc

WORKDIR /cbadc

RUN python -m pip install --upgrade pip
RUN python -m pip install . 
RUN python -m pip install -r docs/requirements.txt; python -m pip install -r requirements.txt