FROM python:3.7

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
