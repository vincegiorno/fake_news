FROM python:3.6-slim-buster

# set the working directory
RUN mkdir /han
WORKDIR /han

# install code dependencies
COPY utils ./utils
COPY cyclic ./cyclic
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY train.py .
RUN chmod +x ./train.py

ENV PATH="/han:${PATH}"
ENTRYPOINT ["./train.py"]