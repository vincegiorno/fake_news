# Build an image that can do inference in SageMaker
# This uses the nginx, gunicorn, flask stack

FROM python:3.6-slim-buster

RUN apt-get -y update && apt-get install -y --no-install-recommends \
        nginx \
        ca-certificates \
        gcc \
    && rm -rf /var/lib/apt/lists/*
COPY requirements_predict.txt .
RUN pip3 install -r requirements_predict.txt
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
#RUN (cd /usr/local/lib/python3.6/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
    #rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
WORKDIR /opt/program
COPY prediction .
RUN python -m spacy download en_core_web_sm
RUN chmod +x ./serve.py
RUN chmod +x ./predict.py

ENTRYPOINT ["python", "./serve.py"]