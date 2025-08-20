# For Running NVIDIA FLARE in a Docker container, see
# https://nvflare.readthedocs.io/en/main/quickstart.html#containerized-deployment-with-docker
# This Dockerfile is primarily for building Docker images to publish for dashboard.

FROM python:3.8
RUN pip install -U pip
RUN pip install nvflare
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY dfanalyzer/dfa-lib-python dfa-lib-python
RUN cd dfa-lib-python && make install

