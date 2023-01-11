#FROM 172.16.118.137:8081/ai4sim-docker-repo/ai4sim/toronto:latest
#FROM 172.16.118.137:8081/ai4sim-docker-repo/library/ubuntu:20.10
FROM python:3.8-slim
#RUN export http_proxy=http://193.56.47.8:8080
#RUN export https_proxy=http://193.56.47.8:8080
#RUN export no_proxy=yoda,129.183.101.5,172.16.118.13,naboo0,naboo5,nwadmin,172.16.118.137,172.16.118.134
# MAINTAINER Victor Trappler
RUN pip install --no-cache-dir matplotlib pandas
#Update repositor source list
#RUN apt-get update

################## BEGIN INSTALLATION ######################
#Install python basics
#RUN apt-get -y install \
#	build-essential \
#	python-dev \
#	python-setuptools \
#	python-pip

#Install scikit-learn dependancies
#RUN apt-get -y install \
#	python-numpy \
#	python-scipy \
#	libatlas-dev \
#	libatlas3-base

#
#RUN echo "alias pip=pip3" >> /root/.bashrc
#RUN echo "alias ll=ls -lart" >> /root/.bashrc
#RUN pip3 install numpy matplotlib scipy pyDOE cma argparse tqdm pickle
#Install scikit-learn
#RUN pip3 install scikit-learn 

################## END INSTALLATION ########################
