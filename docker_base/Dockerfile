FROM continuumio/miniconda3
ADD requirements.txt /requirements.txt
RUN conda install python=3.8
RUN conda install --file requirements.txt
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
RUN pip install fbpca
