FROM continuumio/miniconda3
RUN conda install --file requirements.txt
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
RUN pip install fbpca
