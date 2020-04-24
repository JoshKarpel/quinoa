FROM continuumio/miniconda3:latest
ENV PATH=/opt/conda/bin/:${PATH}

RUN conda install -y numpy scipy matplotlib opencv scikit-learn scikit-image

COPY . /tmp/quinoa
RUN pip install --no-cache-dir htmap \
 && pip install --no-cache-dir /tmp/quinoa

ARG USER=htmap
RUN groupadd ${USER} \
 && useradd -m -g ${USER} ${USER}
USER ${USER}
