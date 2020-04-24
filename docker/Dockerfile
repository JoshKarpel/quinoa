FROM continuumio/miniconda3:latest
ENV PATH=/opt/conda/bin/:${PATH}

RUN apt-get update \
 && apt-get install -y libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/* \
 && conda install -c conda-forge -y numpy scipy matplotlib opencv opencv=4.2 scikit-learn scikit-image

COPY . /tmp/quinoa
RUN pip install --no-cache-dir htmap \
 && pip install --no-cache-dir /tmp/quinoa \
 && python -c "import quinoa"

ARG USER=htmap
RUN groupadd ${USER} \
 && useradd -m -g ${USER} ${USER}
USER ${USER}
