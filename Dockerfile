# Training image: conda env from environment.yaml + this repo
# -> docker build -t lolcats .
# -> docker run --gpus all -e HF_TOKEN -e HF_REPO=<hf-user>/<repo> lolcats make lizard
FROM condaforge/miniforge3:26.7.2-0

RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*

COPY environment.yaml /tmp/environment.yaml
# There is no GPU during the build, so tell conda which CUDA to solve for
RUN CONDA_OVERRIDE_CUDA=12.4 conda env create -f /tmp/environment.yaml && conda clean -afy
ENV PATH=/opt/conda/envs/lolcats-env/bin:$PATH

WORKDIR /workspace/lolcats
COPY . .
CMD ["make", "lizard"]
