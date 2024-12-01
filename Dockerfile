FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN python3 -m pip install --upgrade pip \
&&  pip install --no-cache-dir \
    black==22.3.0 \
    jupyterlab==3.4.2 \
    jupyterlab_code_formatter==1.4.11 \
    lckr-jupyterlab-variableinspector==3.0.9 \
    jupyterlab_widgets==1.1.0 \
    ipywidgets==7.7.0 \
    import-ipynb==0.1.4 \
    transformers==4.46.3 \
    datasets==3.1.0 \
    evaluate==0.4.3 \
    scikit-learn==1.5.2 \
    peft==0.13.2

WORKDIR /work

