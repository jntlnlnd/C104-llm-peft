
ポートは8888だと外部から繋げないことが多いため、別の値を使うことをおすすめする
```
$ docker build -t transformers-jupyter .
$ docker run --restart always -v $PWD/notebooks:/work/notebooks -v $HOME/.cache/huggingface:/root/.cache/huggingface  -w /work -it --gpus all -p 0.0.0.0:18888:8888 transformers-jupyter:latest jupyter-lab --no-browser --ip=* --allow-root --NotebookApp.token=''  
```


## memo
https://qiita.com/eijenson/items/25b35916afa38cdf9cea
