# Streaming transformer
Pytorch implementation of [Augmented Memory Transformer](https://arxiv.org/pdf/2005.08042.pdf) for streaming automatic 
speech recognition with linear attention mechanism from [this paper](https://arxiv.org/pdf/2006.16236.pdf).


### Data preparation

Download [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/)

```bash
cd data
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2   # download data 
tar xjf LJSpeech-1.1.tar.bz2                                      # extract data
python prepare_vocabulary.py                                      # building target dictionary
```

### Training

```bash
python train.py
```