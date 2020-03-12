# segment-rnn

## Install

```
pip3 install -r requestment.txt
```

## Directory Structure

- exp/: it contains the experiment that use LSTM, GRU, and SimpleRnn unit, and the comparison between single-sample and multi-sample method.
    - ACP/
    - AMP/
    - cifar10/
    - fashion/
    - imdb/
    - mnist/
    - sentiment/
    - sentiment140/
    - utils/
-** img__________-analyze: it contains the experiment that analyzes the image
    - cifar10/
    - fashion/
    - mnist/
- segment-equal/: it contains the experiment that analyzes the effect of equal-segment method
    - imdb/
    - sentiment/
    - sentiment140/
    - utils/
***
## Experiment

### exp/

#### ACP

***create essential directory
**```
cd ACP/
mkdir log tmp
```

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### AMP

create essential directory
```
cd AMP/
mkdir log tmp
```

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### cifar10 

create essential directory
```
cd cifar10/
mkdir log tmp
```

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### fashion

create essential directory
```
cd fashion/
mkdir log tmp data
```

```
cd data/
```
download [data](https://www.kaggle.com/zalando-research/fashionmnist) and unzip

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### mnist

create essential directory
```
cd mnist/
mkdir log tmp
```

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### sentiment

create essential directory
```
cd sentiment/
mkdir log tmp
```

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### sentiment140

create essential directory
```
cd sentiment140/
mkdir log tmp data
```

```
cd data/
```
download [data](https://www.kaggle.com/kazanova/sentiment140) and unzip

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### imdb

create essential directory
```
cd imdb/
mkdir log tmp
```

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

### img-analyze

#### cifar10 

create essential directory
```
cd cifar10/
mkdir log tmp
```

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### fashion

create essential directory
```
cd fashion/
mkdir log tmp data
```

```
cd data/
```
download [data](https://www.kaggle.com/zalando-research/fashionmnist) and unzip

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### mnist

create essential directory
```
cd mnist/
mkdir log tmp
```

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

### segment-equal

#### sentiment

create essential directory
```
cd sentiment/
mkdir log tmp
```

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### sentiment140

create essential directory
```
cd sentiment140/
mkdir log tmp data
```

```
cd data/
```
download [data](https://www.kaggle.com/kazanova/sentiment140) and unzip

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`

#### imdb

create essential directory
```
cd imdb/
mkdir log tmp
```

run the experiment
```
./run.sh
```

calculate the result
```
python3 util.py
```

see the result on `draw.ipynb`
