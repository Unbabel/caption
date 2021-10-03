# Capitalisation And PuncTuatION (CAPTION)
> PT2020 Transcription project.

In this repository, we explore different strategies for automatic transcription enrichment for ASR data which includes tasks such as automatic capitalization (truecasing) and punctuation recovery.

> [Download IWSLT corpus](https://unbabel-experimental-data-sets.s3-eu-west-1.amazonaws.com/video-pt2020/IWSLT-punkt.tar.gz)

# Publications:
- [Multilingual Simultaneous Sentence End and Punctuation Prediction](http://ceur-ws.org/Vol-2957/sepp_paper3.pdf)
- [Towards better subtitles: A multilingual approach for punctuation restoration of speech transcripts](https://www.sciencedirect.com/science/article/abs/pii/S0957417421011180)
- [Automatic truecasing of video subtitles using BERT: a multilingual adaptable approach](https://link.springer.com/chapter/10.1007/978-3-030-50146-4_52)

## Sentence end and punctuation prediction shared task
To replicate our winning submission to SEPP 2021 please go to the `shared-task` branch.

## Model architecture:

![base_model](images/base_model.png)

### Available Encoders:
- [BERT](https://arxiv.org/abs/1810.04805)
- [RoBERTa](https://arxiv.org/abs/1907.11692)
- [XLM-RoBERTa](https://arxiv.org/pdf/1911.02116.pdf)

## Requirements:

This project uses Python >3.6

Create a virtual env with (outside the project folder):

```bash
virtualenv -p python3.6 caption-env
```

Activate venv:
```bash
source caption-env/bin/activate
```

Finally, run:
```bash
python setup.py install
```

If you wish to make changes into the code run:
```bash
pip install -r requirements.txt
pip install -e .
```

## Getting Started:

### Train:
```bash
python caption train -f {your_config_file}.yaml
```

### Testing:
```bash
python caption test \
    --checkpoint=some/path/to/your/checkpoint.ckpt \
    --test_csv=path/to/your/testset.csv
```

### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/lightning_logs/"
```

If you are running experiments in a remote server you can forward your localhost to the server localhost..

### How to run the tests:
In order to run the toolkit tests you must run the following command:

```bash
cd tests
python -m unittest
```

### Code Style:
To make sure all the code follows the same style we use [Black](https://github.com/psf/black).
