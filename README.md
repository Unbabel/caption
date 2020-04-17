# Capitalisation And PuncTuatION (CAPTION)
> PT2020 Transcription project.

In this repository, we explore different strategies for automatic transcription enrichment for ASR data which includes tasks such as automatic capitalization (truecasing) and punctuation recovery.

## Model architecture:

![base_model](https://i.ibb.co/sm3P2Bq/Screenshot-2020-04-14-at-16-19-10.png = 500x)

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
