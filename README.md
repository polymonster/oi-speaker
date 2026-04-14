# oi-speaker

oi-speaker is a cutom build smart speaker and this is the software for it. I'm recroding progress on this project on [YouTube](https://www.youtube.com/watch?v=JSrJxNG3b2o), please check it out for more info!

## Supported Platforms

Currently under development, this has been tested on Raspberry Pi 5 and macOS. But should work on any platform that supports Python.

## Dependencies (Linux)

```bash
sudo apt-get install portaudio19-dev
sudo apt-get install libspeexdsp-dev
```

## Dependencies (macOS)

```bash
brew install portaudio
brew install mpv
```

## Python Version

Python version 3.11 is required, newer python versions (Python 3.13 that now ships with Raspberry Pi) are not supported, you need to roll back and install Python 3.11.

### Python 3.11 (Linux)

```bash
# Install build dependencies
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash
```

```bash
source ~/.bashrc
pyenv install 3.11
pyenv local 3.11
```

### Python 3.11 (macOS)

```bash
brew install python@3.11
python3.11 -m venv ~/oi-speaker-env
```

### Python 3.11 (Windows)

[Installer](https://www.python.org/ftp/python/3.11.0/python-3.11.0rc2-amd64.exe)

## Python Dependencies

Python deps are configured as part of the `pyproject.toml` setup your Python env and install as so:

```bash
python3.11 -m venv ~/oi-speaker-env
source ~/oi-speaker-env/bin/activate
pip install -e .
```

## Cuda Dependencies

(Cuda 12 Toolkit)[https://developer.nvidia.com/cuda-12-0-0-download-archive]

```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12
```

### Downloading Models

Some of the dependencies require additional downloads

```bash
python -c "import openwakeword; openwakeword.utils.download_models()"
python -m piper.download --voice en_GB-northern_english_male-medium
# or
python -m piper.download_voices en_GB-northern_english_male-medium
```

## Running

```bash
speaker
```

## Training

Use Piper-TTS to generate random positive samples + add custom user generated ones.

Generate negative samples:

```
python oi-speaker/training/download-negatives.py \
    --output_dir oww-training/negative_samples \
    --n_clips 3000 \
    --clip_duration 1.0 \
    --cache_dir ~/librispeech_cache
```

Train on WSL / Linux:

Setup python env:

```
python3.10 -m venv venv_clean
source venv_clean/bin/activate

# Minimal deps — no tensorflow, no speechbrain, no piper
pip install numpy scipy scikit-learn onnxruntime torch torchaudio onnx soxr
```

Setup openWakeWord

```
https://github.com/dscripka/openWakeWord
cd openWakeWord
# Install openwakeword itself (just the inference bits)
pip install -e . --no-deps
```

Install onnx models in openWakeWord:

```
mkdir -p openWakeWord/openwakeword/resources/models

wget -O openWakeWord/openwakeword/resources/models/melspectrogram.onnx \
    https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx

wget -O openWakeWord/openwakeword/resources/models/embedding_model.onnx \
    https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx

ls -la openWakeWord/openwakeword/resources/models/
```

Install additional deps:

```
pip install onnxscript
```

Kick off training:

```
python oi-speaker/training/oi-speaker.py \
    --positive_dir oww-training/positive_samples \
    --negative_dir oww-training/negative_samples \
    --model_name oi_speaker \
    --clip_duration 1.0 \
    --epochs 100
```
