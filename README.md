# oi-speaker

oi-speaker is a cutom build smart speaker and this is the software for it. I'm recroding progress on this project on [YouTube](https://www.youtube.com/watch?v=JSrJxNG3b2o), please check it out for more info!

## Supported Platforms

Currently under development, this has been tested on Raspberry Pi 5. But should work on any platform that supports Python.

## Dependencies (Linux)

`sudo apt-get install portaudio19-dev`
`sudo apt-get install libspeexdsp-dev`

## Dependencies (macOS)

`brew install portaudio`
`brew install mpv`

## Python Version

Python version 3.11 is required, newer python versions (Python 3.13 that ships with Raspberry Pi) are not supported, on Pi to roll back you need to build and install Python 3.11.

### Python 3.11 (Linux)

```
# Install build dependencies
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash
```

```
source ~/.bashrc
pyenv install 3.11
pyenv local 3.11
```

### Python 3.11 (macOS)

```
brew install python@3.11
python3.11 -m venv ~/oi-speaker-env
```


## Python Dependencies

Python deps are configured as part of the `pyproject.toml` setup your Python env and install as so:

```
python3.11 -m venv ~/oi-speaker-env
source ~/oi-speaker-env/bin/activate
pip install -e .
```

### Downloading Models

Some of the dependencies require additional downloads

```
python -c "import openwakeword; openwakeword.utils.download_models()"
python -m piper.download --voice en_GB-northern_english_male-medium

or

python -m piper.download_voices en_GB-northern_english_male-medium
```

## Running

```
speaker
```
