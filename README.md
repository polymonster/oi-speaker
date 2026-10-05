# oi-speaker

oi-speaker is a cutom build smart speaker and this is the software for it. I'm recroding progress on this project on [YouTube](https://www.youtube.com/watch?v=JSrJxNG3b2o), please check it out for more info!

## Supported Platforms

Currently under development, this has been tested on Raspberry Pi 5 and macOS. But should work on any platform that supports Python.

## Dependencies (Linux)

```bash
sudo apt-get install portaudio19-dev
sudo apt-get install libspeexdsp-dev
sudo apt-get install libmpv-dev
```

## Dependencies (macOS)

```bash
brew install portaudio
brew install mpv
```

### CUDA (GPU inference)

Install the [Cuda 12 Toolkit](https://developer.nvidia.com/cuda-12-0-0-download-archive) then:


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

```bash
# use versions elector
py -3.11 -m pip
```

## Python Dependencies

Python deps are configured as part of the `pyproject.toml` setup your Python env and install as so:

```bash
python3.11 -m venv ~/oi-speaker-env
source ~/oi-speaker-env/bin/activate
pip install -e .
# optional cuda
pip install -e ".[cuda]"
```

## LED Ring (reSpeaker XVF3800)

`src/xvf3800.py` talks to the reSpeaker XVF3800 USB 4-Mic Array over its vendor USB control
interface — LED ring, direction of arrival, and the DSP tuning parameters. It is a vendored
and tidied copy of `python_control/xvf_host.py` from
[reSpeaker_XVF3800_USB_4MIC_ARRAY](https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY),
so no binaries or firmware from that repo are needed.

On Linux the control transfers need permission on the raw USB device, otherwise every command
fails with `usb.core.USBError: [Errno 13] Access denied`. `install-service.sh` installs the udev
rule for you; to do it standalone:

```bash
sudo cp setup/99-respeaker-xvf3800.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=usb
```

The rule is permanent — it re-applies on every boot and replug.

From the CLI (installed as `xvf3800`, or run the file directly):

```bash
xvf3800 --list                       # every supported command
xvf3800 VERSION
xvf3800 DOA_VALUE                    # angle 0-359, and whether speech is detected
xvf3800 LED_EFFECT --values 3        # 0 off, 1 breath, 2 rainbow, 3 solid, 4 doa, 5 ring
xvf3800 LED_COLOR --values 0xFF8800
xvf3800 LED_BRIGHTNESS --values 50
```

From Python:

```python
from xvf3800 import XVF3800

ring = XVF3800.open()   # None if the array isn't plugged in
if ring:
    ring.solid(0xFF8800)        # whole ring one colour
    ring.breathe(0x0000FF, 1)   # breathing, speed 1
    ring.ring([0xFF0000, 0x000000])  # per-LED colours, repeated around the 12 LEDs
    ring.doa_mode()             # firmware direction-of-arrival indicator (the boot default)
    ring.off()
```

The ring boots into rainbow and switches to DoA mode after ~2 seconds, so anything you set at
startup should be set after that.

## Downloading Models

Some of the dependencies require additional downloads

The openWakeWord download is required even though a custom wake word model ships in `models/` — it fetches the shared feature models (melspectrogram, embedding) and `silero_vad.onnx` into openWakeWord's own `resources/models` directory. Without it startup fails with `NO_SUCHFILE ... silero_vad.onnx`.

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

## Running as a Service (Linux)

The service is an instantiated systemd template so it works for any user without editing the file.

```bash
bash install-service.sh
```

This copies `setup/oi-speaker@.service` to `/etc/systemd/system/`, enables and starts `oi-speaker@<your-username>`.

Any args passed to the installer are appended to the speaker command line, so to offload inference to a worker:

```bash
bash install-service.sh --worker-ip 192.168.1.247:8000
```

The interpreter and args are written to `~/.config/oi-speaker/env` (`OI_SPEAKER_PYTHON` / `OI_SPEAKER_ARGS`) rather than baked into the unit — edit that file and `sudo systemctl restart oi-speaker@$USER` to change them without reinstalling. The installer picks up your active venv if one is sourced, otherwise `python3.11` from `PATH`.

Run the installer from the repo you want the service to use: it records that directory, your uid and the env file path in a drop-in at `/etc/systemd/system/oi-speaker@$USER.service.d/paths.conf`. These can't live in the template because systemd's `%h` and `%U` resolve against the service manager (ie. `/root` and `0`), not the `User=` the unit runs as.

The installer also runs `loginctl enable-linger`, and orders the unit after `user@<uid>.service`. Without this the speaker starts on boot but has no audio: `PULSE_SERVER` lives in `/run/user/<uid>`, which otherwise only exists while you're logged in.

Useful commands:

```bash
sudo systemctl status oi-speaker@$USER
sudo systemctl stop oi-speaker@$USER
journalctl -u oi-speaker@$USER -f
```

## Training

Training of custom wake word is all located in this repo since the official training was fiddly to get working with dependendencies.

### Additional Dependencies
```bash
pip install -e ".[training]"
```

```bash
pip install -e ".[training,cuda]"
```

Use Piper-TTS to generate random positive samples + add custom user generated ones.

### Generate Negative Samples

```bash
python oi-speaker/training/download-negatives.py \
    --output_dir oww-training/negative_samples \
    --n_clips 3000 \
    --clip_duration 1.0 \
    --cache_dir ~/librispeech_cache
```

### Generate Positive Samples

```bash
python training/generate-positives.py --count 200 --model .\models\piper\en_GB-northern_english_male-medium.onnx
```

### Setup python env:

```bash
python3.10 -m venv venv_clean
source venv_clean/bin/activate
pip install -e ".[training]"
```

### Setup openWakeWord

```bash
https://github.com/dscripka/openWakeWord
cd openWakeWord
# Install openwakeword itself (just the inference bits)
pip install -e . --no-deps
```

### Install onnx models in openWakeWord:

```bash
mkdir -p openWakeWord/openwakeword/resources/models

wget -O openWakeWord/openwakeword/resources/models/melspectrogram.onnx \
    https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx

wget -O openWakeWord/openwakeword/resources/models/embedding_model.onnx \
    https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx

ls -la openWakeWord/openwakeword/resources/models/
```

#### Kick off training

```bash
python oi-speaker/training/oi-speaker.py \
    --positive_dir training/training_data/positive_samples \
    --negative_dir training/training_data/negative_samples \
    --model_name oi_speaker \
    --epochs 100
```

#### Diagnose

Script can be run to diagnose model performance

```bash
python training/diagnose.py --pos training/training_data/positive_samples_processed --neg training/training_data/negative_samples
```

