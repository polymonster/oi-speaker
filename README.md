`sudo apt-get install portaudio19-dev`
`sudo apt install libspeexdsp-dev`

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
python -m venv ~/oi-speaker-env
source ~/oi-speaker-env/bin/activate
pip install oi-speaker
```
