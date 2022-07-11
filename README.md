# Deep generative genomic offset

Conditional VAEs for computing genomic offset.

## Set up
```
git clone git@github.com:israfelsr/cvae-genomic-offset.git
cd cvae-genomic-offset
python3 -m venv cvae-genomic-offset-venv
source cvae-genomic-offset-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

## Training a simple CVAE
```
# Change parameters and hyperparameters in the .sh file
sh sbatchTrainCVAE.sh
```