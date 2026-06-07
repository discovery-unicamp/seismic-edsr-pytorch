# Seismic EDSR-PyTorch

A modified [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) framework for single image Super-Resolution of seismic and other single-channel images. Adapted to work with arbitrary dynamic ranges (not limited to 0–255 RGB), as used in seismic data.

Available models: EDSR, RDN, RCAN, WDSR, and D-DBPN.

This work was developed as part of the following master's research:

> Araujo, Lucas de M. "Análise Comparativa de Redes Neurais Convolucionais para Super-Resolução de Imagens Naturais e Sísmicas (Comparative Analysis of Convolutional Neural Networks for Super-Resolution of Seismic and Natural Images)," Master's thesis, University of Campinas (Unicamp), 2026, in preparation.

Original EDSR framework:

> Lim, Bee et al. (2017). "Enhanced Deep Residual Networks for Single Image Super-Resolution". In: *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*. [[original repo](https://github.com/sanghyun-son/EDSR-PyTorch)]

WDSR model from [JiahuiYu/wdsr_ntire2018](https://github.com/JiahuiYu/wdsr_ntire2018).

## Datasets

### DIV2K

[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is the photographic image dataset used by the original work. Download the High Resolution training and validation sets:

```bash
cd datasets/
mkdir -p DIV2K

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

unzip -n DIV2K_train_HR.zip -d DIV2K
unzip -n DIV2K_valid_HR.zip -d DIV2K
```

Convert to single-channel TIFF (grayscale, float32, normalized to [-1, 1]):

```bash
python to_tiff.py DIV2K
```

This produces a `DIV2K_TIFF/` directory with the same structure. See `python to_tiff.py -h` for other options.

### Unicamp-NAMSS

[Unicamp-NAMSS](https://doi.org/10.5281/zenodo.18330486) is a general-purpose diversified 2D seismic image dataset. Download the three splits from Zenodo and extract them into `datasets/`:

```bash
cd datasets/

tar -xzf train.tar.gz
tar -xzf validation.tar.gz
tar -xzf test.tar.gz
```

Prepare the images (trim dimensions to be divisible by 12, split wide images):

```bash
python prepare_namss_from_zenodo.py
```

Use `--delete_source` to remove the extracted files after processing and avoid duplicating ~30GB of data. The expected output structure is:

```
datasets/NAMSS/
├── NAMSS_train_HR/
├── NAMSS_valid_HR/
└── NAMSS_test_HR/
```

Shuffle the dataset filenames for balanced training validation:

```bash
python shuffle_namss.py
```

This uses a fixed seed (3309) for reproducibility. To undo, run `python unshuffle_namss.py`.

For more information on how the dataset was collected and processed, see [discovery-unicamp/unicamp-namss-dataset](https://github.com/discovery-unicamp/unicamp-namss-dataset).

### Generating Low Resolution samples

Once the HR images are in place, generate the Low Resolution (bicubic downsampled) counterparts:

```bash
cd datasets/

python generate_LR_images.py DIV2K_TIFF
python generate_LR_images.py NAMSS
```

By default this generates 2x and 4x downsampled versions. Use `--scales` to specify other factors (e.g. `--scales=2,3,4`).

## Reproducing Experiments

### Environment

The experiments were run on a Docker container based on:

```
pytorch/pytorch@sha256:1e26efd426b0fecbfe7cf3d3ae5003fada6ac5a76eddc1e042857f5d049605ee
```

Key dependencies: Python 3.10, PyTorch 1.13, imageio 2.15.0, scikit-image 0.20.0, matplotlib, tensorboard, tqdm.

### Retraining the models

The scripts `src/experiment_scripts/run-experiment-NN.sh` train and test all five models (EDSR, RDN, RCAN, WDSR, D-DBPN) on both datasets at a given scale, using the original experiment seeds. From the `src/` directory:

```bash
bash experiment_scripts/run-experiment-01.sh
```

Each script sets the `SCALE` variable (2 or 4) at the top. All models are trained for 300 epochs. For scales >2, EDSR, RDN, and RCAN use the pre-trained x2 model as initialization.

Trained models and results are saved under `experiment-<NN>/`.

### Trained weights

Pre-trained model weights will be available at [REDU (Repositório de Dados da Unicamp)](https://redu.unicamp.br/) — link pending.

To reproduce the benchmark results from the trained weights, download and extract the archives from the project root:

```bash
tar -xzf EDSR-DIV2K-trained-weights.tar.gz
tar -xzf EDSR-Unicamp-NAMSS-trained-weights.tar.gz
# ... etc.
```

This places model files under `experiment-NN/<MODEL>-X<SCALE>-<DATASET>-Seed<SEED>/model/model_best.pt`. Then run from `src/`:

```bash
bash experiment_scripts/run-all-benchmarks.sh
```

This discovers all model directories and runs inference on the corresponding test sets.
