# GradCAM for OpenCLIP

Implementation of GradCAM for different OpenCLIP models

## Installation

Install poetry

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

Install dependencies

```sh
poetry config virtualenvs.in-project true
poetry env use python3.11
poetry install
```

## Usage

The GradCAM implementation works for the following model types:

- OpenCLIP:
  - ResNet
  - ConvNeXt
  - ViT
- Huggingface
  - timm/ViT

For OpenCLIP models: The CLI options `IMAGE_PATH`, `RESULT_PATH`, `CAPTION_TEXT`, `MODEL_NAME` and `PRETRAIN_TAG` have to be specified.

```sh
poetry run python main.py sample.jpg attention_map.jpg "a cat" RN50 openai
```

For Huggingface models: The CLI options `IMAGE_PATH`, `RESULT_PATH`, `CAPTION_TEXT` and `MODEL_NAME` have to be specified.

```sh
poetry run python main.py sample.jpg attention_map.jpg "a cat" hf-hub:timm/ViT-B-16-SigLIP
```
