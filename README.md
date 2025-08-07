# IREE Inferencer

## Overview
This repo contains IREE-based inference scripts for ArtCNN. It is an alternative to the usual ONNX-based workflows. The code here is a modified version of the inference scripts within the main [ArtCNN repo](https://github.com/Artoriuz/ArtCNN), but with fewer features due to some extra constraints.

> [!TIP]
> By default, the "engine" is configured to target Vulkan SPIR-V. IREE supports various other targets, including ROCm, CUDA and Metal. A list of stable targets can be found [here](https://iree.dev/guides/deployment-configurations/).

> [!WARNING]
> While importing models from ONNX is fully supported, performance seems to be considerably worse when compared to models converted to MLIR from TensorFlow's savedmodel format. Furthermore, models with static shapes also seem to produce better performance. For these reasons, the models in this repository are hardcoded to receive 1x256x256x1 inputs (NHWC format). They will not work with other resolutions or other data formats.

## Instructions
```shell
usage: inferencer.py [-h] [-m MODEL] [-t TASK] input

IREE Inferencer

positional arguments:
  input                 Input image

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        ArtCNN Model
  -t TASK, --task TASK  Task to perform
```

## Examples
Luma upscaling using a standard luma model:
```shell
python inferencer.py image.png --model ArtCNN_R8F64_256_256_FP32.mlir --task luma
```

RGB upscaling using a standard luma model on each channel:
```shell
python inferencer.py image.png --model ArtCNN_R8F64_256_256_FP32.mlir --task rgb
```
