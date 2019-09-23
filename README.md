# Embodied Vision-and-Language Navigation with Dynamic Convolutional Filters

This is the PyTorch implementation for our paper:

**Embodied Vision-and-Language Navigation with Dynamic Convolutional Filters**<br>
__***Federico Landi***__, Lorenzo Baraldi, Massimiliano Corsini, Rita Cucchiara<br>
British Machine Vision Conference (BMVC), 2019<br>
**Oral Presentation**<br>

Visit the main [website](http://imagelab.ing.unimore.it/vln-dynamic-filters) for more details.

## Reference

If you use our code for your research, please cite our paper (BMVC 2019 oral):

### Bibtex:
```
@inproceedings{landi2019embodied,
      title={Embodied Vision-and-Language Navigation with Dynamic Convolutional Filters},
      author={Landi, Federico and Baraldi, Lorenzo and Corsini, Massimiliano and Cucchiara, Rita},
      booktitle={30th British Machine Vision Conference},
      year={2019}
    }
```

## Installation

Please follow the instructions on the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) to install the simulator.


### Clone Repo

Clone the Matterport3DSimulator repository:
```
# Make sure to clone with --recursive
git clone --recursive https://github.com/peteanderson80/Matterport3DSimulator.git
cd DynamicConv-agent
```

If you didn't clone with the `--recursive` flag, then you'll need to manually clone the pybind submodule from the top-level directory:
```
git submodule update --init --recursive
```

### Python setup

Python 3.6 is required to run our code. You can install the other modules via:
```
pip install -t requirements.txt
```

### Precomputing ResNet Image Features

Alternatively, skip the generation and just download and extract our tsv files into the `img_features` directory:
- [ResNet-152-imagenet features [380K/2.9GB]](https://www.dropbox.com/s/715bbj8yjz32ekf/ResNet-152-imagenet.zip?dl=1)
- [ResNet-152-places365 features [380K/2.9GB]](https://www.dropbox.com/s/gox1rbdebyaa98w/ResNet-152-places365.zip?dl=1)


## Training and Testing

You can train our agent by running:
```
python tasks/R2R/main.py
```
The number of dynamic filters can be set with the `--num_heads` parameter:
```
python tasks/R2R/main.py --num_heads=4
```

## License

The Matterport3D dataset, and data derived from it, is released under the [Matterport3D Terms of Use](http://dovahkiin.stanford.edu/matterport/public/MP_TOS.pdf). Our code is released under the MIT license.
