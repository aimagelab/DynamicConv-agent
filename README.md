# Embodied Vision-and-Language Navigation with Dynamic Convolutional Filters

This is the PyTorch implementation for our paper:

[**Embodied Vision-and-Language Navigation with Dynamic Convolutional Filters**](https://bmvc2019.org/wp-content/uploads/papers/0384-paper.pdf)<br>
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

### Clone Repo

Clone the repository:
```
# Make sure to clone with --recursive
git clone --recursive https://github.com/fdlandi/DynamicConv-agent.git
cd DynamicConv-agent
```

If you didn't clone with the `--recursive` flag, then you'll need to manually clone the pybind submodule from the top-level directory:
```
git submodule update --init --recursive
```

### Python setup

Python 3.6 is required to run our code. You can install the other modules via:
```
pip install -r requirements.txt
cd speaksee
pip install -e .
``` 

### Building with Docker

Please follow the instructions on the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) to install the simulator via Docker.

### Bulding without Docker

The simulator can be built outside of a docker container using the cmake build commands described above. However, this is not the recommended approach, as all dependencies will need to be installed locally and may conflict with existing libraries. The main requirements are:
- Ubuntu >= 14.04
- Nvidia-driver with CUDA installed 
- C++ compiler with C++11 support
- [CMake](https://cmake.org/) >= 3.10
- [OpenCV](http://opencv.org/) >= 2.4 including 3.x
- [OpenGL](https://www.opengl.org/)
- [GLM](https://glm.g-truc.net/0.9.8/index.html)
- [Numpy](http://www.numpy.org/)

Optional dependences (depending on the cmake rendering options):
- [OSMesa](https://www.mesa3d.org/osmesa.html) for OSMesa backend support
- [epoxy](https://github.com/anholt/libepoxy) for EGL backend support

### Build and Test

Build the simulator and run the unit tests:
```
cd DynamicConv-agent
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make
cd ../
./build/tests ~Timing
```

If you use a conda environment for your experiments, you should specify the python path in the cmake options:
```
cmake -DEGL_RENDERING=ON -DPYTHON_EXECUTABLE:FILEPATH='path_to_your_python_bin' ..
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

## Reproducibility Note

Results in our paper were obtained with version v0.1 of the Matterport3DSimulator. Due to this difference, results could vary from the one in the paper. Using different GPUs for training, as well as different random seeds, may also affect results.

We provide the weights obtained with our training. To reproduce results from the paper, run:
```
python tasks/R2R/main.py --name=normal_data --num_heads=4 --eval_only
```

or:
```
python tasks/R2R/main.py --name=data_augmentation --num_heads=4 --eval_only
```

## License

The Matterport3D dataset, and data derived from it, is released under the [Matterport3D Terms of Use](http://dovahkiin.stanford.edu/matterport/public/MP_TOS.pdf). Our code is released under the MIT license.
