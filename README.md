# HENKES_PINN
Code of the publication "Physics informed neural networks for continuum micromechanics" published in https://doi.org/10.1016/j.cma.2022.114790 by Alexander Henkes and Henning Wessels from TU Braunschweig and Rolf Mahnken from University of Paderborn.

Please cite the following paper:

    @article{henkes2022physics,
      title={Physics informed neural networks for continuum micromechanics},
      author={Henkes, Alexander and Wessels, Henning and Mahnken, Rolf},
      journal={Computer Methods in Applied Mechanics and Engineering},
      volume={393},
      pages={114790},
      year={2022},
      publisher={Elsevier}
    }

... and the code using the CITATION.cff file.

# Requirements
The requirements can be found in
    
    src/requirements.txt

and may be installed via pip:

    $pip install -r requirements.txt

# Docker image
For the Docker image, the official TensorFlow image is needed:

    https://www.tensorflow.org/install/docker

Build via

    $cd src
    $docker build -f ./Dockerfile --pull -t henkes/pinn:1.0.0 .


Execute via

    $docker run --gpus all -it --rm --mounttype=bind,source='/home/ah/HENKES_PINN/src',target='/home/' henkes/pinn:1.0.0 
    $cd home
    $python3 main.py --help

# Using XLA
The code may run using XLA (faster) using the following flag:

    $XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.2 python3 main.py

where the correct cuda path and version have to be used.