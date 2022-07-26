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
    
    requirements.txt

and may be installed via pip:

    pip install -r requirements.txt

# Docker image
You can download a pre-built Docker image via:

    docker pull ahenkes1/pinn:1.0.0

If you want to build the Docker image, the official TensorFlow image is needed:

    https://www.tensorflow.org/install/docker

Build via

    docker build -f ./Dockerfile --pull -t ahenkes1/pinn:1.0.0 .

Execute via

    docker run --gpus all -it -v YOUR_LOCAL_OUTPUT_FOLDER:/home/docker_user/src/saved_nets/CPINN/ --rm henkes/pinn:1.0.0 --help

where 'YOUR_LOCAL_OUTPUT_FOLDER' is an absolute path to a directory on your 
system. This will show the help.

Execute the code using standard parameters as

    docker run --gpus all -it -v YOUR_LOCAL_OUTPUT_FOLDER:/home/docker_user/src/saved_nets/CPINN/ --rm henkes/pinn:1.0.0 

# Using XLA
The code may run using XLA (faster) using the following flag:

    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.2 python3 main.py --help

where the correct cuda path and version have to be used.
The Docker image runs XLA natively.

# GPU
The code runs on GPU natively using single precision. It was observed, that on
some GPUs, which are not capable of double precision calculations, the BFGS
algorithm used may interupt before the desired number of iterations is reached.
In this case, either switch your GPU, use CPU computation or try mixed precision
loss scaling described in 
    https://www.tensorflow.org/guide/mixed_precision#loss_scaling.
There is no plan to tackle this problem in the next future.
