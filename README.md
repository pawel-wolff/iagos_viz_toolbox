# IAGOS viz toolbox

A python package with visualization routines used in the IAGOS project


## Installation

### Install python environment

- Reuse the conda environment on NUWA:
    ```sh
    conda activate /home/wolp/miniconda3/envs/matplotlib-viz
    ```

or

- Install your own conda environment:
    ```sh
    conda create --name iagos-viz-toolbox-env --file requirements.txt -c conda-forge
    conda activate iagos-viz-toolbox-env
    ```
    and then install `xarray-extras` package in the environment. Its source code is on NUWA in the git repository and can follow these steps:
    ```sh
    git clone [<your-nuwa-ssh-access>:]/home/wolp/projects/xarray_extras.git
    cd xarray_extras
    python setup.py develop
    ```
    The last command uses pip to install the package in the *development* mode (i.e. any modifications to the source code of the package will take effect)


### Install the package

Once the conda environment described above is setup and active, you can install `iagos-viz-toolbox` package:

```sh
git clone [<your-nuwa-ssh-access>:]/home/wolp/projects/iagos-viz-toolbox.git
cd iagos-viz-toolbox
python setup.py develop
```
