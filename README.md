# A reduced basis method for parabolic PDEs based on a space-time least squares formulation

This is the proof-of-concept Python code used in our paper on the space-time least squares formulation for parabolic problems assuming minimal regularity, enabling the reproduction of the results.

## Usage
We use the FEniCSx FEM backend, implemented by DOLFINx (https://github.com/FEniCS/dolfinx). The code was originally developed and tested with the following Python packages:
- NumPy (https://github.com/numpy/numpy) Version 1.26.4
- SciPy (https://github.com/scipy/scipy) Version 1.13.1
- DOLFINx (https://github.com/FEniCS/dolfinx) Version 0.8.0
- UFL (https://github.com/FEniCS/ufl) Version 2024.1.0
- petsc4py (https://github.com/simnibs/petsc4py) Version 3.21.1
- mpi4py (https://github.com/mpi4py/mpi4py) Version 3.1.6
- Gmsh (https://gitlab.onelab.info/gmsh/gmsh/) Version 4.13.1
Newer versions of the packages may work as well.

All of the simulations were tested with the IPython interpreter in a conda environment. To obtain the same environment that we use for developing all our projects, open 'spec-file.yml' and enter a desired name and the standard path for the environment in 'name' and 'prefix'.

Open the repository folder in your terminal and type
```bash
conda env create -f spec-file.yml
```
to install this environment. Alternatively, if you only require standard DOLFINx support, follow the installation instructions at https://github.com/FEniCS/dolfinx.


## Demo
Two example problems are provided:
- Example1.py (2D thermal block)
- Example2.py (3D problem with minimal regularity)

To get the results from the article, run 
```bash
ipython Example1.py
```
and
```bash
ipython Example2.py
```
for computation. The algorithm will then explain what it is doing. To obtain the results with the relative error estimator in the second example, set the "use_rel_estimator" variable to True in the code.

The POD-greedy algorithm produces CSV output files after each iteration step, which can also be found in the "Results" folder. This data is used in the article:
- val_set_errors.csv: The average error on the validation set;
- val_set_rough_eff.csv: The average effectivity of the standard estimator on the validation set;
- val_set_fine_eff.csv: The effectivity of the estimator with exact residual on the validation set;
- err_estimator.csv: The (maximal) error estimator of the current iteration of the POD-greedy algorithm;
- thm_estimator_real_res.csv: The estimator with exact residual evaluated there;
- err_real.csv: The true error evaluated there.

## Support
For support, contact mstahl@uni-koblenz.de