


# Installation / Debugging / Setup on Ubuntu 18.04

## Setup
1. Create a conda virtual environment and activate it.
```bash
conda create -n diffsim python=3.6 -y
conda activate diffsim

# install dependencies
sudo apt install gcc-4.8 gcc-5
sudo apt-get install libblas-dev liblapack-dev
sudo apt-get install libopenblas-dev
sudo apt-get install gfortran
sudo apt install scons
sudo apt install libpng-dev
```

2. Install diffsim:
```bash
git clone https://github.com/YilingQiao/diffsim.git
```

3. Download + install dependencies
```bash
cd diffsim
pip install -r requirements.txt
```
4. Building - NOTE: I had issues running the `script_build.sh` command out of the box, so I worked through the following parts manually
The original `script_build.sh` command contains:
```bash
sudo ./change_gcc.sh 4.8
cd arcsim/dependencies/
make 
cd ../..
sudo ./change_gcc.sh 5
make -j 8
```

Basically, the `change_gcc.sh` script calls this command, which errors out.
```
sudo update-alternatives --set gcc "/usr/bin/gcc-4.8"
```

Following this thread (https://askubuntu.com/questions/372248/downloaded-g-4-8-from-the-ppa-but-cant-set-it-as-default), it mentions "You need to let update-alternatives to know that you have 2 C++ compilers, create a record for each one, and then configure which one you want to use." 

So I had to do the following:
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 20
```

Followed by:
```
sudo update-alternatives --config gcc
```

This prompts for the `gcc` version to pick, and I selected the 4.8 option as in the `change_gcc.sh` script.

Next, I moved on to the next part of `script_build.sh`:
```
cd arcsim/dependencies/
make
```

This errored out, claiming that a script in `taucs/` was not executable, so I did the following workaround:
```
cd taucs
chmod +x configure
cd ..
```

Also had to install `g++` and `gfortran` with the correct versions:
```
sudo apt-get install gcc-4.8 g++-4.8 gfortran-4.8
```

After this, I ran the following without problems:
```
make
```

The next part of `script_build.sh` requires `gcc-5` and `g++-5`:
```
sudo apt-get install g++-5
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 20
```

Then I ran the following and selected the `gcc-5` and `g++-5` options when prompted:
```
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

You can verify the versions with:
```
gcc --version
g++ --version
```

Then, I tried running `make -j 8` as in `script_build.sh`, and received the following error complaining `libboost` is not installed, which I fixed with:
```
sudo apt-get install libboost-all-dev
make -j 8
```

At this stage, `arcsim` compiled properly

4. Create symlinks
In `script_build.sh`, the last part contains the following:

```
cd pysim
ln -s ../arcsim/conf ./conf
ln -s ../arcsim/materials ./materials
ln -s ../arcsim/meshes ./meshes
cd ..
```

However, when you run this, it mentions the following:
```
ln: failed to create symbolic link './meshes/meshes': File exists
```

It appears that the appropriate directories are already symlinked, so no need to do this step!

The examples can be run as follows (per the rest of the original README)

5. Run the examples
## Examples
### Optimize an inverse problem
```bash
python exp_inverse.py
```
By default, the simulation output would be stored in `pysim/default_out` directory. 
If you want to store the results in some other places, like `./test_out`, you can specify it by `python exp_inverse.py test_out`

To visualize the simulation results, use
```bash
python msim.py
```
You can change the source folder of the visualization in `msim.py`. More functionality of `msim.py` can be found in `arcsim/src/msim.cpp`.

The visualization is the same for all other experiments.
<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/inverse.gif"> 
</div>


### Learn to drag a cube using a cloth
```bash
python exp_learn_cloth.py
```

<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/darg.gif"> 
</div>


### Learn to hold a rigid body using a parallel gripper
```bash
python exp_learn_stick.py
```

<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/stick.gif"> 
</div>

### Scalability experiments
Figure 3, first row.
```bash
bash script_multibody.sh
```

Figure 3, second row.
```bash
bash script_scale.sh
```

### Ablation study
Table 1, sparse collision handling.
```bash
bash script_absparse.sh
```

Table 2, fast differentiation.
```bash
bash script_abqr.sh
```

### Estimate the mass of a cube
```bash
python exp_momentum.py
```

<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/momentum.gif"> 
</div>

### Two-way coupling - Trampoline
```bash
python exp_trampoline.py
```

<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/trampoline.gif"> 
</div>


### Two-way coupling - Domino
```bash
python exp_domino.py
```
<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/domino.gif"> 
</div>


### Two-way coupling - armadillo and bunny
```bash
python exp_bunny.py
```


### Domain transfer - motion control in MuJoCo

This experiment requires MuJoCo environment. Install [MuJoCo](http://www.mujoco.org/) and its python interface [mujoco_py](https://github.com/openai/mujoco-py) before running this script.
```bash
python exp_mujoco.py
```
<div align="center">
<img width="100px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/mj_mismatch.gif"> 
</div>

## Bibtex
```
@inproceedings{Qiao2020Scalable,
author  = {Qiao, Yi-Ling and Liang, Junbang and Koltun, Vladlen and Lin, Ming C.},
title  = {Scalable Differentiable Physics for Learning and Control},
booktitle = {ICML},
year  = {2020},
}
```
