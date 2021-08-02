


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
### Learn to "fold" a cloth in half (while it is dropping with gravity)
```bash
python exp_cloth_fold.py
```
By default, the simulation output would be stored in `pysim/default_out` directory. 

To visualize the simulation results, use
```bash
python msim.py
```

The visualization is the same for all other experiments.
<div align="center">
<img width="300px" src="https://github.com/priyasundaresan/diffsim/blob/master/pysim/visualize_results/rollout_0.gif"> 
<img width="300px" src="https://github.com/priyasundaresan/diffsim/blob/master/pysim/visualize_results/rollout_5.gif"> 
<img width="300px" src="https://github.com/priyasundaresan/diffsim/blob/master/pysim/visualize_results/rollout_10.gif"> 
</div>

