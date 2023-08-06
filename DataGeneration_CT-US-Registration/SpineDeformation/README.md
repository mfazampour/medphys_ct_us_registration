# SpineDeformation

Deform spine from supine to prone with sofa framework


# To Build SofaPython plugin (On jakku)
- install Sofa framework
- clone the plugin repo
- create your sofa conda env (python 3.7)   
- install missing requirements based on the `readme` of the plugin.
- create `build` folder and run `CMAKE` according to the following:
```shell
cmake -DCMAKE_PREFIX_PATH="/home/farid/SOFA/v22.12.00/lib/cmake;/home/farid/miniconda3/envs/cycle_gan/lib/python3.7/site-packages/pybind11/share/cmake/" -DPython_EXECUTABLE=/home/farid/miniconda3/envs/sofa/bin/python ..
```
- run `make -j 8` and `sudo cmake --install .` afterwards.
  - there might be an error in one of the headings, just remove `override` and run make again.
- to make it work on your conda env, you need to copy the build libraries from `Site-Packages` folder of the system python folder to the conda one:

```shell
cp -r /home/farid/.local/lib/python3.9/site-packages/* /home/farid/.local/lib/python3.7/site-packages/
```

These are the packages:
` Sofa  SofaRuntime  SofaTypes  splib`

- then run python using these environment variables:

```shell
SOFA_ROOT=/home/farid/SOFA/v22.12.00/ LD_LIBRARY_PATH="/home/farid/SOFA/v22.12.00/lib/:/home/farid/SOFA/v22.12.00/collections/SofaSimulation/lib/:$LD_LIBRARY_PATH" python
```

- then `import Sofa` and the rest should work

## NOTICE

verse 561 and 584 failed