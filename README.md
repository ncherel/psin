# A Patch-Based Algorithm for Diverse and High Fidelity Single Image Generation

This is a fork for proposed modifications of the official implementation for the paper presented at ICIP 2022:

**A Patch-Based Algorithm for Diverse and High Fidelity Single Image Generation**  
N. Cherel, A. Almansa, Y. Gousseau, A. Newson

Link to [[Preprint]](https://hal.science/hal-03822204/) [[Paper]](https://ieeexplore.ieee.org/document/9897913)

We present a pure patch-based solution to single image generation that does not require learning.
As a result new samples are possible using this code in a few seconds.

This algorithm contains the code for our **PSin** algorithm only.  
The reference code for the optimal transport initialization is found at optimization https://github.com/ahoudard/wgenpatex . Our fork with minor modifications will be released soon.

Reference | Generated
:--------:|:---------:
![Reference Image](balloons.png) | ![Algorithm output](output.png)

# Install

The requirements are:
- opencv
- numpy
- scipy
- cffi
- numba


## Accelerate
You can accelerate processing by compiling the source file `patch_measure.cpp` with the following command (tested on Linux only):
```
g++ -fPIC -shared patch_measure.cpp -O3 -o libpatch_measure.so
```
And then activate it in `config.py` with `USE_CPP=True`.


## Run

The code is then run using :
```
python synthesis.py
```

The default file used as reference is available is `balloons.png`.

