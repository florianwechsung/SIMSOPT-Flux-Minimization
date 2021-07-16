# A simple flux minimisation code using SIMSOPT and coilpy


## Requirements

    pip3 install h5py xarray matplotlib pyevtk sympy
    pip3 install coilpy 

Then install SIMSOPT using the `fw/vjp` branch, i.e.
    
    git clone --recursive git@github.com:hiddenSymmetries/simsopt
    cd simsopt
    git checkout fw/vjp
    pip3 install -e .
    
## Running the code

    python3 driver.py

This will create output `/tmp/init.{png, vts, vtu}` and `/tmp/det.{png, vts, vtu}`.
