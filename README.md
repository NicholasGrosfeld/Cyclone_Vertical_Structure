This repository contains the code required to identify cyclones as local minima in geopotential height fields (script 1), join individual cyclone points into tracks through time (script 2),
and link the tracks at different levels into a full vertical profile of the cyclones. 

Note that a future version of this code will be written using either pandas or polars for the array operations, to remove the need to refer to columns in a numpy array with magic integers. 
