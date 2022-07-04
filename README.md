## Install

1. Download OpenMX package version 3.9.9 from http://www.openmx-square.org/download.html.

1. Install the dependent environment of the original OpenMX. Check that the original OpenMX can be installed successfully.

1. Install [HDF5 1.12.1](https://www.hdfgroup.org/package/hdf5-1-12-1-tar-gz/?wpdmdl=15727&refresh=62bc78374c8081656518711) for C language.

1. **Apply the 'overlap only' patch**. Copy patch files `overlap_only_patch/openmx.c` and `overlap_only_patch/truncation.c` to the source directory of the original OpenMX (`openmx3.9/source`).

1. **Edit `makefile` in the source directory of the original OpenMX**. Add ` -I${HDF5_path}/include -L${HDF5_path}/lib` at the end of the `CC` in the original `makefile`, with `${HDF5_path}` replaced by the HDF5 path; add ` -lhdf5` at the end of the `LIB` in the original `makefile`. Notice: depending on installation setup, the library path may be `${HDF5_path}/lib` or `${HDF5_path}/lib64`.

1. Run `make` command to install:
    ```bash
    make clean
    make all
    make install
    ```
    Executable file `openmx` can be found in the source directory.

## Usage

1. Before running 'overlap only' OpenMX, the path to the HDF5 shared objects must be added to the runtime library search path. For example,
    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HDF5_path}/lib
    ```
    with `${HDF5_path}` replaced by the HDF5 path. Notice: depending on installation setup, the library path may be `${HDF5_path}/lib64` or `${HDF5_path}/lib64`.

1. Perform the 'overlap only' program like the original OpenMX. For example, for overlap matrices of magic angle twisted bilayer graphene (MATBG), move to the directory `examples` of current repository, edit your path to the VPS and PAO (`openmx3.9/DFT_DATA19`) for `DATA.PATH` keyword in file `MATBG.dat`, and run:
    ```bash
    mpirun -np 1 ${openmx_path} MATBG.dat > openmx.std
    ```
    with `${openmx_path}` replaced by the path of installed 'overlap only' OpenMX. The overlap matrices could be found in the `output` directory in the format of HDF5.
