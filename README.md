# JansenIR

## What is it?
This program calculates infrared spectroscoy of the OH stretch in bulk liquid water from gromacs molecular dynamics trajectories. The calculation is based on the theory and maps developed by Jim Skinner and coworkers. For more information about the theory, see, for example

Li and Skinner J. Chem. Phys 2010

Gruenbaum et al. JCTC 2013

The references in these papers are also useful.

## How do I use it?
The program can be built using the supplied Makefile. The prerequisits are:
(1) The xdrfile library, which can be downloaded [here](http://www.gromacs.org/Developer_Zone/Programming_Guide/XTC_Library)
(2) An MPI installation, such as open-mpi, see [here](https://www.open-mpi.org/)
(3) The fftw library for fast Fourier transforms, see [here](http://www.fftw.org/)

The program can be run from the command line using the command

mpiexec -np 8 JansenIR.exe input.inp

where input.inp is an input file and np 8 requests 8 MPI processes. An example of the necessary components of the input file, along with an explination, is supplied in the input.inp in this repositiory. Once started, the program will read the supplied xtc trajectory file and calculate the IR spectroscopy of the system.


## Features
This program can calculate infrared and Raman spectroscopy for trajectories obtained using the TIP4P, TIP4P/2005, E3B and E3B3 water models. It is only slightly different from the program in the calcIR repository. The difference is that here, an approximation (see Liang and Jansen JCTC 2012) is used to propigate the system Hamiltonian whereas in the CalcIR repository, the propigation is exact. This program is much more efficient for large systems (with >2000 water molecules) and the approximation is very accurate. 

The makefile compiles a double, with extension '_d' and single precision version of the program. I have found single precision version to be accurate and much faster than the double precision version.

## Contact
This readme is very brief. If questions arise, please contact me at nicholasjhestand@gmail.com.
