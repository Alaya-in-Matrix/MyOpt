# README

An unconstrained optimization library, with an interface like nlopt. 

Three algorithms are to be supported:

- Conjugate gradient
- BFGS
- RProp

Target problem: 1D-100D objective function with gradient infomation

Two reasons I make this library:

- Nlopt's L-BFGS does not perform well in my another project, I want't to see if it could be done better
- Nlopt does not provide conjugate gradient and rprop algorithm, I want to compare them with BFGS

## Dependency

- CMake
- Eigen
