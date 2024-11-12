# FMM-BEM 

# Implementation of the Fast Multipole Method for accelerating. Boundary Element Method calculations for the Laplace equation.


# Abstract
The Poisson equation is a fundamental partial differential equation with numerous applications in physics and engineering. However, solving it can be computationally challenging, especially for large-scale problems. The Boundary Element Method (BEM) offers an effective approach, but it often requires substantial memory resources due to the dense matrix operations involved. To address this, we implement the Fast Multipole Method (FMM) within the BEM framework, significantly enhancing the computational efficiency. The FMM reduces the complexity of operations, allowing for faster solutions while maintaining accuracy. This repository implements a FMM-BEM algorithm for typical potential problems with laplace equation and mixed boundary conditions. 


For a correct use of this repository, the following versions and libraries were used:

- Matplotlib = 3.9.1
- Numpy = 2.0.1
- Meshio = 5.3.5








