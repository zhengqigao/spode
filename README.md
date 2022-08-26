<img src="docs/figs/spode_logo_v0.png" width="400" align="center"/>

Spode is an opensource differentiable simulator specialized for programmable photonics. **It is currently under active development.**

## Why to use Spode

Spode provides the most comprehensive functions for research, design of programmable photonics.  

* Spode is a frequency-domain simulator specialized for programmable photonics.
* It supports derivative calculation of any port magnitude to any parameter.
* The built-in generator enables users easily produce triangular, square, and hexagonal mesh.
* The built-in visualization functions produces high-quality figures satisfying academic purposes.
* A few functions are available for analyze the imperfections (e.g., random variation) in programmable photonics.


## Installation

Spode is written in Python 3, with dependency on Numpy. It should be installed successfully with ```pip```:

```
pip install spode
```

## Quick Tutorial

[Lesson 1: a tunable basic unit.](https://github.com/zhengqigao/spode/blob/main/tutorials/lesson1_verify_tbu/) We show how to use Spode to define a tunable basic unit (TBU), the building block of programmable photonics, and verify the simulation result by comparing with Lumerical Interconnect.

[Lesson 2: a 2 by 2 square mesh.](https://github.com/zhengqigao/spode/blob/main/tutorials/lesson2_verify_2by2_mesh/) We show two ways to define a 2 by2 square mesh (i.e., manually and using built-in generator), and verify the simulation result by comparing with Lumerical Interconnect. 

[Lesson 3: Automatica circuit generator.](https://github.com/zhengqigao/spode/tree/main/tutorials/lesson3_circuit_generator) We illustrate a few built-in circuit generators, which could be used in one-line manner to generate triangular, square, hexagonal mesh. We also introduce a systematic way to name the TBUs, ports presented in the circuit.

## Contact and Bug Report

If you find any bugs, or want a new feature, please open an issue, or contact me at zhengqi@mit.edu.
