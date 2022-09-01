<img src="docs/figs/spode_logo_v0.png" width="600" align="center"/>

Spode is an opensource differentiable simulator specialized for programmable photonics. It almost has no learning cost.

**It is currently under active development.**

## Why to use Spode

Spode provides the most comprehensive functions for research, design of programmable photonics.  

* Spode is a frequency-domain simulator specialized for programmable photonics.
* It supports derivative calculation of any node response to any parameter.
* The built-in generator enables users easily produce triangular, square, and hexagonal mesh.
* The built-in visualization function produces high-quality figures satisfying academic purposes.
* A few functions are available for analyze the imperfections (e.g., random variation) in programmable photonics.

## Installation

Spode is written in Python 3, with dependency on Numpy and Scipy. It should be installed successfully with ```pip```:

```
pip install spode
```

## A Friendly Example

```python
from spode.util import generate
from spode.core import Circuit
import numpy as np

# generator instance for a 2 by 2 square mesh
# initialize all TBUs in the circuit

init_dict = {'theta': 0.0, 'phi': 0.0, 'l': 250e-6}
circuit_element = generate('square_1', [2, 2], init_dict)

 
# define the circuit instance and run the simulation

circuit = Circuit(
                  circuit_element=circuit_element,
                  mode_info={'neff':2.35}, # effective index
                  omega=np.linspace(192.5,193.5,1000) * 2 * np.pi, # [192.5Thz, 193.5Thz]
                  srce_node={'n_0#2_br': 1.0},
                  prob_node=['n_2#0_br'],
                  deri_node=['n_2#0_br'],
                  deri_vari=['']) 
                  
response, grads = circuit.solve() 

# Shapes by pseudo code:
# response.shape = (len(prob_node), len(omega), 2)
# grads.shape = (len(deri_node), len(deri_vari), len(omega), 2)
```


## Tutorials

[Lesson 1: a tunable basic unit.](https://github.com/zhengqigao/spode/blob/main/tutorials/lesson1_verify_tbu/) We show how to use Spode to define a tunable basic unit (TBU), the building block of programmable photonics, and verify the simulation result by comparing with Lumerical Interconnect.

[Lesson 2: a 2 by 2 square mesh.](https://github.com/zhengqigao/spode/blob/main/tutorials/lesson2_verify_2by2_mesh/) We show two ways to define a 2 by2 square mesh (i.e., manually and using built-in generator), and verify the simulation result by comparing with Lumerical Interconnect. 

[Lesson 3: Automatic circuit generators.](https://github.com/zhengqigao/spode/tree/main/tutorials/lesson3_circuit_generator) We illustrate a few built-in circuit generators, which could be used in a one-line manner to generate triangular, square, hexagonal mesh. We also introduce a systematic way to name the TBUs, ports presented in the circuit.

[Lesson 4: Built-in visualization methods.](https://github.com/zhengqigao/spode/tree/main/tutorials/lesson4_visualization) We first illustrate the built-in visualization functions for triangular, square, hexagonal mesh. Then we explain how to visualize a customized topology by taking advantage of our provided functions.

## Contact and Bug Report

If you find any bugs, or want a new feature, please open an issue, or contact me at zhengqi@mit.edu.
