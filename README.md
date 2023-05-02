# Advanced Manufacturing Systems - CIM Indystry 4.0 

Programming Exercises for course DPMS_AS/1110 - Intelligent Manufacturing Systems, School of Mechanical Engineering, National Technical University of Athens.

# Exercise 1 - Timed Petri net for simulating control of flexible manufacturing system
  
In this exercise we created a model of a manufacturing system, using petri nets. In the petri net P-Invariant and T-Invariant analysis was used, in order to check the the bound of the system as well and the deadlocks. 

## How to run

In order to run, first you must downnload software [PIPE v.4.3.0](https://sourceforge.net/projects/pipe2/).
In addition open a terminal or cmd inside Pipe directory and execute the command
```bash
java Pipe
```

# Exercise 2 - Prediction of Temprature - Time at welding plate using Neural Networks

In this exercise we have created two neural network models for predicting maximum temperature at a welding plate process and predicting the time interval for temperatires above 723C.

## Installation

```bash
pip3 install virtualenv
virtualenv neural_network
source neural_network/bin/activate
pip3 install -r requirements.txt
```
## Training

For maximum temperature NN training, run this command

```bash
python3 MaxTemperatureNN_training.py
```

For time over 723 Celcius degrees NN training, run this command

```bash
python3 AboveStandardTemperatureNN_training.py
```
## Usage

For maximum temperature prediction, run this command

```bash
python3 MaxTemperatureNN_inference.py --plate_thickness 0.004 --initial_temperature 180 --heat_input 900 --electrode_velocity 0.004 --X 0.0 --Y 0.02 --Z 0.002
```

For time over 723 Celcius degrees, run this command

```bash
 python3 AboveStandardTemperatureNN_inference.py --plate_thickness 0.005 --initial_temperature 200 --heat_input 1200 --electrode_velocity 0.0035 --X 0.025 --Y 0.025 --Z 0.0025 
```

# Exercise 3 - Genetic Algorithm for Optimazition of welding process

In this exercise we have created a genetic algorithm for optimizing the neural networks for welding plate process. 
The cost funtion of the algorithm implements an euclidean distance between two temperatures. It also uses a penalty criterion on temperature on two occasions.

-The first occasion is when  
```math
      T \lt A
```
-The second occasion is when
```math
    T \gt K
```
# Authors

[Lampis Papakostas](https://github.com/LPapakostas)

George Kassavetakis

[George Krommydas](https://github.com/GeoKrom)
