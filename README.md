# ABM20 - An Agent-Based poker model

Presented in this repository is an agent based model of simplified poker, in which agents compete against
their von Neumann Neighbors on a 2D-Toroidal plane.

## Features
- Simulated evolutionary poker strategies, based on private valuation of base pokerhands modified by agent risk aversion
- Multi-scenario batchrunning and analysis
- Animated plots of wealth distribution
- Sensitivity analysis of model parameters

## Usage
To use the program, clone the repository to your device. Then from the root folder run `python ./poker_model/ [jobFile.json: Default=/jobs/job.json]`

## Job Files
Simulation parameters are loaded from JSON-formatted "job" files. An example jobfile has been provided in `/root/jobs/jobExample.json` and an explaination of relevant properties has been provided below.

### Root properties
- [Optional] `jobName`: Name of the specific job for user-readability. May be used as documentation of the purpose of the job.
- `replications`: Amount of repeat runs for every set of scenario parameters for statistical analysis.
- [Optional] `seed`: Seed for the seed generator of sucessive runs. If no seed has ben given, a random seed wil be used instead.
- `sharedParameters`: A JSON object containing [simulation parameters](#simulation-parameters) to use for all provided scenarios.
- `scenarios`: A list of [Scenario](#scenario) JSON objects that contain model properties as well as analytical visualization settings.
- `sens_analysis`: A boolean value that determines whether sensitivity analysis is performed at the end of the job.

### Scenario
Contains the necessary parameters for a single set of simulations for further analysis. 
- `name`: Name of the scenario, as well as the legend-label in the analysis plots.
- `color`: Linecolor for the relevant data in the analysis plots.
- `parameters`: A JSON object containing [simulation parameters](#simulation-parameters) to be used for specifically this scenario.

## Simulation Parameters
- `games`: Amount of iteration to run the model for. During one iteration an agent will play five games of poker.
- `gridDim`=`[10,10]`: A 2-element array containing the (integer) length and width of the model grid. The model will contain LxW active agents.
- `n_rounds`=`1`: Amount of rounds to play per game. More rounds more accurately reflect the quality of an agent's strategy, but also increase the model runtime.
- `ra_bounds`=`[0,1]`: The risk-aversion bounds of newly generated agents. (Min=0, Max=1).
- `agentType`=`"evo"`: A string that dictates the whether risk-aversion is an inheritable trait (`"evo"`) or not (`"fixed"`).
- `nRecombine`=`5`: The amount of strategy points (including risk-aversion) an agent may inherit from a neighbor agent. (Max=10 for `"fixed"` `agentType` and Max=11 for `"evo"` `agentType`). 
- `weightRecombine`=`0.5`: The weight of inherited strategy points. (Min=0, Max=1)
- `nMut`=`1`: The amount of independent mutations an agent performs on their own strategy. Mutations occur by sampling over a normal distribution with as mean the current strategy point value and a given standard deviation.
- `mut_std`=`10`: The standard-deviation of agent strategy point mutation. Strategy points are clamped within a fixed space between 0 and 400.

## Requirements
See `Requirements.txt`