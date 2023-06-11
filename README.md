# Rope Manipulation Using GCL

To run:

```
make
```

To fully install the submodule, use `pip -e`

## Environment

The submodule `Gymnasium-Robotics-Fork` contains the Gynasium/MujuCo environment. This repository is heavily modified to include our rope as well as custom callbacks. In addition, all files irrelevant to our project were removed.

The folder `env/` contains our wrapper around the MuJuCo enviroment.

## Agents

For the discrete action space implementation, the folder `discrete_actions` contains the releveant files.
 - `agent.py` contains the trajectory distribution (policy) networks. The implemenation of the loss function for updating these networks. Also contains implementation for generating rollouts.
 - `cost.py` contains the cost networks used. Includes implementation of Algorithm 2 from Finn et al. 2016, for updating the cost network via non-linear IOC
 - `trainer.py` contains the main training loop. It implements most of Algorithm 1 calling on Agent() and Cost() as needed