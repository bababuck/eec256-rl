# Rope Manipulation Using GCL

To run:

```
cd discrete
-- or --
cd easy_to_max

make
```

To fully install the submodule, use `pip -e`

## Repo File Tree  
The repository is semi-duplicated, there is a seperate directory for `discrete` and `easy_to_max`. Both have their own Makefiles that can be run.
We recommened looking at `discrete` first, only minor difference exist between the too outside the `agent.py` file.

### Environment

The submodule `Gymnasium-Robotics-Fork` contains the Gynasium/MujuCo environment. This repository is heavily modified to include our rope as well as custom callbacks. In addition, all files irrelevant to our project were removed.


The folder `discrete/env/` contains our wrapper around the MuJuCo enviroment for discrete.
The folder `easy_to_max/env/` contains our wrapper around the MuJuCo enviroment for easy_to_max.

### Agents

For the discrete action space implementation, the folder `discrete/discrete_agents/` contains the releveant files.
For the discrete action space implementation, the folder `easy_to_max/agents` contains the releveant files.
 - `agent.py` contains the trajectory distribution (policy) networks. The implemenation of the loss function for updating these networks. Also contains implementation for generating rollouts.
 - `cost.py` contains the cost networks used. Includes implementation of Algorithm 2 from Finn et al. 2016, for updating the cost network via non-linear IOC
 - `trainer.py` contains the main training loop. It implements most of Algorithm 1 calling on Agent() and Cost() as needed
