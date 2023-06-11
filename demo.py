from env.control import ControlEnv
import numpy as np

def create_demo(env):
    """ Generate the expert demonstrations to be used as training data by the model.

    Every pair of numbers in `setups` and `demos` is a (segment, direction) pair. While the setup uses
    4 different segments, the model will only be given the option of one of two segments to grab, 0 and 3.

    The directions are encoded as follows, each pair is (dx, dy)
    0 = [1, 0]
    1 = [-1, 0]
    2 = [0, 1]
    3 = [0,-1]

    Output states is

    Inputs:
    env = ControlEnv() that to simulate the given actions. Will capture the states from this as part
          of expert state-action pairs.
    """

    # Created using model trained via supervised learning
    # Mixes the ropes up into different shapes
    setups = [[1, 3, 2, 2, 3, 1, 0, 0, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 1],
              [2, 2, 3, 3, 1, 2, 3, 1, 3, 1, 0, 0, 0, 0, 2, 0, 3, 1, 0, 0, 3, 1, 1, 1],
              [1, 3, 1, 3, 3, 3, 0, 2, 2, 3, 3, 1, 0, 0, 0, 0, 3, 1, 2, 1, 3, 3, 0, 0, 3, 2, 3, 3],
              [3, 3, 1, 2, 0, 0, 0, 0, 3, 1, 1, 2, 1, 3],
              [0, 2, 1, 3, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1],
              [1, 2, 2, 2, 1, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 1, 2, 0],
              [3, 3, 3, 1, 3, 1, 2, 0, 3, 1, 2, 0, 0, 0, 1, 2, 3, 3, 3, 0],
              [3, 3, 1, 2, 1, 2, 0, 0, 3, 3, 3, 1, 3, 1, 0, 0, 3, 1, 3, 1, 3, 3, 2, 0, 0, 0, 2, 0],
              [1, 3, 2, 2, 2, 2, 1, 3, 1, 3, 1, 3, 0, 2, 2, 0],
              [1, 2, 1, 3, 1, 2, 1, 2, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 3, 1],
              [2, 2, 0, 0, 1, 2, 1, 2, 3, 1, 1, 0, 3, 0, 3, 1],
              [2, 2, 0, 0, 3, 1, 3, 3, 3, 1, 1, 3, 2, 1, 0, 0, 3, 2, 0, 0, 3, 3, 0, 0],
              [0, 2, 2, 2, 3, 1, 2, 2, 3, 3, 3, 1, 3, 1, 0, 0, 3, 3, 1, 2, 3, 1],
              [2, 2, 2, 2, 0, 2, 1, 2, 3, 1, 3, 1, 0, 0, 0, 0, 0, 0, 2, 0, 3, 1, 0, 0, 3, 1],
              [0, 2, 0, 2, 1, 3, 3, 1, 1, 3, 0, 0, 2, 0, 2, 1, 0, 0, 0, 0]
              ]
    
    # These were demonstrations given by an expert (me, not really an expert)
    # The goal is to make the rope into a circle.
    demos = [
        [3,3, 3,3, 3,1, 3,1, 3,2, 3,2, 3,1, 3,2, 3,2, 0,1, 0,3, 3,0, 0,0, 3,0],
        [0,1, 0,1, 0,1, 0,2, 0,0, 0,3, 3,3, 3,1, 3,1, 3,2, 3,0, 3,2],
        [3,2, 0,3, 0,3],
        [3,1, 3,1, 3,1, 3,2, 3,2],
        [3,0, 3,2, 3,2, 3,2, 3,2, 3,1, 3,1, 3,1, 3,3, 3,3, 0,2, 3,3],
        [0,1, 0,1, 0,1, 0,3, 3,1, 3,1, 3,1, 3,2],
        [3,1, 3,1, 0,3, 3,2, 3,2], #V-Good
        [0,2, 0,2, 0,1, 0,1, 0,3, 0,3, 3,1, 3,1, 3,1, 3,2, 3,0, 3,2, 3,0],
        [3,2, 3,2, 3,1, 3,1, 0,2, 3,1, 3,1, 3,1, 3,3, 3,3],
        [3,3, 3,3, 3,3, 0,2, 3,1, 0,1, 0,1, 0,1, 0,3, 0,3, 0,3, 0,0, 3,1, 3,1, 3,2, 3,2],
        [3,1, 0,1, 0,1, 0,3, 0,3, 0,3, 0,0, 0,0, 0,0, 3,1],
        [0,1, 0,1, 0,1, 0,3, 0,3, 0,3, 0,0, 0,0, 0,0, 0,0, 3,0, 3,0, 3,3, 3,3, 3,3, 3,1, 3,1],
        [0,3, 3,1, 3,2, 3,2],
        [0,2, 0,2, 0,2, 0,1, 0,1, 0,1, 0,1, 0,3, 0,3, 3,3, 3,1, 3,1, 3,1, 3,2, 3,2, 3,2],
        [3,0, 3,0, 3,0, 3,2, 3,2, 3,1, 3,1, 3,1, 3,1, 3,3, 3,1]
        ]

    # The following run the simulation and gathers the actions and states
    states = [] 
    actions = []
    assert len(setups) == len(demos)
    for index in range(len(setups)):
        # Initialize the enviroment to a mixed up pre-set state
        env.reset()
        setup = setups[index]
        for i in range(0, len(setup), 2):
            # Decode direction to continuous
            directions = [0, 0]
            if setup[i+1] == 0:
                directions = [1, 0]
            elif setup[i+1] == 1:
                directions = [-1, 0]
            elif setup[i+1] == 2:
                directions = [0, 1]
            elif setup[i + 1] == 3:
                directions = [0, -1]
            ob, _, _ = env.step(segment=setup[i], direction=directions)

        # Simulate the expert actions one by one
        demo = demos[index]
        for i in range(0, len(demo), 2):
            states.append(ob.tolist())
            d_action = [0, 0, 0]
            d_action[0] = demo[i]
            if demo[i+1] == 0:
                d_action[1:3] = [1, 0]
            elif demo[i+1] == 1:
                d_action[1:3] = [-1, 0]
            elif demo[i+1] == 2:
                d_action[1:3] = [0, 1]
            elif demo[i + 1] == 3:
                d_action[1:3] = [0, -1]
            actions.append(d_action) # Save action pairs

            ob, _, _ = env.step(segment=demo[i], direction=d_action[1:3])

    return actions, states


if __name__ == '__main__':
    """ Generate expert state-action pairs. """
    np.random.seed(0)
    env = ControlEnv(False)

    actions, states = create_demo(env)

    np.save("demos/demo_cont_states", np.array(states))
    np.save("demos/demo_cont_actions", np.array(actions))
