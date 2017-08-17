import tensorflow as tf

# so perhaps we perform a rollout prior to actually running the 
# a basic implementation of dqn for the cartpole environment
# ultimately we will have a function that actually runs the episode

def run_episode(env,
                max_steps=200,
                ):
    """
    :param env: An environment object
    :param max_steps: The number of steps to run the episode over
    :return: The total reward over the episode
    """
    # the first thing to do is to reset the environment and get the
    # corresponding observation
    observation = env.reset()
    # initialize the total reward to 0 to begin with
    total_reward = 0
    # now step through the episode compute the actions and continue
    for _ in range(max_steps):
        # this is where we will get the action
        action = action_fn(observation)
        # once we have an action we can observe its affect on the environment
        observations, reward, done, info = env.step(action)
        total_reward += reward
        # if the the episode goal has been achieved break out of the episode
        # at what point should we update the q function
        if done:
            break
    return total_reward
    
def action_fn(observation):
    """
    Given an observation provide an action to apply to the environment.
    The action space in this case is of dimension 2
    :param observation: A 1d array of shape (4,)
    :return A scalar which is either a 1 or 0
    """
    # the first thing is to compute the action value function over the
    # set of possible actions that will be used a correct action
    q_vals = q_fn(observation)
    # q_vals should be of shape [2] where the first element will correspond
    # to the Q val for the action 0 and the second q value for the action
    # corresponding to 1
    return np.argmax(q_vals)
    
def q_fn(observation):
    """
    Compute the q values for the provided observation this is equivalent to
    just using the correct
    """
    
class HParams(object):
    """
    Specify the hyperparameters for the DQN model
    """

class ExperienceReplay(object):
    """
    Define an object that holds the previously encountered transistions
    """
    