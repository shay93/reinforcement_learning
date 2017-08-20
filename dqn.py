import tensorflow as tf
import random as rand
import numpy as np 
# so perhaps we perform a rollout prior to actually running the 
# a basic implementation of dqn for the cartpole environment
# ultimately we will have a function that actually runs the episode

def main(buffer_size,
        epsilon,
        target_update,
        num_episodes,
        env):
    # first thing to do is to initialize the buffer
    # we can also specify the maximum size of the buffer before initializing
    # it
    replay_buffer = ExperienceReplay(buffer_size)
    # also initialize a Q object
    q_function = QFunction()
    # initialize a counter for the number of steps taken
    step_counter = 0 
    for episode_num in range(num_episodes):
        # we will now loop through the steps in each episode
        # at the start of each episode reset the environment 
        observation = env.reset()
        # also set the total reward in the episode to 0
        total_reward = 0
        for step in range(episode_length):
            # increment step counter
            step_counter += 1
            # presumption is that a random number is generated between 0 and 1
            # then with probability epsilon we will select a random action
            if rand.random() < epsilon:
                # define an action space which will be sampled for random actions
                action = action_space.sample()
            else:
                # on the other hand with 1 - epsilon probability select an
                # action as the action that yields the maximum q value over the
                # action space keep this should make use of the target Q function
                action = q_function.find_action(observation)
        # now that we have an action we can execute it in the emulator
        new_observation, reward, done, info = env.step(action)
        # increment the total reward in the episode using the step
        total_reward += reward
        # we want to store the transisition tuple in the buffer
        replay_buffer.add((observation, action, reward, new_observation, 
                            done))
        # at this point we update the action_value function if the buffer
        # crosses some threshold
        if replay_buffer.size > threshold:
            # update the Qfunction by passing it a minibatch of transitions
            transitions = replay_buffer.sample(batch_size)
            # transisitions can just be a list of tuples where the list is of
            # length batch_size
            q_function.train(transitions)
        # update the target Q function if that threshold has been reached
        if step_counter % target_update == 0:
            q_function.update_target()
        # if the episode has reached completion we can break out of it and
        # reset the episode
        if done:
            break
        # at the of the the new observation should become the current obs
        observation = new_observation

class ExperienceReplay(object):
    """
    Experience buffer to keep track of environment transitions and provide
    samples for training
    """
    def __init__(self,
                 max_size):
        """
        :param max_size: The maximum size of the buffer
        """
        # perhaps initialize the arrays based on the provided input
        
    def add(self,
            transition_tuple):
        """
        Add a single transition tuple to the buffer
        :param transition_tuple: A tuple of form 
            (observation, action, reward, next_observation, done)
        :return : None
        """
        return NotImplemented
        
    def sample(self,batch_size):
        """
        Sample a sequence of transitions from the replay buffer
        :param batch_size: An integer specifying the number of transistions to
                sample
        :return A list of transition tuples of length equal to batch_size
        """
        return NotImplemented
        
class QFunction(object):
    """
    An object that represents the action value function that will be used to
    choose actions by computing action values on the target Q function
    """
    def __init__(self):
        """
        Perhaps build the Tensorflow graphs here
        """

        
    def find_action(self,
                    observation):
        """
        Find the action with the highest q value by evaluating the target_qfn
        on the 
        :param observation: A numpy array of shape [observation_dim]
        :return A numpy array of shape [action_dim]
        """
        raise NotImplementedError
        
    def train(self, transitions):
        """
        Update the parameters of the surrogate Q function using the
        Bellman update
        """
        
    def update_target(self):
        """
        Update the parameters of the target Q function by copying over the
        parameters of the surrogate Q function 
        """
        
class Model(object):
    """
    The Tensorflow graph that represents the model that is being used to
    compute action values 
    """
    
    def __init__(self, scope):
        """
        initialize some model attributes that will be useful later
        """
        self.scope = scope
    
    def build_graph(self):
        """
        Build a tensorflow graph in the given scope
        return an op dict that will be used to create the
        """
    

        
    