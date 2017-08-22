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
                 max_size,
                 observation_dim,
                 action_dim):
        """
        :param max_size: The maximum size of the buffer
        :param observation_dim: The size of the observations 
        """
        # set an attribute for the max buffer size and the observation
        # as well as the action dimension
        self.max_size = max_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        # also initialize an attribute that specifies when the buffer is full
        self.full = False
        # also initialize a pointer which will be used to add new entries
        # to the buffer
        self.pos_point = 0
        # initialize numpy arrays for the transitions
        self.obs = np.ndarray(shape=[self.max_size, self.observation_dim],
                                 dtype=np.float32)
        
        self.act = np.ndarray(shape=[self.max_size, self.action_dim],
                            dtype=np.float32)
        
        self.reward = np.ndarray(shape=[self.max_size],
                                dtype=np.float32)
        
        self.next_obs = np.ndarray(shape=[self.max_size, self.observation_dim],
                                    dtype=np.float32)
        
        self.done = np.ndarray(shape=[self.max_size],
                                dtype=bool)
        
    def add(self,
            transition_tuple):
        """
        Add a single transition tuple to the buffer
        :param transition_tuple: A tuple of form 
            (observation, action, reward, next_observation, done)
        :return : None
        """
        # add each element of the tuple to matching array using the pointer
        self.obs[self.pos_point] = transition_tuple[0]
        self.act[self.pos_point] = transition_tuple[1]
        self.reward[self.pos_point] = transition_tuple[2]
        self.next_obs[self.pos_point] = transition_tuple[3]
        self.done[self.pos_point] = transition_tuple[4]
        # check to see if the buffer has been filled
        if self.pos_point == (max_size - 1):
            self.full = True
        # now increment the pointer
        self.pos_point = (self.pos_point + 1) % self.max_size
        
    def sample(self,
            batch_size):
        """
        Sample a sequence of transitions from the replay buffer
        :param batch_size: An integer specifying the number of transistions to
                sample
        :return A list of transition tuples of length equal to batch_size
        """
        # first thing to do is to make sure that the buffer contains atleast
        # a number of elements equal to batch size
        assert self.full or (self.pos_point > batch_size - 1),
            "There aren't enough elements in the buffer to sample from"
        # first thing to do is is obtain a random sample of indexes
        if self.full:
            random_indexes = rand.sample(range(self.max_size), batch_size)
        else:
            random_indexes = rand.sample(range(self.pos_point), batch_size)
            
        # now get the transistions corresponding to the random samples
        obs_batch = [self.obs[i] for i in random_indexes]
        act_batch = [self.act[i] for i in random_indexes]
        reward_batch = [self.reward[i] for i in random_indexes]
        next_obs_batch = [self.next_obs[i] for i in random_indexes]
        done_batch = [self.done[i] for i in random_indexes]
        
        # zip these elements together and return a list of tuples
        return list(zip(obs_batch, act_batch, reward_batch
                        next_obs_batch, done_batch))
         
        
class QFunction(object):
    """
    An object that represents the action value function that will be used to
    choose actions by computing action values on the target Q function
    """
    def __init__(self):
        """
        Build the tensorflow graphs and get the op dicts
        """
        self.target_ops = Model("target").build_graph()
        self.surrogate_ops = Model("surrogate").build_graph()
        # create a session
        self.sess = tf.Session()
        # initialize the variables for both the target and surrogate ops
        self.sess.run([self.target_ops["init"],
                       self.surrogate_ops["init"]])
        

    def find_action(self,
                    observation):
        """
        Find the action with the highest q value
        :param observation: A numpy array of shape [observation_dim]
        :return A numpy array of shape [action_dim]
        """
        # pass an observation through the network to get Q values for all
        # the candidate actions 
        observation = np.expand_dims(observation, 0)
        # create the feed dict
        feed_dict = {self.target_ops["observation"]: observation}
        q_vals = self.sess.run(self.target_ops["y"],
                                feed_dict=feed_dict)
        # q_vals should be an array of shape [1, size of action space]
        return np.argmax(np.squeeze(q_vals))
        
    def train(self, 
            transitions):
        """
        Update the parameters of the surrogate Q function using the
        Bellman update
        :param transitions: A list of tuples of the form 
                    (observation, action, reward, next_observation, done)
        """
        # first thing to do is to unpack the list of transitions into batches
        # that are easier to work with
        obs_batch = np.array([t[0] for t in transitions])
        next_obs_batch = np.array([t[3] for t in transitions])
        
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
        return an op dict
        """
        
class ActionSpace(object):
    """
    An object that represents the action space of an environment
    """
    
    def __init__(self,
                 vals):
        """
        :param vals: A python list of discrete vals that form a set from
        which an action may be chosen from
        """
        self.vals = vals
        
    def sample(self):
        """
        Sample a random action from the action space
        :return a valid action from the action space
        """
        return rand.sample(self.vals,1)[0]
    

        
    