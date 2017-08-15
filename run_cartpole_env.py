import gym
from gym import wrappers
import numpy as np

def run_episode(env,parameters):
    """
    :param env: An environment object
    :param parameters: An array corresponding to the policy parameters
    :return: the total reward
    """
    observation = env.reset()
    total_reward = 0
    for _ in range(200): # 200 steps in each episode
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == '__main__':
    # initialize the cartpole environment
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env,
                           '/tmp/cartpole-experiment-2',
                           force=True)

    bestreward = 0

    # loop over 20 episodes
    for i_episode in range(20):
        # at the start of each episode reset the environment
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
        # considered solved if agent lasts 200 timesteps
        if reward == 200:
            break

    # use a random guessing algorithm to
    # close out of the environment before uploading uploading the monitor file
    env.close()
    # upload your results to OpenAI gym
    gym.upload('/tmp/cartpole-experiment-1', api_key='sk_o27h6fgSSYyOkVTKKpb7lQ')

