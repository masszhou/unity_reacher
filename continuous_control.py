from unityagents import UnityEnvironment
import numpy as np
import torch
from DDPG_batchsample import DDPG


def train(env, brain_name, agent, n_episode=100, max_t=1000):

    for i_episode in range(n_episode):
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment

        batch_states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)

        steps = 0
        while True:
            # batch_actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            # batch_actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            batch_actions = agent.act(batch_states)
            #
            env_info = env.step(batch_actions)[brain_name]  # send all actions to tne environment
            batch_next_states = env_info.vector_observations  # get next state (for each agent)
            batch_rewards = env_info.rewards  # get reward (for each agent)
            batch_dones = env_info.local_done  # see if episode finished

            # agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)

            scores += env_info.rewards  # update the score (for each agent)
            batch_states = batch_next_states  # roll over states to next time step

            steps += 1
            if np.any(batch_dones):  # exit loop if episode finished
                break
        print('episode {}, steps {}, avg_score : {}'.format(i_episode, steps, np.mean(scores)))
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_reacher20.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_reacher20.pth')


def test(env, brain_name, agent, n_episode=100):
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor_reacher20.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic_reacher20.pth'))

    for i_episode in range(n_episode):
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment

        batch_states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)

        steps = 0
        while True:
            batch_actions = agent.act(batch_states)

            env_info = env.step(batch_actions)[brain_name]  # send all actions to tne environment
            batch_next_states = env_info.vector_observations  # get next state (for each agent)
            batch_rewards = env_info.rewards  # get reward (for each agent)
            batch_dones = env_info.local_done  # see if episode finished

            scores += env_info.rewards  # update the score (for each agent)
            batch_states = batch_next_states  # roll over states to next time step

            steps += 1
            if np.any(batch_dones):  # exit loop if episode finished
                break
        print('episode {}, steps {}, avg_score : {}'.format(i_episode, steps, np.mean(scores)))


if __name__ == "__main__":

    env = UnityEnvironment(file_name='./Reacher_Linux_20/Reacher.x86_64', no_graphics=True)
    # env = UnityEnvironment(file_name='./Reacher_Linux_20/Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    batch_states = env_info.vector_observations
    state_size = batch_states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(batch_states.shape[0], state_size))
    print('The state for the first agent looks like:', batch_states[0])

    agent = DDPG(state_size=33,
                 action_size=4,
                 sample_batch_size=20,
                 lr_actor=0.0001,
                 lr_critic=0.0001,
                 gamma=0.99,
                 tau=0.01,
                 memory_size=int(1e6),
                 batch_size=128)

    # train(env, brain_name, agent)
    test(env, brain_name, agent)

    env.close()
