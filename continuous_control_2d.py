import numpy as np
import torch
from Arm2D import ArmEnv
from DDPG import DDPG

if __name__ == "__main__":
    env = ArmEnv()

    agent = DDPG(state_size=env.state_size,
                 action_size=env.action_size,
                 lr_actor=0.001,
                 lr_critic=0.001,
                 gamma=0.9,
                 tau=0.01,
                 memory_size=30000,
                 batch_size=32)

    n_episode = 1000
    max_t = 300

    # agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    # agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

    for i_episode in range(n_episode):
        state = env.reset()
        steps = 0
        total_return = 0
        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)

            total_return += reward
            state = next_state

            if (done is True) or (steps > max_t):
                print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i_episode, '---' if not done else 'done', total_return, steps))
                break
            steps += 1

        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')