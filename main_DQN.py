from environment import DataEnv
from DQN import DeepQNetwork
from classifier import Classifier

MAX_EPISODES = 900
ON_TRAIN = True

# set env
env = DataEnv()
s_dim = env.state_dim
a_dim = env.action_dim
clf = Classifier('logistic')

# set RL method
rl = DeepQNetwork(a_dim, s_dim,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=2000,
                  # output_graph=True
                  )

steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        state = env.reset(clf)
        ep_r = 0.
        while True:
            # env.render()

            action = rl.choose_action(state)

            state_, reward, done = env.step(action, clf)

            rl.store_transition(state, action, reward, state_)

            ep_r += reward
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            state = state_
            if done:
                print('Ep: %i | %s | ep_r: %.1f' % (i, '---' if not done else 'done', ep_r))
                break
    rl.save()


def eval():
    rl.restore()
    # env.render()
    state = env.reset(clf)
    while True:
        env.render()
        action = rl.choose_action(state)
        state, reward, done = env.step(action, clf)


if ON_TRAIN:
    train()
    rl.plot_cost()
else:
    eval()



