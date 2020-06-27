import numpy as np
import parl
from parl.utils import logger
from parl.utils import action_mapping  # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory  # 经验回放
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境
from parl.algorithms import DDPG


class QuadrotorAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim=4):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(QuadrotorAgent, self).__init__(algorithm)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost


import paddle.fluid as fluid
import parl
from parl import layers


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hidden_dim_1, hidden_dim_2 = 64, 64
        self.fc1 = layers.fc(size=hidden_dim_1, act='tanh')
        self.fc2 = layers.fc(size=hidden_dim_2, act='tanh')
        self.fc3 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):
        x = self.fc1(obs)
        x = self.fc2(x)
        return self.fc3(x)


class CriticModel(parl.Model):
    def __init__(self):
        hidden_dim_1, hidden_dim_2 = 64, 64
        self.fc1 = layers.fc(size=hidden_dim_1, act='tanh')
        self.fc2 = layers.fc(size=hidden_dim_2, act='tanh')
        self.fc3 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        x = self.fc1(obs)
        concat = layers.concat([x, act], axis=1)
        x = self.fc2(concat)
        Q = self.fc3(x)
        Q = layers.squeeze(Q, axes=[1])
        return Q


class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()



GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
ACTOR_LR = 0.0002  # Actor网络更新的 learning rate
CRITIC_LR = 0.001  # Critic网络更新的 learning rate
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward


def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        # 给输出动作增加探索扰动
        action = np.random.normal(action, 1.0)
        action = np.squeeze(action)

        # 动作从 5 个压缩为 4 个
        temp = np.zeros((1, 4))
        temp_1 = np.array([ [ 1.0, 0.0, 0.0, 0.0 ] ])
        temp_2 = np.array([ [ 0.0, 1.0, 0.0, 0.0 ] ])
        temp_3 = np.array([ [ 0.0, 0.0, 1.0, 0.0 ] ])
        temp_4 = np.array([ [ 0.0, 0.0, 0.0, 1.0 ] ])

        temp += list(action)[ 0 ]

        temp_1 *= list(action)[ 1 ]
        temp_2 *= list(action)[ 2 ]
        temp_3 *= list(action)[ 3 ]
        temp_4 *= list(action)[ 4 ]

        action_4 = temp + 0.1 * (temp_1 + temp_2 + temp_3 + temp_4)
        action_4 = np.squeeze(action_4)
        action_4 = np.squeeze(action_4)
        action_4 = np.clip(action_4, -1.0, 1.0)


        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        action_4 = action_mapping(action_4, env.action_space.low[ 0 ],
                                  env.action_space.high[ 0 ])

        next_obs, reward, done, info = env.step(action_4)
        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
            batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = [ ]
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)

            # 动作从 5 个压缩为 4 个
            temp = np.zeros((1, 4))
            temp_1 = np.array([ [ 1.0, 0.0, 0.0, 0.0 ] ])
            temp_2 = np.array([ [ 0.0, 1.0, 0.0, 0.0 ] ])
            temp_3 = np.array([ [ 0.0, 0.0, 1.0, 0.0 ] ])
            temp_4 = np.array([ [ 0.0, 0.0, 0.0, 1.0 ] ])

            temp += list(action)[ 0 ]

            temp_1 *= list(action)[ 1 ]
            temp_2 *= list(action)[ 2 ]
            temp_3 *= list(action)[ 3 ]
            temp_4 *= list(action)[ 4 ]

            action_4 = temp + 0.1 * (temp_1 + temp_2 + temp_3 + temp_4)
            action_4 = np.squeeze(action_4)
            action_4 = np.squeeze(action_4)
            action_4 = np.clip(action_4, -1.0, 1.0)


            action_4 = action_mapping(action_4, env.action_space.low[ 0 ],
                                      env.action_space.high[ 0 ])

            next_obs, reward, done, info = env.step(action_4)

            obs = next_obs
            total_reward += reward
            steps += 1

            if render:
                env.render()

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


# 创建飞行器环境
env = make_env("Quadrotor", task="velocity_control", seed=0)
env.reset()
obs_dim = env.observation_space.shape[ 0 ]
act_dim = 5

# 使用parl框架搭建Agent：QuadrotorModel, DDPG, QuadrotorAgent三者嵌套
model = QuadrotorModel(act_dim)
algorithm = DDPG(
    model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = QuadrotorAgent(algorithm, obs_dim, act_dim)

# 加载模型
# save_path = 'model_dir_3/steps_1000000.ckpt'
# agent.restore(save_path)

# parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)

test_flag = 0
total_steps = 0


testing = 1

if (not testing):
    while total_steps < TRAIN_TOTAL_STEPS:
        train_reward, steps = run_episode(env, agent, rpm)
        total_steps += steps
        # logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward))

        if total_steps // TEST_EVERY_STEPS >= test_flag:
            while total_steps // TEST_EVERY_STEPS >= test_flag:
                test_flag += 1

            evaluate_reward = evaluate(env, agent)
            logger.info('Steps {}, Test reward: {}'.format(total_steps,
                                                           evaluate_reward))

            # 保存模型
            ckpt = 'model_dir_1/steps_{}.ckpt'.format(total_steps)
            agent.save(ckpt)

else:
    # 加载模型
    save_path = 'steps_1000000.ckpt'
    agent.restore(save_path)

    evaluate_reward = evaluate(env, agent, render=True)
    logger.info('Test reward: {}'.format(evaluate_reward))