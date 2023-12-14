from ruamel.yaml import YAML, dump, RoundTripDumper
from ME491_2023_project.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from ME491_2023_project.helper.raisim_gym_helper import ConfigurationSaver, load_param, load_param_env, load_paramO, tensorboard_launcher, load_param_envO_return_mean_var
from ME491_2023_project.env.bin.rsg_anymal import NormalSampler
from ME491_2023_project.env.bin.rsg_anymal import RaisimGymEnv
from ME491_2023_project.env.RewardAnalyzer import RewardAnalyzer
import os
import math
import time
import ME491_2023_project.algo.ppo.module as ppo_module
import ME491_2023_project.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse


# task specification
task_name = "ME491_2023_project"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-ww', '--weightO', help='opponent pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight
weight_pathO = args.weightO


# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file

num_envs = cfg['environment']['num_envs']
render = cfg['environment']['render']
cfg['environment']['render'] = False
env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
cfg['environment']['num_envs'] = 1
cfg['environment']['render'] = render
env_visual = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
cfg['environment']['num_envs'] = num_envs
env.seed(cfg['seed'])
env_visual.seed(cfg['seed'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
num_threads = cfg['environment']['num_threads']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           5.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/ME491_2023_project/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", 
                                       task_path + "/runner.py", 
                                       task_path + "/Environment.hpp",
                                       task_path + "/AnymalController_20190430.hpp",
                                       task_path + "/AnymalController_oppo_1.hpp"])

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.99,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir
              )

reward_analyzer = RewardAnalyzer(env, ppo.writer)

if mode == 'vt':
    load_param_env(weight_path, env_visual)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    done_sum_test = 0
    reward_sum_test = 0
    env_visual.turn_on_visualization()
    
    env_visual.reset()
    test_steps = math.floor(300 / cfg['environment']['control_dt'])
    currentIteration = 10000
    env_visual.curriculum_callback(min((currentIteration - 1000)/2000, 1))
    for step in range(test_steps):
        with torch.no_grad():
            frame_start = time.time()
            obs_vt = env_visual.observe(False)
            action_vt = loaded_graph.architecture(torch.from_numpy(obs_vt).cpu())
            reward, dones, _ = env_visual.step(action_vt.cpu().detach().numpy())
            done_sum_test += np.sum(dones)
            reward_sum_test += np.sum(reward)

            frame_end = time.time()
            wait_time = 1*cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
               time.sleep(wait_time)

    average_ll_performance = reward_sum_test / test_steps
    average_dones = done_sum_test / test_steps
    env_visual.turn_off_visualization()

    print('(test--test--test-------------------------------------')
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('---------------------------------------test--test--test\n')


if mode == 'plain':
    tensorboard_launcher(saver.data_dir, 24681)  # press refresh (F5) after the first ppo update
    
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
    load_param_env(weight_path, env_visual)
    env.turn_off_visualization()
    env.reset()
    trial_sum = 0
    win_sum = 0
    continueLearningUpdate = 1200
    for update in range(continueLearningUpdate, 1000000):
        start = time.time()
        reward_sum = 0
        done_sum = 0
        average_dones = 0.

        if update % cfg['environment']['eval_every_n'] == 0 and update!=continueLearningUpdate:
            print("Visualizing and evaluating the current policy")
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
            }, saver.data_dir+"/full_"+str(update)+'.pt')
            # we create another graph just to demonstrate the save/load method
            env.save_scaling(saver.data_dir, str(update))

            # load observation scaling from files of pre-trained model
            env_visual.load_scaling(saver.data_dir, update)

            loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
            loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

            done_sum_test = 0
            reward_sum_test = 0
            win_sum_test = 0
            env_visual.turn_on_visualization()
            env_visual.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')
            env_visual.curriculum_callback(min((update - 1000)/2000, 1))
            env_visual.reset()
            test_steps = math.floor(30 / cfg['environment']['control_dt'])
            for step in range(test_steps):
                with torch.no_grad():
                    frame_start = time.time()
                    obs_vt = env_visual.observe(False)
                    action_vt = loaded_graph.architecture(torch.from_numpy(obs_vt).cpu())
                    reward, dones, win = env_visual.step(action_vt.cpu().detach().numpy())
                    done_sum_test += np.sum(dones)
                    reward_sum_test += np.sum(reward)
                    win_sum_test += np.sum(win)

                    frame_end = time.time()
                    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

            average_ll_performance = reward_sum_test / test_steps
            average_dones = done_sum_test / test_steps
            win_rate = win_sum_test / (int(done_sum_test == 0) + done_sum_test)
            env_visual.stop_video_recording()
            env_visual.turn_off_visualization()

            print('(test--test--test-------------------------------------')
            print('{:>6}th iteration test'.format(update))
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
            print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
            print('{:<40} {:>6}'.format("win_rate: ", '{:0.6f}'.format(win_rate)))
            print('---------------------------------------test--test--test\n')
            
        # actual training
        env.curriculum_callback(min((update - 1000)/2000, 1))
        for step in range(n_steps):
            obs = env.observe()
            action = ppo.act(obs)
            reward, dones, wins = env.step(action)
            reward_analyzer.add_reward_info(env.get_reward_info())
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_sum = reward_sum + np.sum(reward)
            trial_sum += np.sum(dones)
            win_sum += np.sum(wins)

        # take st step to get value obs
        obs = env.observe()
        ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
        average_ll_performance = reward_sum / total_steps
        average_dones = done_sum / total_steps
        avg_rewards.append(average_ll_performance)

        actor.update()
        actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))
        
        
        action = ppo.just_act(obs)
        r, d, w = env.step(action)
        end = time.time()

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                        * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')
        ppo.writer.add_scalar('Env/reward', average_ll_performance, update)
        ppo.writer.add_scalar('Env/dones', average_dones, update)
        reward_analyzer.analyze_and_plot(update)
