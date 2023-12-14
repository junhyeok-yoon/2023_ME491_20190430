from ruamel.yaml import YAML, dump, RoundTripDumper
from ME491_2023_project.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from ME491_2023_project.helper.raisim_gym_helper import ConfigurationSaver, \
    load_param, load_param_env, load_paramO, tensorboard_launcher, load_param_envO_return_mean_var
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
                                       task_path + "/AnymalController_20190430.hpp"])

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.985,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir
              )
tensorboard_launcher(saver.data_dir, 24681)  # press refresh (F5) after the first ppo update

reward_analyzer = RewardAnalyzer(env, ppo.writer)

if mode == 'vt':
    # load_param_env(weight_path, env_visual)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
    # loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    done_sum_test = 0
    reward_sum_test = 0
    env_visual.turn_on_visualization()
    
    env_visual.reset()
    test_steps = math.floor(300 / cfg['environment']['control_dt'])
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
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
    load_param_env(weight_path, env_visual)
    env.turn_off_visualization()
    env.reset()
    trial_sum = 0
    win_sum = 0
    continueLearningUpdate = 0
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
            loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
            loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

            done_sum_test = 0
            reward_sum_test = 0
            win_sum_test = 0
            # we create another graph just to demonstrate the save/load method
            env.save_scaling(saver.data_dir, str(update))

            # load observation scaling from files of pre-trained model
            env_visual.load_scaling(saver.data_dir, update)
            env_visual.turn_on_visualization()
            env_visual.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

            env_visual.reset()
            test_steps = math.floor(60 / cfg['environment']['control_dt'])
            for step in range(test_steps):
                with torch.no_grad():
                    frame_start = time.time()
                    obs_vt = env_visual.observe()
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

        if(update % 10 == 9):
            ppo.writer.add_scalar('Env/win_rate', win_sum/(int(trial_sum == 0) + trial_sum), update)
            ppo.writer.add_scalar('Env/win_num', win_sum, update)
            ppo.writer.add_scalar('Env/trial_num', trial_sum, update)
            print("win_rate  : ", win_sum/(int(trial_sum == 0) + trial_sum))
            print("win_num   : ", win_sum)
            print("trial_num : ", trial_sum)
            trial_sum = 0
            win_sum = 0
            
        # actual training
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


oppoOpDim = 95      
# construct graph_list and mean_list and var_list
list1 = []
list2 = []
# note that in weight_path0, there are folders and in that folders there is pt file
for root, dirs, files in os.walk(weight_pathO):
# if root starts with self ~ then, make list1
    if root.split('/')[-1].startswith('self'):
        for file in files:
            if file.endswith('.pt'):
                list1.append(os.path.join(root, file))
# if root starts with oppo ~ then, make list2
    if root.split('/')[-1].startswith('target'):
        for file in files:
            if file.endswith('.pt'):
                list2.append(os.path.join(root, file))
print(list1, list2)
graph_list1 = []
mean_list1 = []
var_list1 = []
for i in range(len(list1)):
    module = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim).to(device)
    graph_list1.append(module)
    graph_list1[i].load_state_dict(torch.load(list1[i])['actor_architecture_state_dict'])
    mean1, var1 = load_param_envO_return_mean_var(list1[i], env)
    mean_list1.append(mean1)
    var_list1.append(var1)
graph_list2 = []
mean_list2 = []
var_list2 = []
for i in range(len(list2)):
    module = ppo_module.MLP(cfg['architecture']['oppo_policy_net'], nn.LeakyReLU, oppoOpDim, act_dim).to(device)
    graph_list2.append(module)
    graph_list2[i].load_state_dict(torch.load(list2[i])['actor_architecture_state_dict'])
    mean2, var2 = load_param_envO_return_mean_var(list2[i], env)
    mean_list2.append(mean2)
    var_list2.append(var2)
len1 = len(graph_list1)
len2 = len(graph_list2)

def normalizeObsAndObserve(obs):
    numRow = int(num_envs/(len1 + len2))
    action = torch.zeros((num_envs, act_dim))
    for i in range(len1):
        for j in range(numRow):
            obs[i*numRow + j, :] -= mean_list1[i]
            obs[i*numRow + j, :] /=  np.sqrt(var_list1[i])
            action[i*numRow + j, :] = graph_list1[i].architecture(torch.from_numpy(obs[i*numRow + j, :]).to(device)).detach()
    for i in range(len2):
        for j in range(numRow):
            obs[(i + len1)*numRow + j, :][:oppoOpDim] -= mean_list2[i]
            obs[(i + len1)*numRow + j, :][:oppoOpDim] /=  np.sqrt(var_list2[i])
            action[(i + len1)*numRow + j, :] = graph_list2[i].architecture(torch.from_numpy(obs[(i + len1)*numRow + j, :][:oppoOpDim]).to(device)).detach()
    for j in range(num_envs - numRow*(len1 + len2)):
        obs[(len1 + len2)*numRow + j, :][:oppoOpDim] -= mean_list2[len2 - 1]
        obs[(len1 + len2)*numRow + j, :][:oppoOpDim] /=  np.sqrt(var_list2[len2 - 1])
        action[(len1 + len2)*numRow + j, :] = graph_list2[len2 - 1].architecture(torch.from_numpy(obs[(len1 + len2)*numRow + j, :][:oppoOpDim]).to(device)).detach()
    return action.cpu().numpy()
def normalizeObsAndObserveOnce(obs, idx):
    action = torch.zeros((1, act_dim)).to(device)
    if(idx < len1):
        obs[0] -= mean_list1[idx]
        obs[0] /=  np.sqrt(var_list1[idx])
        action[0] = graph_list1[idx].architecture(torch.from_numpy(obs[0]).to(device))
    else:
        obs[0][:oppoOpDim] -= mean_list2[idx - len1]
        obs[0][:oppoOpDim] /=  np.sqrt(var_list2[idx - len1])
        action[0] = graph_list2[idx - len1].architecture(torch.from_numpy(obs[0][:oppoOpDim]).to(device))
    return action
def normalizeObsAndObserveOnce2(obs, idx):
    action = torch.zeros((1, act_dim))
    if(idx < len1):
        obs[0] -= mean_list1[idx]
        obs[0] /=  np.sqrt(var_list1[idx])
        action[0] = graph_list1[idx].architecture(torch.from_numpy(obs[0]))
    else:
        obs[0][:oppoOpDim] -= mean_list2[idx - len1]
        obs[0][:oppoOpDim] /=  np.sqrt(var_list2[idx - len1])
        action[0] = graph_list2[idx - len1].architecture(torch.from_numpy(obs[0][:oppoOpDim]))
    return action
  
def constructModeVec():
    modeVec = np.zeros((num_envs,), dtype=bool)
    for i in range(len1*int(num_envs/(len1 + len2))):
        modeVec[i] = True
    return modeVec

if mode == 'fight':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
    load_param_env(weight_path, env_visual)
    oppoMode = constructModeVec()
    oppoMode_visual = np.zeros((1,), dtype=bool)

    # opponent do not scale
    env.no_scaling()
    env_visual.no_scaling()
    # check every element is weith_pathO directory and make directory list1 and list2
    # while check if element name is self~ make list1, oppo~ make list2
        
    
    trial_sum = 0
    win_sum = 0
    env.turn_off_visualization()
    env.reset2(oppoMode)
    continueLearningUpdate = 7400
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
            loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
            loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

            # we create another graph just to demonstrate the save/load method
            env.save_scaling(saver.data_dir, str(update))
            
            ### update opponent ###
            # oppoIdx = (update//cfg['environment']['eval_every_n'])%len1
            # oppoModule = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim).to(device)
            # graph_list1[oppoIdx] = oppoModule
            # graph_list1[oppoIdx].load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])
            # mean1, var1 = load_param_envO_return_mean_var(saver.data_dir+"/full_"+str(update)+'.pt', env)
            # mean_list1[oppoIdx] = mean1
            # var_list1[oppoIdx] = var1
            # print("update opponent with current policy")
            ###

            idx = (update//cfg['environment']['eval_every_n'])%(len1 + len2)
            oppoMode_visual[0] = (idx < len1)
            print("control mode = ", oppoMode_visual[0])
            # load observation scaling from files of pre-trained model
            env_visual.load_scaling(saver.data_dir, update)
            done_sum_test = 0
            reward_sum_test = 0
            env_visual.turn_on_visualization()
            env_visual.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

            env_visual.reset2(oppoMode_visual)
            test_steps = math.floor(60 / cfg['environment']['control_dt'])
            for step in range(test_steps):
                with torch.no_grad():
                    frame_start = time.time()
                    obs_vt = env_visual.observe(False)
                    obs_vtO = env_visual.observe2()
                    action_vt = loaded_graph.architecture(torch.from_numpy(obs_vt).cpu())
                    action_vtO = normalizeObsAndObserveOnce(obs_vtO, idx)
                    reward, dones, _ = env_visual.step2(action_vt.cpu().detach().numpy(), action_vtO.cpu().detach().numpy())
                    done_sum_test += np.sum(dones)
                    reward_sum_test += np.sum(reward)

                    frame_end = time.time()
                    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

            average_ll_performance = reward_sum_test / test_steps
            average_dones = done_sum_test / test_steps
            env_visual.stop_video_recording()
            env_visual.turn_off_visualization()\

            print('(test--test--test-------------------------------------')
            print('{:>6}th iteration test'.format(update))
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
            print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
            print('---------------------------------------test--test--test\n')

        if(update % 10 == 9):
            ppo.writer.add_scalar('Env/win_rate', win_sum/(int(trial_sum == 0) + trial_sum), update)
            ppo.writer.add_scalar('Env/win_num', win_sum, update)
            ppo.writer.add_scalar('Env/trial_num', trial_sum, update)
            print("win_rate  : ", win_sum/(int(trial_sum == 0) + trial_sum))
            print("win_num   : ", win_sum)
            print("trial_num : ", trial_sum)
            trial_sum = 0
            win_sum = 0

        # actual training
        for step in range(n_steps):
            obs = env.observe()
            obsO = env.observe2()
            action = ppo.act(obs)
            actionO = normalizeObsAndObserve(obsO)
            reward, dones, wins = env.step2(action, actionO)
            reward_analyzer.add_reward_info(env.get_reward_info())
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_sum = reward_sum + np.sum(reward)
            trial_sum += np.sum(dones)
            win_sum += np.sum(wins)

        # take st step to get value obs
        obs = env.observe()
        obsO = env.observe2()
        ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
        average_ll_performance = reward_sum / total_steps
        average_dones = done_sum / total_steps
        avg_rewards.append(average_ll_performance)

        actor.update()
        actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))
        
        actionO = normalizeObsAndObserve(obsO)
        action = ppo.just_act(obs)
        r, d, w = env.step2(action, actionO)
        trial_sum += np.sum(d)
        win_sum += np.sum(w)

        # curriculum update. Implement it in Environment.hpp
        env.curriculum_callback()

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

if mode == 'fight_vt':
    load_param_env(weight_path, env_visual)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])
    env_visual.no_scaling()
    
    for i in range(len(graph_list1)):
        graph_list1[i].cpu()
    for i in range(len(graph_list2)):
        graph_list2[i].cpu()
    
    done_sum_test = 0
    reward_sum_test = 0
    env_visual.turn_on_visualization()
    # env_visual.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(0)+'.mp4')
    oppoMode_visual = np.zeros((1,), dtype=bool)
    for test_episode in range(10000):
        idx = test_episode%(len(graph_list1) + len(graph_list2))
        oppoMode_visual[0] = (idx < len(graph_list1))
        env_visual.reset2(oppoMode_visual)
        test_steps = math.floor(5 / cfg['environment']['control_dt'])
        for step in range(test_steps):
            with torch.no_grad():
                frame_start = time.time()                    
                obs_vt = env_visual.observe(False)
                obs_vtO = env_visual.observe2()                    
                action_vt = loaded_graph.architecture(torch.from_numpy(obs_vt).cpu())
                action_vtO = normalizeObsAndObserveOnce2(obs_vtO, idx)
                # action_vtO *= 0
                reward, dones, wins = env_visual.step2(action_vt.cpu().numpy(), action_vtO.cpu().numpy())
                done_sum_test += np.sum(dones)
                reward_sum_test += np.sum(reward)

                frame_end = time.time()
                wait_time = 2*cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)

        average_ll_performance = reward_sum_test / test_steps
        average_dones = done_sum_test / test_steps
        env_visual.stop_video_recording()
        print('(test--test--test-------------------------------------')
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('---------------------------------------test--test--test\n')
    env_visual.turn_off_visualization()

