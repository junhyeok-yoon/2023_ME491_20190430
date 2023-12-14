from ruamel.yaml import YAML, dump, RoundTripDumper
from ME491_2023_project.env.RaisimGymVecEnv_for_test import RaisimGymVecEnv as VecEnv
import ME491_2023_project.algo.ppo.module as ppo_module
from importlib import import_module
import os
import math
import time
import torch
import argparse
import re

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# import module from the built environment library
files = os.listdir(task_path)
pattern = re.compile(r'test_AnymalController_(\d+).hpp')
for file in files:
    match = pattern.match(file)
    if match:
        student_id = match.group(1)
        # print(student_id)
module_name = f"ME491_2023_project.env.bin.SI{student_id}vsSI20190431"
module = import_module(module_name)

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
parser.add_argument('-ww', '--weightO', help='trained opponent weight path', type=str, default='')
args = parser.parse_args()

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1
cfg['environment']['render'] = True

env = VecEnv(module.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
ob_dimO = ob_dim
act_dim = env.num_acts

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'
weight_pathO = args.weightO
iteration_numberO = weight_pathO.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dirO = weight_pathO.rsplit('/', 1)[0] + '/'

if weight_path == "" or weight_pathO == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    loaded_graphO = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dimO, act_dim)
    loaded_graphO.load_state_dict(torch.load(weight_pathO)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.load_scalingO(weight_dirO, int(iteration_numberO))

    max_steps = 1000000

    for step in range(max_steps):
        with torch.no_grad():
            frame_start = time.time()
            obs = env.observe(False)
            obsO = env.observe2()
            action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu()).cpu().detach().numpy()
            action_llO = loaded_graphO.architecture(torch.from_numpy(obsO).cpu()).cpu().detach().numpy()
            env.step2(action_ll, action_llO)
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)
