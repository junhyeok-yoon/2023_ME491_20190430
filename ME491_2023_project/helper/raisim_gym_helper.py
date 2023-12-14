from shutil import copyfile
import datetime
import os
import ntpath
import torch
import numpy as np

class ConfigurationSaver:
    def __init__(self, log_dir, save_items):
        self._data_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir)

        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                copyfile(save_item, self._data_dir + '/' + base_file_name)

    @property
    def data_dir(self):
        return self._data_dir


def tensorboard_launcher(directory_path, port = 24681):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    # tb.configure(argv=[None, '--logdir', directory_path])
    tb.configure(argv=[None, '--logdir', directory_path, '--port', str(port), '--host', '0.0.0.0'])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    # webbrowser.open_new(url)


def load_param(weight_path, env, actor, critic, optimizer, data_dir):
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nRetraining from the checkpoint:", weight_path+"\n")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    mean_csv_path = weight_dir + 'mean' + iteration_number + '.csv'
    var_csv_path = weight_dir + 'var' + iteration_number + '.csv'
    items_to_save = [weight_path, mean_csv_path, var_csv_path, weight_dir + "cfg.yaml", weight_dir + "Environment.hpp"]

    if items_to_save is not None:
        pretrained_data_dir = data_dir + '/pretrained_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1]
        os.makedirs(pretrained_data_dir)
        for item_to_save in items_to_save:
            copyfile(item_to_save, pretrained_data_dir+'/'+item_to_save.rsplit('/', 1)[1])

    # load observation scaling from files of pre-trained model
    env.load_scaling(weight_dir, iteration_number)

    # load actor and critic parameters from full checkpoint
    checkpoint = torch.load(weight_path)
    actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    
    reuse_std_tensor = actor.distribution.std.data.clone()
    for i in range(12):
        reuse_std_tensor[i] = max(1.5, reuse_std_tensor[i])
    actor.distribution.enforce_minimum_std(reuse_std_tensor) # reset std
    critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def load_param_env(weight_path, env):
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nRetraining from the checkpoint env only:", weight_path+"\n")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    # load observation scaling from files of pre-trained model
    env.load_scaling(weight_dir, iteration_number)

def load_paramO(weight_path, env, loaded_graph, data_dir):
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nRetraining from the checkpoint of opponent:", weight_path+"\n")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    mean_csv_path = weight_dir + 'mean' + iteration_number + '.csv'
    var_csv_path = weight_dir + 'var' + iteration_number + '.csv'
    items_to_save = [weight_path, mean_csv_path, var_csv_path, weight_dir + "cfg.yaml", weight_dir + "Environment.hpp"]

    if items_to_save is not None:
        pretrained_data_dir = data_dir + '/Opretrained_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1]
        os.makedirs(pretrained_data_dir)
        for item_to_save in items_to_save:
            copyfile(item_to_save, pretrained_data_dir+'/'+item_to_save.rsplit('/', 1)[1])

    # load observation scaling from files of pre-trained model
    env.load_scalingO(weight_dir, iteration_number)

    # load actor and critic parameters from full checkpoint
    checkpoint = torch.load(weight_path)
    loaded_graph.load_state_dict(checkpoint['actor_architecture_state_dict'])

def load_param_envO_return_mean_var(weight_path, env):
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nRetraining from the checkpoint of opponent env only:", weight_path+"\n")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'
    
    mean_file_name = weight_dir + "/mean" + str(iteration_number) + ".csv"
    var_file_name = weight_dir + "/var" + str(iteration_number) + ".csv"
    return np.loadtxt(mean_file_name, dtype=np.float32), np.loadtxt(var_file_name, dtype=np.float32)