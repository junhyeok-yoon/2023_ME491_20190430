## ME 491 2023 project - Junhyeok Yoon

### Dependencies
- raisim

### environment setting
- If you use conda (We recommend)
  1. install CUDA 11.3 if you have NVIDIA GPU
  2. Make an environment the same as TA`s computer, which evaluates your final project:  ```conda env create -f ME491_2023_project_conda.yaml```
  3. Activate conda environment: ```conda activate ME491_2023_project```

- If you do not use conda (TA`s computer setting)
  1. python 3.9.7
  2. pytorch v1.11.0
  3. CUDA 11.3

### Run(main policy)
1. Compile raisimgym **(If you fix c++ code, every time do this)**: 
- ```python3 setup develop```
2. Move to desired directory:
- ```cd ME491_2023_project/env/envs/rsg_anymal```
3. run runner.py of the task (without self-play, plain mode): - ```python3 ./runner.py -m plain```
4. run runner.py of the task (with self-play, fight mode):
-  ```python3 ./runner.py -m fight -w data/ME491_2023_project/MY_FOLDER_NAME/full_XXX.pt -ww data/ME491_2023_project/OPPONENT_FOLDER_NAME```
* Note that, in fight mode, you need to give opponent policies. You can make robot to compete with multiple policies within single trainging.
* See 20190430/auxiliary/[6-7][oppoPolicy] directory. The policies in the directory are used for final training of my policy.
* In this case, you should only make two kinds of opponent policy. One is same as my structure, and the other one is other structre of policy(whose structre is like env_oppo)
* You should revise some codes for make compatible with various opponent structres.

5. run runner.py for visualize (vt, fight_vt mode): 
- ```python3 ./runner.py -m vt -w data/ME491_2023_project/MY_FOLDER_NAME/full_XXX.pt```
- ```python3 ./runner.py -m fight_vt -w data/ME491_2023_project/MY_FOLDER_NAME/full_XXX.pt -ww data/ME491_2023_project/OPPONENT_FOLDER_NAME```



### Run(opponent policy)
Different structure of control policy can be learnt through compling \env_oppo folder instead of \env

### Test policy
test.py only can compete various controller.hpp each other.
1. Rename or make controller file whose format as 'test_AnymalController_XXXXXXXX.hpp' in order to distinguish with other 'AnymalCOntroller_XXXXXXXX.hpp files
- In current repository, 'test_AnymalController_20190430.hpp', and 'test_AnymalController_20190431' is given for the example. These two files have same observation and action space.
2. Change the opponent observation dimension on tester.py line 44.
```python
ob_dimO = 95 # opponent's observation dimension
```
3. Compile raisimgym: ```python3 setup develop```
4. run tester.py of the task with policy (only test competition): ``` python3 ME491_2023_project/env/envs/rsg_anymal/tester.py -w data/ME491_2023_project/MY_FOLDER_NAME/full_XXX.pt -ww data/ME491_2023_project/OPPONENT_FOLDER_NAME/full_XXX.pt```

### Retrain policy
1. run runner.py of the task with policy: 
* uncomment below line in runner.py line 135~136 as below:

```python
 # load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
# load_param_env(weight_path, env_visual)
```

* Just add ```-w ~``` or(and) ```-ww ~``` on original command

### Debugging
1. Compile raisimgym with debug symbols: ```python3 setup develop --Debug```. This compiles <YOUR_APP_NAME>_debug_app
2. Run it with Valgrind. I strongly recommend using Clion for 
