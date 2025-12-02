import numpy as np
import torch
import collections
import sys

import glob
import os
import re
import matplotlib.pyplot as plt
import pandas as pd 


import numpy as np





def qlearning_dataset_plus(env, dataset=None, terminate_on_end=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    use_timeouts = 'timeouts' in dataset

    start_, obs_, next_obs_, action_, next_act_, reward_ = [], [], [], [], [], []
    done_, timeout_, traj_, step_ = [], [], [], []
    qvel_, qpos_ = [], []
    has_qvel = 'qvel' in dataset
    has_qpos = 'qpos' in dataset

    traj_length = []
    episode_end_list = []
    traj_bounds = []  # [(start_i, end_i)] 真实轨迹边界（含终止步）

    episode_step = 0
    curr_traj_len = 0
    traj_idx = -1
    curr_traj_start_i = None

    for i in range(N - 1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        act = dataset['actions'][i].astype(np.float32)
        next_act = dataset['actions'][i + 1].astype(np.float32)
        rew = dataset['rewards'][i].astype(np.float32)

        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = bool(dataset['timeouts'][i])
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        if episode_step == 0:
            traj_idx += 1
            start_flag = True
            curr_traj_start_i = i
        else:
            start_flag = False

        # 不采样末步，但仍记真实终止
        if (not terminate_on_end) and final_timestep:
            episode_end_list.append(i)
            traj_bounds.append((curr_traj_start_i, i))
            traj_length.append(curr_traj_len)
            episode_step = 0
            curr_traj_len = 0
            continue

        # 采样该 transition
        start_.append(start_flag)
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(act)
        next_act_.append(next_act)
        reward_.append(rew)
        done_.append(done_bool)
        timeout_.append(final_timestep)
        traj_.append(traj_idx)
        step_.append(i)
        if has_qvel:
            qvel_.append(np.array(dataset['qvel'][i], dtype=np.float32))
        if has_qpos:
            qpos_.append(np.array(dataset['qpos'][i], dtype=np.float32))

        episode_step += 1
        curr_traj_len += 1

        if done_bool or final_timestep:
            episode_end_list.append(i)
            traj_bounds.append((curr_traj_start_i, i))
            traj_length.append(curr_traj_len)
            episode_step = 0
            curr_traj_len = 0

    # ---------- 从当前到终止的 return / len_traj / avg_return ----------
    M = len(step_)
    ret_to_end = np.zeros(M, dtype=np.float32)
    len_to_end = np.zeros(M, dtype=np.int32)

    from collections import defaultdict
    traj_to_positions = defaultdict(list)  # t_idx -> [pos_in_output]
    for pos, t in enumerate(traj_):
        traj_to_positions[t].append(pos)

    rewards_all = dataset['rewards'].astype(np.float32)

    # 记录每条轨迹的总长度与总回报（用于逐级递增 reward）
    traj_total_len = {}
    traj_total_return = {}

    for t_idx, (s_i, e_i) in enumerate(traj_bounds):
        if s_i is None or e_i is None or e_i < s_i:
            traj_total_len[t_idx] = 0
            traj_total_return[t_idx] = 0.0
            continue

        seg = rewards_all[s_i: e_i + 1]               # 含终止步
        prefix = np.zeros(len(seg) + 1, dtype=np.float32)
        prefix[1:] = np.cumsum(seg, dtype=np.float32)

        L_total = int(e_i - s_i + 1)
        R_total = float(prefix[-1])
        traj_total_len[t_idx] = L_total
        traj_total_return[t_idx] = R_total

        positions = traj_to_positions.get(t_idx, [])
        if not positions:
            continue

        for pos in positions:
            i = step_[pos]
            off = i - s_i
            R = float(prefix[-1] - prefix[off])       # 从 i 到 e_i 的累计回报
            L = int(e_i - i + 1)                      # 从 i 到 e_i 的长度（含终止步）
            ret_to_end[pos] = R
            len_to_end[pos] = L

    avg_return = ret_to_end / np.maximum(len_to_end, 1)

    # 内置：-exp(avg_return)
    reward_builtin = -np.exp(avg_return).astype(np.float32)

    # 逐级递增 reward: (traj_total_len - len_to_end) / L_total * R_total
    reward_increasing = np.zeros(M, dtype=np.float32)
    for pos, t in enumerate(traj_):
        L_total = traj_total_len.get(t, 0)
        R_total = traj_total_return.get(t, 0.0)
        if R_total == 0.0:
            reward_increasing[pos] = 0.0
        else:
            reward_increasing[pos] = float((L_total - int(len_to_end[pos])) / max(L_total, 1) * R_total)

    # ========= 新增：每个样本所属轨迹的“总回报” =========
    reward_return = np.zeros(M, dtype=np.float32)
    for pos, t in enumerate(traj_):
        reward_return[pos] = np.float32(traj_total_return.get(t, 0.0))
    # ===============================================

    data = {
        "start": np.array(start_, dtype=bool),
        "observations": np.array(obs_, dtype=np.float32),
        "actions": np.array(action_, dtype=np.float32),
        "next_observations": np.array(next_obs_, dtype=np.float32),
        "next_actions": np.array(next_act_, dtype=np.float32),
        "rewards": np.array(reward_, dtype=np.float32),
        "terminals": np.array(done_, dtype=bool),
        "timeouts": np.array(timeout_, dtype=bool),
        "trajectory": np.array(traj_, dtype=np.int64),
        "step": np.array(step_, dtype=np.int64),

        # 逐条指标
        "return": ret_to_end.astype(np.float32),          # 从当前到终止的累计回报
        "len_traj": len_to_end.astype(np.int32),          # 从当前到终止的长度（含终止步）
        "avg_return": avg_return.astype(np.float32),      # 平均回报
        "reward_builtin": reward_builtin,                 # -exp(avg_return)
        "reward_increasing": reward_increasing,           # (L_total - len_to_end) / L_total * R_total
        "reward_return": reward_return,                   # ← 新增：样本所属轨迹的总回报
    }
    if has_qvel:
        data["qvel"] = np.array(qvel_, dtype=np.float32)
    if has_qpos:
        data["qpos"] = np.array(qpos_, dtype=np.float32)

    return traj_length, episode_end_list, data



def qlearning_dataset_all(env, task=None, dataset=None, terminate_on_end=False, num_traj=0, **kwargs):
    """
    导出全部样本，不丢终止步
    返回
      traj_length         每个轨迹长度列表
      episode_end_list    截止到每个轨迹末尾的累计样本数
      data                标准 Q 学习格式的数据字典
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    print("*"*100) 
    print(dataset['rewards'].shape[0])
    # print(dataset["rewards"].shape, max(dataset["rewards"]), min(dataset["rewards"]), sum(dataset["rewards"]))

    has_next_obs = "next_observations" in dataset
    has_next_act = "next_actions" in dataset
    use_timeouts = "timeouts" in dataset

    print(use_timeouts)
    # sys.exit()

    N = dataset["rewards"].shape[0]
    need_shift = (not has_next_obs) or (not has_next_act)
    N_iter = N - 1 if need_shift else N

    start_ = []
    obs_ = []
    next_obs_ = []
    action_ = []
    next_act_ = []
    reward_ = []
    done_ = []
    timeout_ = []
    traj_ = []
    step_ = []
    qvel_ = []
    qpos_ = []

    episode_step = 0
    traj_step = 0
    traj_count = 0

    traj_length = []
    episode_end_list = []

    for i in range(N_iter):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = (
            dataset["next_observations"][i].astype(np.float32)
            if has_next_obs else dataset["observations"][i + 1].astype(np.float32)
        )
        action = dataset["actions"][i].astype(np.float32)
        new_act = (
            dataset["next_actions"][i].astype(np.float32)
            if has_next_act else dataset["actions"][i + 1].astype(np.float32)
        )

        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        # qpos qvel 若不存在就跳过收集
        if "infos/qpos" in dataset:
            qpos = dataset["infos/qpos"][i].astype(np.float32)
        else:
            qpos = None
        if "infos/qvel" in dataset:
            qvel = dataset["infos/qvel"][i].astype(np.float32)
        else:
            qvel = None

        if use_timeouts:
            final_timestep = bool(dataset["timeouts"][i])
        else:
            # 退而求其次的近似
            final_timestep = done_bool

        is_end = done_bool or final_timestep

        # 写入样本
        start_.append(episode_step == 0)
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_act_.append(new_act)
        reward_.append(reward)
        done_.append(done_bool)
        timeout_.append(final_timestep if use_timeouts else done_bool)
        traj_.append(traj_count)
        step_.append(i)
        if qvel is not None:
            qvel_.append(qvel)
        if qpos is not None:
            qpos_.append(qpos)

        episode_step += 1
        traj_step += 1

        # 回合结束后再结算并可截断
        if is_end:
            traj_length.append(episode_step)
            episode_end_list.append(traj_step)
            episode_step = 0
            traj_count += 1
            if num_traj > 0 and traj_count >= num_traj:
                break

    # 如果最后还有未封口的尾部回合才补统计
    if episode_step > 0:
        traj_length.append(episode_step)
        episode_end_list.append(traj_step)

    data = {
        "start": np.array(start_),
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "next_actions": np.array(next_act_), # 下一个actions
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "timeouts": np.array(timeout_),
        "trajectory": np.array(traj_), # 第几条traj
        "step": np.array(step_), # 数据集中的数据索引
    }
    if qvel_:
        data["qvel"] = np.array(qvel_)
    if qpos_:
        data["qpos"] = np.array(qpos_)

    return traj_length, episode_end_list, data # traj_length是 traj 长度的列表，episode_end_list




def qlearning_dataset(env, task=None, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    # 归一化
    # min_reward = dataset['rewards'].min()
    # max_reward = dataset['rewards'].max()
    # dataset['rewards'] = (dataset['rewards'] - min_reward) / (max_reward - min_reward)
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False
    # 更健壮的写法（处理task可能为None的情况）
    is_antmaze = bool(task) and 'antmaze' in str(task).lower()
    # print(is_antmaze)


    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # print(dataset['rewards'])
    print(sum(dataset['rewards']))
    # print(min(dataset['rewards']))
    # print(max(dataset['rewards']))
    # print(len(dataset['terminals']))
    # sys.exit()

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True


    episode_step = 0
    
    traj_min, traj_max, traj_diff = 0, 0, 0
    traj_r_list = []
    traj_var_list = []
    traj_diff_list = []
    traj_list = []
    
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        
        
        # # traj
        # traj_max = max(traj_max, reward)
        # traj_min = min(traj_min, reward)
        # traj_r_list.append(reward)


        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
            # print("1")
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
            # print("2")
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            # print("5")
            if not is_antmaze:
                continue  
        if done_bool or final_timestep:
            traj_list.append(episode_step) 
            episode_step = 0
            
            # traj
            # traj_r_tensor = torch.tensor(traj_r_list)
            # traj_r_var = torch.var(traj_r_tensor)
            # traj_var_list.append(traj_r_var.item())
            
            traj_diff = traj_max - traj_min
            traj_diff_list.append(traj_diff)
            traj_min, traj_max, traj_diff = 0, 0, 0
            # print("3")
            if (not has_next_obs) and (not is_antmaze):
                # print("4")
                continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1 
        
    # print(len(traj_diff_list))
    # print(traj_diff_list[:50])
    # print(max(traj_diff_list))
    # print(min(traj_diff_list))
    # print((sum(traj_diff_list) / len(traj_diff_list)))
    # print(traj_list)
    # sys.exit()
    # # traj_var_tensor = torch.tensor(traj_var_list)

    # # # 使用 PyTorch 提供的函数计算方差
    # # variance = torch.var(traj_diff_tensor)
    # # print("方差:", variance.item())
    
    
    # plt.figure()
    # # plt.hist(reward_arr, bins=150, alpha=0.5, color='blue')
    # plt.hist(traj_diff_list, bins=50, alpha=0.5, color='blue', density=True) 
    # plt.xlabel('Traj Reward')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    
    # # 用文件名命名图像并保存
    # plt.savefig('/home/qiaonan/ORL/run_example/distribution.png')
    # plt.close()
    
    # sys.exit()


    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }

def qlearning_dataset_checktraj(env, task=None, dataset=None, terminate_on_end=False, num_traj=0, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    print("*"*100)
    print(dataset['rewards'].shape[0])
    
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False
    has_next_act = True if 'next_actions' in dataset.keys() else False
    is_antmaze = bool(task) and 'antmaze' in str(task).lower()
    print(dataset["rewards"].shape, max(dataset["rewards"]), min(dataset["rewards"]), sum(dataset["rewards"]))
    # print(dataset["infos/qvel"][0])
    # sys.exit()

    N = dataset['rewards'].shape[0]
    start_ = []
    obs_ = []
    next_obs_ = []
    action_ = []
    next_act_ = []
    reward_ = []
    done_ = []
    timeout_ = []
    traj_ = []
    step_ = []
    qvel_ = []
    qpos_ = []
    # timeout_[0]

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    print(dataset['timeouts'][:100])
    # sys.exit()

    episode_step = 0
    traj_step = 0
    check_traj = 0
    check_traj1 = 0
    check_traj2 = 0
    traj_length = []
    episode_end_list = []
    
    
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        if has_next_act:
            new_act = dataset['next_actions'][i].astype(np.float32)
        else:
            new_act = dataset['actions'][i+1].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        qpos = dataset["infos/qpos"][i].astype(np.float32)
        qvel = dataset["infos/qvel"][i].astype(np.float32)

        # sys.exit()

        # timeouts = False
        # print(i)


        if use_timeouts:
            final_timestep = dataset['timeouts'][i]

            # if final_timestep:
            #     print(final_timestep,episode_step,check_traj)
        else:
            # print(1)
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # print(2)
            # Skip this transition and don't apply terminals on the last step of an episode
            # print("if (not terminate_on_end) and final_timestep:")
            traj_length.append(episode_step)
            episode_end_list.append(traj_step)
            if (len(timeout_) > 1) and (not use_timeouts):
                timeout_[-1] = True
            episode_step = 0
            check_traj += 1
            if not is_antmaze:
            # done_bool = False
                continue  
        if done_bool or final_timestep:
        # if ((not has_next_obs) and (not is_antmaze)) or ((done_bool or final_timestep) and is_antmaze):
            # print(3)
            traj_length.append(episode_step)
            episode_end_list.append(traj_step)
            # timeout_[-1] = True
            episode_step = 0
            check_traj += 1
            
            
            if (not has_next_obs) and (not is_antmaze):
                # print(4)
                # timeout_[-1] = True
                # print("yes")
                continue


        # print("yes")
        # print(5)
        # sys.exit()
        if episode_step == 0:
            start_.append(True)
        else:
            start_.append(False)
        # print(episode_step)
        # print(start_)
        # sys.exit()
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_act_.append(new_act)
        reward_.append(reward)
        done_.append(done_bool)
        traj_.append(check_traj)
        step_.append(i)
        qvel_.append(qvel)
        qpos_.append(qpos)
        if use_timeouts:
            timeout_.append(final_timestep)
        else:
            timeout_.append(done_bool)

        episode_step += 1
        traj_step += 1

        if num_traj > 0 and num_traj==check_traj:
            break
    
    traj_length.append(episode_step)
    episode_end_list.append(traj_step)

    # print(traj_step)
    # print(episode_end_list[-1:])
    # print("hi")
    # print(sum(traj_length))
    # print(len(traj_length))
    # print(len(episode_end_list))
    # print("hi")
    # print(len(traj_length))
    # sys.exit()
    # print(traj_step)
    # print(traj_length[-10:])
    # a=[]
    # a=episode_end_list-traj_length
    # print(a[-10:])


    return traj_length, episode_end_list,  {
        'start': np.array(start_),
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_act_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'timeouts': np.array(timeout_),
        'trajectory': np.array(traj_),
        'step': np.array(step_),
        'qvel': np.array(qvel_),
        'qpos': np.array(qpos_),
    }


def qlearning_dataset_wo_timeout(env, dataset=None, terminate_on_end=False, num_traj=0, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False
    has_next_act = True if 'next_actions' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_act_ = []
    reward_ = []
    done_ = []
    timeout_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    traj_step = 0
    check_traj = 0
    check_traj1 = 0
    check_traj2 = 0
    traj_length = []
    episode_end_list = []
    
    
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        if has_next_act:
            new_act = dataset['next_actions'][i].astype(np.float32)
        else:
            new_act = dataset['actions'][i+1].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        # sys.exit()

        # timeouts = False


        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
            # print("yes")
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        final_timestep = False
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            # print("if (not terminate_on_end) and final_timestep:")
            traj_length.append(episode_step)
            episode_end_list.append(traj_step)
            timeout_[-1] = True
            episode_step = 0
            check_traj += 1
            continue  
        if done_bool or final_timestep:
            traj_length.append(episode_step)
            episode_end_list.append(traj_step)
            # timeout_[-1] = True
            episode_step = 0
            check_traj += 1
            
            
            if not has_next_obs:
                # timeout_[-1] = True
                # print("yes")
                continue



        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_act_.append(new_act)
        reward_.append(reward)
        done_.append(done_bool)
        timeout_.append(done_bool)


        episode_step += 1
        traj_step += 1

        if num_traj > 0 and num_traj==check_traj:
            break
    
    traj_length.append(episode_step)
    episode_end_list.append(traj_step)

    # print(traj_step)
    # print(episode_end_list[-1:])
    # print("hi")
    # print(sum(traj_length))
    # print(len(traj_length))
    # print(len(episode_end_list))
    # print("hi")
    # print(len(traj_length))
    # sys.exit()
    # print(traj_step)
    # print(traj_length[-10:])
    # a=[]
    # a=episode_end_list-traj_length
    # print(a[-10:])


    return traj_length, episode_end_list,  {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_act_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'timeouts': np.array(timeout_),
    }



def qlearning_dataset_traj(env, dataset=None, terminate_on_end=False, num_traj=0, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal 
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    print(dataset['observations'].shape)
    # print(dataset['observations'][245:255])
    print(dataset['actions'].shape)
    # print(dataset['actions'][245:255])
    # print(dataset['terminals'][245:255])
    
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True
    
            
    keys_list = list(dataset.keys()) 
    print(keys_list)
    # # data = dataset['terminals']
    # # print(data[-10:])
    # true_indices = np.where(dataset['timeouts'])[0]  
    # print(true_indices)
    # print("*"*100)
    # true_indices = np.where(dataset['terminals'])[0]  
    # print(true_indices)
    # sys.exit()

    episode_step = 0
    check_traj = 0
    check_traj1 = 0
    check_traj2 = 0
    traj_length = []
    
    
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        
        

        


        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
            # print("dataset['timeouts'][i]")
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
            # print("final_timestep = (episode_step == env._max_episode_steps - 1)")
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            # print("if (not terminate_on_end) and final_timestep:")
            traj_length.append(episode_step)
            episode_step = 0
            check_traj += 1
            check_traj1 += 1
            continue  
        if done_bool or final_timestep:
            # print("if done_bool or final_timestep:")
            traj_length.append(episode_step)
            episode_step = 0

            # check num of traj
            check_traj += 1
            check_traj2 += 1
            
            
            if not has_next_obs:
                continue
            


        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

        if num_traj > 0 and num_traj==check_traj:
            # print(check_traj)
            print("yes")
            break
        
    # print(len(traj_diff_list))
    # print(traj_diff_list[:10])
    # print(max(traj_diff_list))
    # print(min(traj_diff_list))
    # print((sum(traj_diff_list) / len(traj_diff_list)))
    
    # # traj_var_tensor = torch.tensor(traj_var_list)

    # # # 使用 PyTorch 提供的函数计算方差
    # # variance = torch.var(traj_diff_tensor)
    # # print("方差:", variance.item())
    
    
    # plt.figure()
    # # plt.hist(reward_arr, bins=150, alpha=0.5, color='blue')
    # plt.hist(traj_diff_list, bins=50, alpha=0.5, color='blue', density=True) 
    # plt.xlabel('Traj Reward')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    
    # # 用文件名命名图像并保存
    # plt.savefig('/home/qiaonan/ORL/run_example/distribution.png')
    # plt.close()
    
    
    # print(traj_length)
    # print(len(done_))
    # print(sum(done_))
    # sys.exit()
    
    
    # file_path = '/home/qiaonan/ORL/test/env_traj_0224.txt' 
    # if not os.path.exists(os.path.dirname(file_path)):
    #     os.makedirs(os.path.dirname(file_path))  
    # with open(file_path, 'a', encoding='utf-8') as file: 
    #     file.write(str(sum(done_))+' / '+str(len(done_))+'\n')  
    #     file.write('num of traj = '+str(len(traj_length))+'\n')  
    #     for item in traj_length:  
    #         file.write(str(item) + ' ') 
    #     file.write('\n') 

    #     # 写入所有键的名称
    #     keys_list = list(dataset.keys())
    #     file.write("所有键的名称：" + str(keys_list) + '\n')

    #     # 写入每个键对应的值中的第一个元素
    #     file.write("每个键对应的值中的第一个元素：\n")
    #     for key in keys_list:
    #         value = dataset[key]
    #         # 检查对应的值是否为可迭代对象，并且长度大于 0
    #         # if isinstance(value, (list, tuple, str)) and len(value) > 0: 
    #         file.write(f"{key}: {value}\n")
    
    # # 获取所有键的列表
    # keys_list = list(dataset.keys())
    # print("所有键的名称：", keys_list)

    # # 输出每个键对应的值中的第一个元素
    # print("每个键对应的值中的第一个元素：")
    # for key in keys_list:
    #     value = dataset[key]
    #     print(f"{key}: {value}")
        # if isinstance(value, (list, tuple, str)) and len(value) > 0:
        #     print(f"{key}: {value[0]}")
        # else:
        #     print(f"{key}: 无有效第一个元素")
    # print(dataset["rewards"][0])
    
    
    # print(check_traj)
    # print(check_traj1)
    # print(check_traj2)
    # print(traj_length)
    # sys.exit()


    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }
    
def qlearning_dataset_distance(env, dataset=None, terminate_on_end=False, num_traj=0, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    print(dataset['observations'].shape)
    print(dataset['observations'][245:255])
    print(dataset['actions'].shape)
    print(dataset['actions'][245:255])
    print(dataset['terminals'][245:255])
    
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    in_num_traj, out_num_traj = num_traj, num_traj
    k=602
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True
    
            
    keys_list = list(dataset.keys()) 
    print(keys_list)
    # data = dataset['terminals']
    # print(data[-10:])
    true_indices = np.where(dataset['timeouts'])[0]  
    print(true_indices)
    print("*"*100)
    true_indices = np.where(dataset['terminals'])[0]  
    print(true_indices)
    # sys.exit()

    episode_step = 0
    check_traj = 0
    check_traj1 = 0
    check_traj2 = 0
    traj_length = []

    traj_stack = []
    
    
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        
        

        


        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
            # print("dataset['timeouts'][i]")
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
            # print("final_timestep = (episode_step == env._max_episode_steps - 1)")
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            # print("if (not terminate_on_end) and final_timestep:")
            
            traj, obs_, action_, next_obs_, reward_, done_ = nem_traj(obs_, action_, next_obs_, reward_, done_)
            traj_stack.append(traj)
            traj_length.append(episode_step)
            episode_step = 0
            check_traj += 1
            check_traj1 += 1
            

            continue  
        if done_bool or final_timestep:
            # print("if done_bool or final_timestep:")s
            traj, obs_, action_, next_obs_, reward_, done_ = nem_traj(obs_, action_, next_obs_, reward_, done_)
            traj_stack.append(traj)
            traj_length.append(episode_step)
            episode_step = 0

            # check num of traj
            check_traj += 1
            check_traj2 += 1
            
            
            if not has_next_obs:
                continue

        # if (num_traj > 0 and num_traj > check_traj):
        if episode_step < 3:
            print(f"obs:{obs}")
            print(f"new_obs:{new_obs}")
        if episode_step == 3:
            print("episode_step == 3")
            # sys.exit()
        
        
        
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

        if num_traj > 0 and num_traj + k==check_traj:
            # print(check_traj)
            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step += 1
            break

    
    
    print(check_traj)
    print(check_traj1)
    print(check_traj2)
    print(traj_length)
    
    print("*"*100)
    print(len(traj_stack))
    print(len(traj_stack[0:-1]))
    print(len(traj_stack[-1:]))
    print(len(traj_stack[0][0]))
    print(dataset['observations'].shape)
    # sys.exit()
    
    
    # # find traj
    # train_traj = []
    # test_traj = traj_stack[15]
    # test_traj_list = traj_stack[9:-1]
    # for lst in traj_stack[0:num_traj]:  
    # # 将每个列表的元素添加到merged_list中  
    #     train_traj.extend(lst)  
    # print(len(train_traj))
    # print(len(test_traj_list))

    # for test_traj in test_traj_list:
    #     distance = []
    #     for a in test_traj: 
    #         norm_diff = []
    #         a_array = np.array(a)
    #         for b in train_traj: 
    #             b_array = np.array(b) 
    #             diff = a_array - b_array 
    #             norm_diff.append(np.linalg.norm(diff))
    #         distance.append(min(norm_diff))
        
    #     min_max = [min(distance), max(distance)]
    #     print(min_max)
    # print(len(distance))
    # print(min(distance))
    # print(max(distance))
    # sys.exit()

    # # halfcheetah-medium-replay
    # train_traj = []
    # test_traj = traj_stack[18]
    # test_traj_list = traj_stack[9:-1]
    # for lst in traj_stack[0:num_traj]:  
    # # 将每个列表的元素添加到merged_list中  
    #     train_traj.extend(lst)  
    # print(len(train_traj))
    # print(len(test_traj_list))
    # distance = []
    # for a in test_traj: 
    #     norm_diff = []
    #     a_array = np.array(a)
    #     for b in train_traj: 
    #         b_array = np.array(b) 
    #         diff = a_array - b_array 
    #         norm_diff.append(np.linalg.norm(diff))
    #     distance.append(min(norm_diff))
        
    # min_max = [min(distance), max(distance)]



    # # walker-expert
    # train_traj = []
    # test_traj = traj_stack[21]
    # test_traj_list = traj_stack[9:-1]
    # for lst in traj_stack[0:num_traj]:  
    # # 将每个列表的元素添加到merged_list中  
    #     train_traj.extend(lst)  
    # print(len(train_traj))
    # print(len(test_traj_list))
    # distance = []
    # for a in test_traj: 
    #     norm_diff = []
    #     a_array = np.array(a)
    #     for b in train_traj: 
    #         b_array = np.array(b) 
    #         diff = a_array - b_array 
    #         norm_diff.append(np.linalg.norm(diff))
    #     distance.append(min(norm_diff))
        

    # min_max = [min(distance), max(distance)]
    # print(min_max)
    # print(len(distance))
    # print(min(distance))
    # print(max(distance))
    # sys.exit()





    #  hopper-expert
    train_traj = []
    # test_traj = traj_stack[-1]
    #halfcheetah
    test_traj = traj_stack[18]
    test_traj_list = traj_stack[9:-1]
    for lst in traj_stack[0:num_traj]:  
    # 将每个列表的元素添加到merged_list中  
        train_traj.extend(lst)  
    print(len(train_traj))
    print(len(test_traj_list))
    distance = []
    # for a in test_traj: 
    #     norm_diff = []
    #     a_array = np.array(a)
    #     for b in train_traj: 
    #         b_array = np.array(b) 
    #         diff = a_array - b_array 
    #         norm_diff.append(np.linalg.norm(diff))
    #     distance.append(min(norm_diff))
    #     min_max = [min(distance), max(distance)]
    
    train_traj_array = np.array(train_traj)
    test_traj_array = np.array(test_traj)
    for a in test_traj_array:
        # a[np.newaxis, :] 使得a变成一个二维数组，便于广播
        # 计算a与train_traj_array中所有点的差值，然后求范数
        diffs = a - train_traj_array
        norms = np.linalg.norm(diffs, axis=1)
        distance.append(np.min(norms))
        
    # print((distance))
    # print(min_max)
    indices_0 = [idx for idx, val in enumerate(distance) if 0 < val < 6]  
    print(len(indices_0))
    indices_1 = [idx for idx, val in enumerate(distance) if 6 < val < 10]  
    print(len(indices_1))
    indices_2 = [idx for idx, val in enumerate(distance) if 10 < val < 13]  
    print(len(indices_2))
    indices_3 = [idx for idx, val in enumerate(distance) if 13 < val < 30]  
    print(len(indices_3))
    values_greater_than_3 = [distance[idx] for idx in indices_3]  
    # print(values_greater_than_3)


    # 绘制直方图  
    plt.hist(distance, bins=100, edgecolor='black', alpha=0.7)  
    plt.xlabel('Values', fontdict={'size': 15})  
    plt.ylabel('Frequency', fontdict={'size': 15})  
    plt.title(f'hopper-expert-10', fontdict={'size': 15})
    dri = f'/home/qiaonan/ORL/offlinerlkit/utils/p_hoe_{num_traj}k.png'
    print(f"dir: {dri}")
    plt.savefig(dri)
    plt.show()

    # print(train_traj)
    print(len(train_traj))
    print(len(train_traj[0]))
    total_elements = sum(len(sublist) for sublist in train_traj)
    print(total_elements)
    print(num_traj)

    sys.exit()
    
    indices_list = [indices_0, indices_1, indices_2, indices_3]
    
    in_traj_obs = traj_stack[0]
    out_traj_obs = traj_stack[15]
    return  indices_list, in_traj_obs, out_traj_obs
    
    

    
    # sys.exit()


    # return {
    #     'observations': np.array(obs_),
    #     'actions': np.array(action_),
    #     'next_observations': np.array(next_obs_),
    #     'rewards': np.array(reward_),
    #     'terminals': np.array(done_),
    # }


def nem_traj(obs_, action_, next_obs_, reward_, done_):
    traj_obs = np.array(obs_)
    # {
    #     'observations': np.array(obs_),
    #     # 'actions': np.array(action_),
    #     # 'next_observations': np.array(next_obs_),
    #     # 'rewards': np.array(reward_),
    #     # 'terminals': np.array(done_),
    # }
    
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    
    return traj_obs, obs_, action_, next_obs_, reward_, done_
    
    

def qlearning_dataset_rewarr(env, dataset=None, terminate_on_end=False, reward_arrays=None, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False
    



    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    reward_arrays_ = [[] for _ in range(len(reward_arrays))]


    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    episode_step_list = []
    c1, c2 = 0, 0


    

    
    
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        

        
        
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step_list.append(episode_step)
            episode_step = 0
            c1 += 1
            continue  
        if done_bool or final_timestep:
            episode_step_list.append(episode_step)
            episode_step = 0
            
            if not has_next_obs:
                c2 += 1
                continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

        # print("hi"+str(len(reward_arrays[0])))
        # print("hi"+str(len(reward_arrays[1])))
        # 这里处理每个reward数组
        for j in range(len(reward_arrays)):
            reward = reward_arrays[j][i].astype(np.float32)  # 假设dataset['reward_arrays']是一个包含10个reward的列表
            reward_arrays_[j].append(reward)

        

        episode_step += 1

        # print(obs)
        # print(new_obs)
        # print(reward)
        # print(done_bool)
        # sys.exit()


    # print("util")
    # print(f"c1={c1}")
    # print(f"c2={c2}")
    # print(np.array(reward_).shape)
    # print(np.array(obs_).shape)
    # print(np.array(next_obs_).shape)
    # print(np.array(done_).shape)
    # # print(episode_step_list.size)
    # print("*" * 100)
    # sys.exit()
    # print(type(reward_arrays_))
    # print(type(new_reward_arrays))

    new_dataset = {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }
    new_reward_arrays = np.array(reward_arrays_)
    arr_dict = dict(zip(range(len(new_reward_arrays)), new_reward_arrays))

    # print(new_dataset['observations'].shape)
    # print(new_reward_arrays.shape)
    # print(new_dataset['rewards'][-5:])
    # print(new_reward_arrays[0])
    # print(reward_arrays_[0])
    # sys.exit()

    return new_dataset, arr_dict


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, max_ep_len=1000, device="cpu"):
        super().__init__()

        self.obs_dim = dataset["observations"].shape[-1]
        self.action_dim = dataset["actions"].shape[-1]
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.device = torch.device(device)
        self.input_mean = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).mean(0)
        self.input_std = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).std(0) + 1e-6

        data_ = collections.defaultdict(list)
        
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        self.trajs = []
        for i in range(dataset["rewards"].shape[0]):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000-1)
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                self.trajs.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1
        
        indices = []
        for traj_ind, traj in enumerate(self.trajs):
            end = len(traj["rewards"])
            for i in range(end):
                indices.append((traj_ind, i, i+self.max_len))

        self.indices = np.array(indices)
        

        returns = np.array([np.sum(t['rewards']) for t in self.trajs])
        num_samples = np.sum([t['rewards'].shape[0] for t in self.trajs])
        print(f'Number of samples collected: {num_samples}')
        print(f'Num trajectories: {len(self.trajs)}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_ind, start_ind, end_ind = self.indices[idx]
        traj = self.trajs[traj_ind].copy()
        obss = traj['observations'][start_ind:end_ind]
        actions = traj['actions'][start_ind:end_ind]
        next_obss = traj['next_observations'][start_ind:end_ind]
        rewards = traj['rewards'][start_ind:end_ind].reshape(-1, 1)
        delta_obss = next_obss - obss
    
        # padding
        tlen = obss.shape[0]
        inputs = np.concatenate([obss, actions], axis=1)
        inputs = (inputs - self.input_mean) / self.input_std
        inputs = np.concatenate([inputs, np.zeros((self.max_len - tlen, self.obs_dim+self.action_dim))], axis=0)
        targets = np.concatenate([delta_obss, rewards], axis=1)
        targets = np.concatenate([targets, np.zeros((self.max_len - tlen, self.obs_dim+1))], axis=0)
        masks = np.concatenate([np.ones(tlen), np.zeros(self.max_len - tlen)], axis=0)

        inputs = torch.from_numpy(inputs).to(dtype=torch.float32, device=self.device)
        targets = torch.from_numpy(targets).to(dtype=torch.float32, device=self.device)
        masks = torch.from_numpy(masks).to(dtype=torch.float32, device=self.device)

        return inputs, targets, masks