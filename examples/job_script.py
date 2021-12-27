"""
Job script to optimize trajectories with trajopt
See: https://github.com/aravindr93/trajopt.git
"""

from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
from tqdm import tqdm
import time as timer
import numpy as np
import pickle
import gym
import mjrl.envs
import trajopt.envs
import mj_envs
import json
import os
import hydra


@hydra.main(config_path="configs", config_name="continual_kitchen_config")
def main(job_data):
    print(mj_envs.__path__)  # Necessary for mj_envs to work on SLURM. No clue why.
    # Unpack args and make files for easy access
    ENV_NAME = job_data['env_name']
    PICKLE_FILE = 'trajectories.pickle'
    if 'visualize' in job_data.keys():
        VIZ = job_data.visualize
    else:
        VIZ =False

    # helper function for visualization
    def trigger_tqdm(inp, viz=False):
        if viz:
            return tqdm(inp)
        else:
            return inp

    # =======================================
    # Train loop
    e = get_environment(ENV_NAME)
    mean = np.zeros(e.action_dim)
    sigma = 1.0*np.ones(e.action_dim)
    filter_coefs = [sigma, job_data.filter.beta_0, job_data.filter.beta_1, job_data.filter.beta_2]
    trajectories = []

    ts=timer.time()
    for i in range(job_data.num_traj):
        start_time = timer.time()
        print("Currently optimizing trajectory : %i" % i)
        seed = job_data.seed + i*12345
        e.reset(seed=seed)
        
        agent = MPPI(e,
                    H=job_data.plan_horizon,
                    paths_per_cpu=job_data.paths_per_cpu,
                    num_cpu=job_data.num_cpu,
                    kappa=job_data.kappa,
                    gamma=job_data.gamma,
                    mean=mean,
                    filter_coefs=filter_coefs,
                    default_act=job_data.default_act,
                    seed=seed)
        
        for t in trigger_tqdm(range(job_data.H_total), VIZ):
            step_time = timer.time()
            agent.train_step(job_data.num_iter)
            if not VIZ:
                print(f"Step in {timer.time() - step_time}.")
            if agent.done:
                print(f"Done early at {t}.")
                break
        
        end_time = timer.time()
        print("Trajectory reward = %f" % np.sum(agent.sol_reward))
        print("Optimization time for this trajectory = %f" % (end_time - start_time))
        trajectories.append(agent)
        pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))
        
    print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))
    pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))

    if VIZ:
        vid_dir = "vids"
        os.mkdir(vid_dir)
        for i, traj in enumerate(trajectories):
            solved = traj.render_result(os.path.join(vid_dir, f"traj{i}.mp4"))
            print(f"Traj {i} is_solved: {solved}")


if __name__ == "__main__":
    main()