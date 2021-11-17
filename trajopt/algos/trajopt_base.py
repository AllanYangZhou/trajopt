"""
Base trajectory class
"""

import numpy as np
import imageio

class Trajectory:
    def __init__(self, env, H=32, seed=123):
        self.env, self.seed = env, seed
        self.n, self.m, self.H = env.observation_dim, env.action_dim, H

        # following need to be populated by the trajectory optimization algorithm
        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []

        self.env.reset_model(seed=self.seed)
        self.sol_state.append(self.env.get_env_state())
        self.sol_obs.append(self.env._get_obs())
        self.act_sequence = np.zeros((self.H, self.m))

    def update(self, paths):
        """
        This function should accept a set of trajectories
        and must update the solution trajectory
        """
        raise NotImplementedError

    def animate_rollout(self, t, act):
        """
        This function starts from time t in the solution trajectory
        and animates a given action sequence
        """
        self.env.set_env_state(self.sol_state[t])
        for k in range(act.shape[0]):
            try:
                self.env.env.env.mujoco_render_frames = True
            except AttributeError:
                self.env.render()
            self.env.set_env_state(self.sol_state[t+k])
            self.env.step(act[k])
            print(self.env.env_timestep)
            print(self.env.real_step)
        try:
            self.env.env.env.mujoco_render_frames = False
        except:
            pass

    def animate_result(self):
        self.env.reset(self.seed)
        self.env.set_env_state(self.sol_state[0])
        for k in range(len(self.sol_act)):
            self.env.env.env.mujoco_render_frames = True
            self.env.render()
            self.env.step(self.sol_act[k])
        self.env.env.env.mujoco_render_frames = False

    def render_result(self, fname):
        self.env.reset(self.seed)
        self.env.set_env_state(self.sol_state[0])
        vid_writer = imageio.get_writer(fname, mode="I", fps=20)
        solved = False
        for k in range(len(self.sol_act)):
            curr_frame = self.env.env.sim.render(width=640, height=480, mode='offscreen', device_id=0)
            vid_writer.append_data(curr_frame[::-1,:,:])
            _, _, _, info = self.env.step(np.clip(self.sol_act[k], -0.999, .999))
            solved = solved or info["solved"]
        vid_writer.close()
        return solved