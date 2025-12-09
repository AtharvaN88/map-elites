import numpy as np
import gymnasium as gym

class BipedalDomain:
    def __init__(self, render_mode=None):
        self.env = gym.make("BipedalWalker-v3", render_mode=render_mode)
        self.state = None
        self.t = 0.0

    def reset(self, theta):
        """
        theta = [slope, friction]
        Both in [0,1].
        """
        # Map to environment ranges
        slope = (theta[0] - 0.5) * 0.4        # [-0.2, 0.2] radians
        friction = 0.4 + theta[1] * 0.6       # [0.4, 1.0]

        # Configure terrain physics
        # self.env.unwrapped.world.gravity = (0, -9.8)
        self.env.unwrapped.world.ground_friction = friction

        # Reset environment
        state, _ = self.env.reset(options={"slope": slope})
        self.state = state
        self.t = 0.0
        return state

    

    def step_cpg(self, x, dt=0.03):
        """
        x: 13D CPG vector
        """
        # Parse controller params
        A = x[0:4]              # amplitudes
        P = x[4:8] * 2*np.pi    # phases
        B = x[8:12] - 0.5       # biases in [-0.5, 0.5]
        f = 0.5 + x[12] * 2.0   # frequency = [0.5, 2.5]

        self.t += dt
        out = A * np.sin(2*np.pi*f*self.t + P) + B
        out = np.clip(out, -1, 1)

        obs, reward, terminated, truncated, info = self.env.step(out)

        done = terminated or truncated
        
        return obs, reward, done, info



    def close(self):
        self.env.close()
