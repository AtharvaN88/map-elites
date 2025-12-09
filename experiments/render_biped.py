import pickle
import numpy as np
import imageio
import os

from domains.bipedal_walker import BipedalDomain

DATA_PATH = "data/biped"
VIDEO_PATH = "videos"

def get_best_controller(archive):
    best_x = None
    best_f = -1
    for cell in archive.values():
        if cell.f > best_f:
            best_f = cell.f
            best_x = cell.x
    return best_x

def main():
    os.makedirs(VIDEO_PATH, exist_ok=True)

    # Load archive
    archive = pickle.load(open(f"{DATA_PATH}/archive.pkl", "rb"))

    # Pick the best solution found
    x_best = get_best_controller(archive)

    # Use a midpoint task as the test scenario
    theta_test = np.array([0.5, 0.5])

    # Create env with rgb_array mode
    env = BipedalDomain(render_mode="rgb_array")
    env.reset(theta_test)

    frames = []

    for _ in range(600):
        _, _, done, _ = env.step_cpg(x_best)
        frame = env.env.render()  # get RGB frame
        frames.append(frame)
        if done:
            break

    env.close()

    # Save output video (MP4)
    out_file = f"{VIDEO_PATH}/biped_demo.mp4"
    imageio.mimsave(out_file, frames, fps=30)
    print(f"Saved video to {out_file}")

if __name__ == "__main__":
    main()
