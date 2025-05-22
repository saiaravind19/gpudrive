import os
from pathlib import Path
import mediapy
import torch

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

from gpudrive.env.config import EnvConfig, SceneConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader

DYNAMICS_MODEL = "classic" # "delta_local" / "state" / "classic"
DATA_PATH = "data/processed/examples" # Your data path
MAX_NUM_OBJECTS = 1
NUM_ENVS = 3

# Configs
env_config = EnvConfig(dynamics_model=DYNAMICS_MODEL)
# Make dataloader
data_loader = SceneDataLoader(
    root="data/processed/examples", # Path to the dataset
    batch_size=NUM_ENVS, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=NUM_ENVS, # Total number of different scenes we want to use
    sample_with_replacement=False, 
    seed=42, 
    shuffle=True,   
)

# Make environment
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=data_loader,
    max_cont_agents=MAX_NUM_OBJECTS, # Maximum number of agents to control per scenario
    device="cpu", 
    action_type="continuous" # "continuous" or "discrete"
)

control_mask = env.cont_agent_mask

obs = env.reset(control_mask)

# Extract full expert trajectory
expert_actions, _, _, _ = env.get_expert_actions()

# Reset environment
obs = env.reset(control_mask)
done_envs = []

frames = {f"env_{i}": [] for i in range(NUM_ENVS)}

# Step through the scene
for t in range(env.episode_len):
    action = expert_actions[:, :, t, :]
    '''
    ToDo: 
    Calculate control frequency using the expert action data and add it dataset
    Add a diffrent dynamic model for adaptic control rather than modifying the classic model
    '''
    # as the expert actions lack control frequncy a default
    if DYNAMICS_MODEL == "classic":
        control_frequency_tensor = torch.full(
           action[..., :1].shape,  # Match the shape of the first action dimension
           0.1,  # Use the first value or another specific value
           device="cpu",)
        action = torch.cat(
            (action, control_frequency_tensor), dim=-1
        )
    env.step_dynamics(action)
    
    dones = env.get_dones()
    
    # Render the scenes
    env_indices = [i for i in range(NUM_ENVS) if i not in done_envs]
    figs = env.vis.plot_simulator_state(
        env_indices=env_indices,
        time_steps=[t]*NUM_ENVS,
        zoom_radius=100,
        #center_agent_indices=[0]*NUM_ENVS,
    )
    for i, env_id in enumerate(env_indices):
        frames[f"env_{env_id}"].append(img_from_fig(figs[i])) 
    
    # Check if done
    for env_id in range(NUM_ENVS):
        if dones[env_id].all():
            done_envs.append(env_id)


current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "speed_up")
os.makedirs(output_dir, exist_ok=True)

# Write video for each world
for i in range(NUM_ENVS):
    video_path = os.path.join(output_dir, f"gym_expert_demo_{i}.mp4")
    mediapy.write_video(video_path, frames[f"env_{i}"], fps=15)
    print(f"Saved video for env_{i} at {video_path}")
    