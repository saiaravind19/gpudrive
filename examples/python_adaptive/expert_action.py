import os
from pathlib import Path
import mediapy
import torch

# Set working directory to the base directory 'gpudrive'
os.chdir('/gpudrive')
working_dir = Path.cwd()

from gpudrive.env.config import EnvConfig, SceneConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.observation import LocalEgoState 
from gpudrive.datatypes.roadgraph import LocalRoadGraphPoints,GlobalRoadGraphPoints

DYNAMICS_MODEL = "delta_local" # "delta_local" / "state" / "classic"
DATA_PATH = "data/processed/examples" # Your data path
MAX_NUM_OBJECTS = 3 # Maximum number of objects in the scene
NUM_ENVS = 1

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

print(data_loader.dataset)

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
expert_actions, pxy, velxy, yaw = env.get_expert_actions()
# Reset environment
obs = env.reset(control_mask)
done_envs = []


frames = {f"env_{i}": [] for i in range(NUM_ENVS)}
print("starting simulation")
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
    # need to look how to use it
    
    '''
    means_xy = (env.sim.world_means_tensor().to_torch()[:, :2].to(device='cuda:0'))
    print("means_xy:", means_xy[:,0])
    agent_actions = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor())
    
    agent_actions.restore_mean(mean_x=means_xy[:, 0], mean_y=means_xy[:, 1])
    print("aganet_id:",agent_actions.id)
    print("length:",agent_actions.vehicle_length)
    print("width:",agent_actions.vehicle_width)
    print("height:",agent_actions.vehicle_height)
    print("agent position x:",agent_actions.id)
    print("agent position y:",agent_actions.pos_y.shape)
    print("agent global goal x:",agent_actions.goal_x.shape)
    print("agent global goal y:",agent_actions.goal_y.shape)

    local_obs = LocalEgoState.from_tensor(env.sim.self_observation_tensor())
    print("local goal x:",local_obs.unnormalised_rel_goal_x.shape)
    print("local goal y:",local_obs.unnormalised_rel_goal_y.shape)'''

    local_road_polynomials = LocalRoadGraphPoints.from_tensor(env.sim.agent_roadmap_tensor())
    #print("local_road_polynomials X:", local_road_polynomials.x)
    #print("local_road_polynomials Y:", local_road_polynomials.y)

   # print("local_road_polynomials id:", local_road_polynomials.id.shape)
    #print("road type:",local_road_polynomials.type.shape)

    global_road_polynomials = GlobalRoadGraphPoints.from_tensor(env.sim.map_observation_tensor())
    #global_road_polynomials.restore_mean(mean_x=means_xy[:, 0], mean_y=means_xy[:, 1])
    #print("global_road_polynomials X:", global_road_polynomials.x.shape)
    
    agant_tensor = env.get_unormalized_agent_obs()
    #print(agant_tensor.shape)
    road_tensor =  env.get_unnormalised_road_obs()

    print(road_tensor.shape)
    print(road_tensor)
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
    