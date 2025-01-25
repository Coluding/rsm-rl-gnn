import gym
from gym import Env
from gym import spaces
from gym.core import ActType, ObsType, RenderFrame
from dataclasses import dataclass
from torch_geometric.utils.convert import from_networkx
from collections import deque
import numpy as np
import networkx as nx
from typing import Tuple, Optional, Union, List
from java_conn import JavaSimulator, FluidityStepResult
import torch_geometric
from typing import Dict
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from env_logging import initialize_logger


@dataclass
class FluidityEnvironmentConfig:
    jar_path: str
    jvm_options: list[str]
    configuration_directory_simulator: str
    node_identifier: str
    device: str
    feature_dim_node: int = 1

logger = initialize_logger(log_file="fluidity.environment.log")



class FluidityEnvironmentRenderer:
    #TODO: Not working yet
    def __init__(self, env):
        self.env = env  # Reference to the environment
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.pos = None  # Will be initialized with a stable layout
        self.ani = None  # Animation object

    def update(self, frame):
        """ Update function for animation, re-draws graph based on the current state """
        self.ax.clear()

        # Get the updated graph
        G = self.env.graph

        # Ensure a stable layout over time
        if self.pos is None:
            self.pos = nx.spring_layout(G, seed=42)

        # Node colors based on their labels
        color_map = {0: 'blue', 1: 'green', 2: 'red'}
        node_colors = [color_map[G.nodes[n]['label']] for n in G.nodes]


        edge_weights = [G[u][v]['weight'] / 50 for u, v in G.edges]  # Scale for visualization

        # Draw graph elements
        nx.draw(G, self.pos, with_labels=True, node_color=node_colors, edge_color="gray",
                node_size=700, font_size=8, width=edge_weights)

        self.ax.set_title("Real-Time Fluidity Environment Network")

    def animate(self):
        """ Run real-time animation """
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1000)  # Update every second
        plt.show()


class FilteredActionSamplingWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.valid_nodes = self._get_valid_nodes()
        self.action_space = gym.spaces.MultiDiscrete([len(self.valid_nodes), len(self.valid_nodes)])

    def _get_valid_nodes(self):
        """ Returns indices of nodes that are neither active nor passive. """
        return [i for i, loc in self.env.loc_mapping.items()
                if loc not in self.env.active_locations and loc not in self.env.passive_locations]

    def action(self, action):
        """ Maps sampled action indices to the actual node indices in the original environment. """
        mapped_action = [self.valid_nodes[action[0]], self.valid_nodes[action[1]]]
        return mapped_action

    def sample_valid_action(self):
        """ Samples a valid action based on filtered nodes. """
        return np.random.choice(self.valid_nodes, size=2)

    def reset(self, **kwargs):
        """ Update valid nodes after environment reset. """
        obs = self.env.reset(**kwargs)
        self.valid_nodes = self._get_valid_nodes()  # Recompute after reset
        return obs


class FluidityEnvironment(Env):
    def __init__(self, config: FluidityEnvironmentConfig):
        self.config = config
        self.simulator = JavaSimulator(
            jar_path=self.config.jar_path,
            jvm_options=self.config.jvm_options,
            configuration_directory_simulator=self.config.configuration_directory_simulator,
            node_identifier=self.config.node_identifier
        )

        self.replica_latencies = None
        self.loc_mapping = None
        self.visited_locations = set()
        self.location_label_mapping = {0: "Client", 1: "Active", 3: "Inactive", 2: "Passive", 4: "Coordinator"}
        self.label_location_mapping = {v: k for k, v in self.location_label_mapping.items()}
        self.active_locations = set()
        self.latency_history = []
        self.client_quantity = {}

        self._initialize_locations()

        self.action_space = spaces.MultiDiscrete([len(self.loc_mapping),
                                                 len(self.loc_mapping)])
        self.node_space = spaces.Discrete(len(self.location_label_mapping))
        self.edge_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Graph(node_space=self.node_space, edge_space=self.edge_space)
        self.graph = nx.Graph()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        logger.info(f"Taking action {action}")
        raw_observation  = self.simulator.step(action)
        processed_observation = self._process_raw_observation(raw_observation)
        observation = self._build_graph_from_observation(processed_observation)
        reward = self.calculate_reward(processed_observation) + raw_observation.mean_delay
        done = raw_observation.finished

        return observation, reward, done, {}, {}

    def reset(self, **kwargs) -> ObsType:
        raw_observation = self.simulator.step((0, 0))
        self._initialize_replica_latencies(raw_observation)
        processed_observation = self._process_raw_observation(raw_observation)
        observation = self._build_graph_from_observation(processed_observation)
        self.last_observation = observation
        return observation, {}

    def _process_raw_observation(self, raw_observation: FluidityStepResult) -> Dict[str, Dict[str, float]]:
        self.active_locations = raw_observation.active_locations
        self.passive_locations = raw_observation.passive_locations
        self.coordinator = raw_observation.coordinator
        self._update_replica_latencies(raw_observation)
        processed_observation = {}

        for loc in raw_observation.distance_latencies.keys():
            # We need to filter: We want clients and all replicas for active and passive locs
            # We want only active and passive locations for inactive locs
            # Why? The agent should not see the latencies of inactive locations to replicas and clients
            if loc in self.active_locations or loc in self.passive_locations:
                client_keys = list(filter(lambda x: "Client" in x, raw_observation.distance_latencies[loc].keys()))
                self.client_quantity[loc] = len(client_keys)
                replica_keys = list(filter(lambda x: "Client" not in x, raw_observation.distance_latencies[loc].keys()))
                processed_observation[loc] = {**{x: raw_observation.distance_latencies[loc][x] for x in client_keys},
                                                **{x: self.replica_latencies[loc][x] for x in replica_keys}}

            else:
                relevant_keys = list(filter(lambda x: x in self.active_locations or x in self.passive_locations,
                                       raw_observation.distance_latencies[loc].keys()))
                processed_observation[loc] = {x: self.replica_latencies[loc][x] for x in relevant_keys}

        self.visited_locations.update(self.active_locations)
        self.visited_locations.update(self.passive_locations)
        return processed_observation

    def calculate_reward(self, processed_observation: Dict[str, Dict[str, float]]) -> float:
        # We only want clients to be considered for latency aggregation
        latency_per_location = {loc: sum(processed_observation[loc].values()) - sum(self.replica_latencies[loc].values())
                                for loc in self.active_locations}
        total_clients =  sum([v for k,v in self.client_quantity.items() if k in self.active_locations])
        mean_latency = sum(latency_per_location.values()) / total_clients
        self.latency_history.append(mean_latency)
        return mean_latency

    def _get_node_label(self, location: str):
        if location == self.coordinator:
            return self.label_location_mapping["Coordinator"]
        if "Client" in location:
            return self.label_location_mapping["Client"]
        elif location in self.active_locations:
            return self.label_location_mapping["Active"]
        elif location in self.passive_locations:
            return self.label_location_mapping["Passive"]
        else:
            return self.label_location_mapping["Inactive"]

    def _build_graph_from_observation(self, processed_observation: Dict[str, Dict[str, float]]):
        G = nx.Graph()
        for location, clients in processed_observation.items():
            region_label = self._get_node_label(location)
            G.add_node(location, label=region_label)  # 1 for active, 2 for passive, 3 for inactive. TODO: We also need passive here
            for request_partner, weight in clients.items():
                if "Client" in request_partner:
                    region_label = 0
                    G.add_node(request_partner, label=region_label)
                else:
                    if not G.has_node(request_partner):
                        region_label = self._get_node_label(request_partner)
                        G.add_node(request_partner, label=region_label)

                G.add_edge(location, request_partner, weight=weight)

        self.graph = G
        return G

    def _initialize_locations(self):
        with open(self.config.configuration_directory_simulator + "server0/xmr/config/locations.config", "r") as f: #TODO: Change this to a more general path
            locs = f.readlines()

        locs = [x.strip("\n") for x in locs]
        locs = list(filter(lambda x: len(x.split(" ")) > 3, locs))
        loc_mapping = {int(x.split(" ")[-1]): x.split(" ")[0] for x in locs}

        self.loc_mapping = loc_mapping

    def _initialize_replica_latencies(self, raw_observation: FluidityStepResult):
        # This will be given by the JavaSimulator in the future
        # For now, we do not have initial latencies of replicas, so we initialize them to 100.
        self.replica_latencies = {loc: {loc2: raw_observation.distance_latencies[loc][loc2]
                                        for loc2 in self.loc_mapping.values()}
                                  for loc in self.loc_mapping.values()}

    def _update_replica_latencies(self, raw_observation: FluidityStepResult):
        # Only replicas are relevant. We get all the client data from the raw_observation at each step
        # We only update the latencies for active and passive locations. The other ones are historic and do not change
        # until they become active or passive again.
        counter = 0
        for loc in self.loc_mapping.values():
            replica_keys = filter(lambda x: "Client" not in x, raw_observation.distance_latencies[loc].keys())
            replica_active_and_passive_keys = filter(lambda x: x in self.active_locations or x in self.passive_locations, replica_keys)
            logger.info(
                f"Updated replica latencies for active {self.active_locations} and passive location:  {self.passive_locations} ") if counter == 0 else None
            for replica in replica_active_and_passive_keys:
                self.replica_latencies[loc][replica] = raw_observation.distance_latencies[loc][replica]
            counter += 1


class TorchGraphObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, one_hot: bool = False):
        super().__init__(env)
        self.one_hot = one_hot
        self.device = env.config.device

    def observation(self, observation: nx.Graph) -> torch_geometric.data.Data:
        torch_graph = from_networkx(observation)

        if self.one_hot:
            torch_graph.label = torch.nn.functional.one_hot(torch_graph.label)

        torch_graph.label = torch_graph.label.to(torch.long).to(self.device)
        torch_graph.edge_index = torch_graph.edge_index.to(self.device)

        add_mask = torch.tensor([0 if x in self.env.active_locations else -float("inf")
                                 for x in observation.nodes], dtype=torch.float32)


        torch_graph.add_mask = add_mask.to(self.device)

        return torch_graph

class StackedObservationWrapper(TorchGraphObservationWrapper):
    def __init__(self, env, stack_size: int = 3):
        super().__init__(env)
        self.stack_size = stack_size
        self.observation_buffer = deque(maxlen=stack_size)

    def reset(self, **kwargs) -> Union[List[torch_geometric.data.Data], Dict]:
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.stack_size): # evtl hier den step ausführen, damit wir nicht immer die gleiche observation haben
            self.observation_buffer.append(observation)
        return list(self.observation_buffer), info

    def observation(self, observation: torch_geometric.data.Data) -> List[torch_geometric.data.Data]:
        self.observation_buffer.append(observation)
        return list(self.observation_buffer)



if __name__ == "__main__":
    config = FluidityEnvironmentConfig(
        jar_path="/home/lukas/Projects/emusphere/simulator-xmr/target/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar",
        jvm_options=['-Djava.security.properties=/home/lukas/flusim/simurun/server0/xmr/config/java.security'],
        configuration_directory_simulator="/home/lukas/flusim/simurun/",
        node_identifier="server0",
        device="cuda",
        feature_dim_node=1
    )

    env = FluidityEnvironment(config)
    env = TorchGraphObservationWrapper(env, one_hot=False)
    env = StackedObservationWrapper(env, stack_size=3)
    obs = env.reset()
    obs, reward, done, _, _ = env.step((0, 0))
    logger.info(f"Reward: {reward}")
    obs, reward, done, _, _ = env.step((6, 0))
    logger.info(f"Reward: {reward}")
    obs, reward, done, _, _ = env.step((0, 6))
    logger.info(f"Reward: {reward}")
    obs, reward, done, _, _ = env.step((7, 1))
    logger.info(f"Reward: {reward}")
    obs, reward, done, _, _ = env.step((5, 2))
    logger.info(f"Reward: {reward}")
    for i in range(40):
        obs, reward, done, _, _ = env.step((0, 0))
        logger.info(f"Step {i}, Reward: {reward}")
