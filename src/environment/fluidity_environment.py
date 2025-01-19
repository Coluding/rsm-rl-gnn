import gym
from gym import Env
from gym import spaces
from gym.core import ActType, ObsType
from dataclasses import dataclass
from torch_geometric.utils.convert import from_networkx
from enum import Enum
import numpy as np
import networkx as nx
from typing import Tuple
from java_conn import JavaSimulator
import torch_geometric
from typing import Dict
import torch
from torch_geometric.data import Data


@dataclass
class FluidityEnvironmentConfig:
    jar_path: str
    jvm_options: list[str]
    configuration_directory_simulator: str
    node_identifier: str
    device: str
    feature_dim_node: int

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

        self._initialize_locations()
        self._initialize_replica_latencies()

        self.action_space = spaces.Discrete(5) #TODO change this
        self.node_space = spaces.Box(low=0, high=np.inf, shape=(config.feature_dim_node,))
        self.edge_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Graph(node_space=self.node_space, edge_space=self.edge_space)
        self.graph = nx.Graph()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        raw_observation = self.simulator.step(action)
        processed_observation = self._process_raw_observation(raw_observation)
        observation = self._build_graph_from_observation(processed_observation)
        reward = self.calculate_reward(observation)
        done = False

        return observation, reward, done, {}

    def reset(self, **kwargs) -> ObsType:
        raw_observation = self.simulator.step(0)
        processed_observation = self._process_raw_observation(raw_observation)
        observation = self._build_graph_from_observation(processed_observation)
        return observation, {}

    def _process_raw_observation(self, raw_observation: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        unknown_locations = set(self.loc_mapping.values()).difference(raw_observation.keys())
        self._update_replica_latencies(raw_observation)

        for loc in unknown_locations:
            # For now I do not insert the latencies of the unknwn location in the latency dict of known location
            # It should be enough to have one latency pair per location, which I am doing in the next line
            # Existing: {us-north-1: {client1: 10, client2: 20,...}, us-north-2: {client1: 30, client2: 40,...}...}
            # New: {new_loc: {us-north-1: 100, us-north-2: 100}}
            # --> There is no need to also do: {us-north-1: {client1: 10, client2: 20,..., new_loc: 100} ...}
            raw_observation[loc] = {x: self.replica_latencies[loc][x] for x in self.loc_mapping.values()}

        self.visited_locations.update(raw_observation.keys())
        return raw_observation

    def calculate_reward(self, observation: nx.Graph) -> float:
        total_latency = sum(data['weight'] for _, _, data in observation.edges(data=True))
        mean_latency = total_latency / observation.number_of_edges()
        return mean_latency

    def _build_graph_from_observation(self, raw_observation: Dict[str, Dict[str, float]]):
        G = nx.Graph()
        for region, clients in raw_observation.items():
            for client, weight in clients.items():
                G.add_edge(region, client, weight=weight)
        return G

    def _initialize_locations(self):
        with open(self.config.configuration_directory_simulator + "server0/xmr/config/locations.config", "r") as f:
            locs = f.readlines()

        locs = [x.strip("\n") for x in locs]
        locs = list(filter(lambda x: len(x.split(" ")) > 3, locs))
        loc_mapping = {int(x.split(" ")[-1]): x.split(" ")[0] for x in locs}

        self.loc_mapping = loc_mapping

    def _initialize_replica_latencies(self):
        # This will be given by the JavaSimulator in the future
        # For now, we do not have initial latencies of replicas, so we initialize them to 100.
        self.replica_latencies = {loc: {loc2: 100 for loc2 in self.loc_mapping.values()} for loc in self.loc_mapping.values()}

    def _update_replica_latencies(self, raw_observation):
        # Only replicas are relevant. We get all the client data from the raw_observation at each step
        for loc in raw_observation.keys():
            replica_keys = filter(lambda x: "Client" not in x, raw_observation[loc].keys())

            for replica in replica_keys:
                self.replica_latencies[loc][replica] = raw_observation[loc][replica]

class TorchGraphObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, one_hot: bool = False):
        super().__init__(env)
        self.one_hot = one_hot
        self.device = env.config.device

    def observation(self, observation: nx.Graph) -> torch_geometric.data.Data:
        torch_graph = from_networkx(observation)

        if self.one_hot:
            torch_graph.x = torch.nn.functional.one_hot(torch_graph.x)

        torch_graph.x = torch_graph.x.to(torch.float32).to(self.device)
        torch_graph.edge_index = torch_graph.edge_index.to(self.device)

        return torch_graph



if __name__ == "__main__":
    config = FluidityEnvironmentConfig(
        jar_path="/home/lukas/Projects/emusphere/simulator-xmr/target/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar",
        jvm_options=['-Djava.security.properties=/home/lukas/flusim/simurun/server0/xmr/config/java.security'],
        configuration_directory_simulator="/home/lukas/flusim/simurun/",
        node_identifier="server0",
        device="cpu",
        feature_dim_node=10
    )

    env = FluidityEnvironment(config)
#    env = TorchGraphObservationWrapper(env, one_hot=True)

    obs = env.reset()
    env.step(1)