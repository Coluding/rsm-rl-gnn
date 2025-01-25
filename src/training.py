from algorithm import *
from environment import *
from models import *
import argparse

def train(type_embedding_dim: int = 12, hidden_dim: int = 64, action_layer: int = 1, num_locations: int = 8,
          num_heads: int = 2, lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 32, buffer_size: int = 10000,
          target_update: int = 10, priority: bool = False, epsilon: float = 1.0, epsilon_decay: float = 0.995,
          epsilon_min: float = 0.1):
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

    policy_model = StandardModel(input_dim=5, embedding_dim=type_embedding_dim, hidden_dim=hidden_dim,
                          action_layer=action_layer, num_locations=num_locations, num_heads=num_heads)

    target_model = StandardModel(input_dim=5, embedding_dim=type_embedding_dim, hidden_dim=hidden_dim,
                            action_layer=action_layer, num_locations=num_locations, num_heads=num_heads)

    agent_config = DQNAgentConfig(policy_net=policy_model, target_net=target_model, env=env, lr=lr, gamma=gamma,
                                    batch_size=batch_size, buffer_size=buffer_size, target_update=target_update,
                                    priority=priority, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)

    agent = DQNAgent(agent_config)
    agent.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_embedding_dim", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--action_layer", type=int, default=1)
    parser.add_argument("--num_locations", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--priority", type=bool, default=False)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--epsilon_min", type=float, default=0.1)
    args = parser.parse_args()
    train(args.type_embedding_dim, args.hidden_dim, args.action_layer, args.num_locations, args.num_heads, args.lr,
          args.gamma, args.batch_size, args.buffer_size, args.target_update, args.priority, args.epsilon,
          args.epsilon_decay, args.epsilon_min)




