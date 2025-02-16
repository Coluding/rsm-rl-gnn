from algorithm import *
from environment import *
from models import *
import argparse
import copy

def dqn(type_embedding_dim: int = 12, hidden_dim: int = 64, action_layer: int = 1, num_locations: int = 8,
          num_heads: int = 2, lr: float = 3e-5, gamma: float = 0.99, batch_size: int = 32, buffer_size: int = 10000,
          target_update: int = 10, priority: bool = False, epsilon: float = 1.0, epsilon_decay: float = 0.995,
          epsilon_min: float = 0.1, stack_states: int = 4, reward_scaling: bool = False, eval_every: int = 10,
            num_episodes: int = 1000, action_space="separate", use_client_embeddings=False, use_timestep_context=None,
            update_epochs=20, train_every_steps=50, load_path=None):
    config = FluidityEnvironmentConfig(
        jar_path="../ressources/jars/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar",
        jvm_options=['-Djava.security.properties=/home/lukas/flusim/simurun/server0/xmr/config/java.security'],
        configuration_directory_simulator="../ressources/run_configs/400_steps/",
        node_identifier="server0",
        device="cuda",
        feature_dim_node=1
    )

    env = FluidityEnvironment(config)
    env = TorchGraphObservationWrapper(env, one_hot=False)
    env = StackedBatchObservationWrapper(env, stack_size=stack_states)

    if action_space == "cross_product":
        policy_model = StandardCrossProductModel(input_dim=5, embedding_dim=type_embedding_dim, hidden_dim=hidden_dim,
                                                 action_layer=action_layer, num_locations=num_locations,
                                                 num_heads=num_heads,
                                                 max_timesteps=use_timestep_context,
                                                 use_client_embeddings=use_client_embeddings)

    else:
        policy_model = StandardModel(input_dim=5, embedding_dim=type_embedding_dim, hidden_dim=hidden_dim,
                                     action_layer=action_layer, num_locations=num_locations, num_heads=num_heads)

    target_model = copy.deepcopy(policy_model)

    if action_space == "cross_product":
        ca = CrossProductActionSpace.from_json("data/action_space.json")

    agent_config = DQNAgentConfig(policy_net=policy_model, target_net=target_model, env=env, lr=lr, gamma=gamma,
                                    batch_size=batch_size, buffer_size=buffer_size, target_update=target_update,
                                    priority=priority, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                                    reward_scaling=reward_scaling, eval_every_episode=eval_every, update_epochs=update_epochs,
                                    cross_product_action_space=ca, train_every_steps=train_every_steps, temporal_size=stack_states)

    agent = DQNAgent(agent_config)


    if load_path is not None:
        agent.load_model(load_path)

    agent.train(num_episodes)

def on_policy(type_embedding_dim: int = 12, hidden_dim: int = 64, action_layer: int = 1, num_locations: int = 8,
       num_heads: int = 2, lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 32, buffer_size: int = 10000,
       clip_epsilon: float = 0.2, entropy_coeff: float = 0.01, value_coeff: float = 0.5, update_epochs: int = 10,
        reward_scaling: bool = False, train_every: int = 50, stack_states: int = 4, num_episodes: int = 1000,
        lr_value_fn: float = 3e-3, action_space="separate", use_timestep_context=None, load_path=None,
              use_client_embeddings=False, on_policy_algorithm="ppo"):

    config = FluidityEnvironmentConfig(
        jar_path="../ressources/jars/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar",
        jvm_options=['-Djava.security.properties=/home/lukas/flusim/simurun/server0/xmr/config/java.security'],
        configuration_directory_simulator="../ressources/run_configs/easy/",
        node_identifier="server0",
        device="cuda",
        feature_dim_node=1
    )

    env = FluidityEnvironment(config)

    env = TorchGraphObservationWrapper(env, one_hot=False)
    env = StackedBatchObservationWrapper(env, stack_size=stack_states)

    if action_space == "cross_product":
        policy_model = StandardCrossProductModel(input_dim=5, embedding_dim=type_embedding_dim, hidden_dim=hidden_dim,
                            action_layer=action_layer, num_locations=num_locations, num_heads=num_heads,
                                                 max_timesteps=use_timestep_context,
                                                 use_client_embeddings=use_client_embeddings)

    else:
        policy_model = StandardModel(input_dim=5, embedding_dim=type_embedding_dim, hidden_dim=hidden_dim,
                                action_layer=action_layer, num_locations=num_locations, num_heads=num_heads)

    value_model = StandardValueModel(input_dim=5, embedding_dim=type_embedding_dim, hidden_dim=hidden_dim,
                                     num_locations=num_locations, num_heads=num_heads, max_timestep=use_timestep_context,
                                     use_client_embeddings=use_client_embeddings
                                     ) if on_policy_algorithm == "PPO" else None

    if action_space == "cross_product":
        ca = CrossProductActionSpace.from_json("data/action_space.json")

    else:
        ca = None

    agent_config = OnPolicyAgentConfig(algorithm=on_policy_algorithm, lr_value_fn=lr_value_fn,
                                       policy_net=policy_model, value_net=value_model, env=env, lr=lr, gamma=gamma,
                                       batch_size=batch_size, buffer_size=buffer_size, clip_epsilon=clip_epsilon,
                                       entropy_coeff=entropy_coeff, value_coeff=value_coeff, update_epochs=update_epochs,
                                       reward_scaling=reward_scaling, train_every=train_every,
                                       temporal_size=stack_states, cross_product_action_space=ca, use_gae=False,
                                       use_timestep_context=use_timestep_context is not None)

    agent = OnPolicyAgent(agent_config)

    if load_path is not None:
        agent.load_model(load_path)
        print("Model loaded")

    agent.train(num_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="dqn", choices=["dqn", "ppo", "reinforce"])
    parser.add_argument("--action_space", type=str, choices=["cross_product", "separate"], default="separate")
    parser.add_argument("--num_episodes", type=int, default=10_000)
    parser.add_argument("--type_embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--action_layer", type=int, default=2)
    parser.add_argument("--num_locations", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lr_value_fn", type=float, default=4e-3)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--target_update", type=int, default=810)
    parser.add_argument("--priority", type=bool, default=False)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.999)
    parser.add_argument("--epsilon_min", type=float, default=0.1)
    parser.add_argument("--clip_epsilon", type=float, default=0.05)
    parser.add_argument("--entropy_coeff", type=float, default=0.08)
    parser.add_argument("--value_coeff", type=float, default=1.5)
    parser.add_argument("--update_epochs", type=int, default=12)
    parser.add_argument("--reward_scaling", type=bool, default=False)
    parser.add_argument("--train_every", type=int, default=120)
    parser.add_argument("--stack_states", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--max_timestep", type=bool, default=40)
    parser.add_argument("--use_client_embeddings", type=bool, default=True)
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()

    if args.algorithm == "dqn":
        dqn(type_embedding_dim=args.type_embedding_dim, hidden_dim=args.hidden_dim, action_layer=args.action_layer,
            num_locations=args.num_locations, num_heads=args.num_heads, lr=args.lr,
            gamma=args.gamma, batch_size=args.batch_size, buffer_size=args.buffer_size, update_epochs=args.update_epochs,
            target_update=args.target_update,
            priority=args.priority, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min,
            reward_scaling=args.reward_scaling, eval_every=args.eval_every, stack_states=args.stack_states,
            num_episodes=args.num_episodes, action_space=args.action_space, use_client_embeddings=args.use_client_embeddings,
            use_timestep_context=args.max_timestep, train_every_steps=args.train_every, load_path=args.load_path)

    elif args.algorithm == "ppo":
        on_policy(type_embedding_dim=args.type_embedding_dim, hidden_dim=args.hidden_dim, action_layer=args.action_layer,
            num_locations=args.num_locations, num_heads=args.num_heads, lr=args.lr, lr_value_fn=args.lr_value_fn,
            gamma=args.gamma, batch_size=args.batch_size, clip_epsilon=args.clip_epsilon, entropy_coeff=args.entropy_coeff,
            value_coeff=args.value_coeff, update_epochs=args.update_epochs,
            reward_scaling=args.reward_scaling, train_every=args.train_every, num_episodes=args.num_episodes,
            stack_states=args.stack_states, action_space=args.action_space, use_timestep_context=args.max_timestep,
            use_client_embeddings=args.use_client_embeddings, on_policy_algorithm="PPO"
            )
    elif args.algorithm == "reinforce":
        on_policy(type_embedding_dim=args.type_embedding_dim, hidden_dim=args.hidden_dim, action_layer=args.action_layer,
            num_locations=args.num_locations, num_heads=args.num_heads, lr=args.lr,
            gamma=args.gamma, batch_size=args.batch_size, clip_epsilon=args.clip_epsilon, entropy_coeff=args.entropy_coeff,
            value_coeff=args.value_coeff, update_epochs=args.update_epochs,
            reward_scaling=args.reward_scaling, train_every=args.train_every, num_episodes=args.num_episodes,
            stack_states=args.stack_states, action_space=args.action_space, use_timestep_context=args.max_timestep,
            use_client_embeddings=args.use_client_embeddings, on_policy_algorithm="REINFORCE"
            )




