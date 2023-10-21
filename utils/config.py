import argparse
from argparse import ArgumentParser
import json


def write_config_file(args, output_file):
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent=4)


def ParseArguments(argv=None):
    parser = ArgumentParser()
    parser = argparse.ArgumentParser(description="Reinforcement Learning for Dynamic VRP with Stochastic Requests")

    parser.add_argument("--config-file", "-f", type=str, default=None,
                        help="configuration file")
    parser.add_argument("--verbose", "-v", action='store_true', default=True,
                        help="Showing information while processing")
    parser.add_argument("--gpu", action='store_true', default=True,
                        help="Use GPU to run the model")
    parser.add_argument("--seed", type=int, default=None, help="seed to regenerate same result")

    ### Data related arguments

    parser.add_argument_group("Data Generation for DVRPSR")
    parser.add_argument("--problem", "-p", type=str, default='DVRPSR',
                        help="problem to solve is DVRPSR")
    parser.add_argument("--vehicle-count", "-m", type=int, default=2,
                        help='number of vehicles for DVRPSR')
    parser.add_argument("--vehicle-speed", type=int, default=2,
                        help='speed of vehicle for DVRPSR')
    parser.add_argument("--Lambda", type=float, default=0.4,
                        help='Requests rate per minute')
    parser.add_argument("--dod", type=float, default=0.85,
                        help="Degree of dynamism")
    parser.add_argument("--horizon", type=int, default=600,
                        help='Working time for DVRPSR in minutes')
    parser.add_argument("--customers_count", type=int, default=None,
                        help='Working time for DVRPSR in minutes')
    parser.add_argument("--fDmean", type=int, default=10,
                        help="mean value for service duration of customers")
    parser.add_argument("--fDstd", type=float, default=2.5,
                        help="standard deviation for service duration of customers")
    parser.add_argument("--euclidean", action = 'store_true', default=True,
                        help="Wheather to use Euclidean distance or City street network for distance calculation")

    ### Environment related arguments
    parser.add_argument_group(" Environment for DVRPSR")
    parser.add_argument("--pending-cost", type=int, default=0.1,
                        help='Pending cost for not serving a static customers in routes')
    parser.add_argument("--dynamic-reward", type=int, default=0.05,
                        help="Reward for serving a Dynamic customer")

    parser.add_argument_group(" Graph Attention models ")
    parser.add_argument("--model-size", type=int, default=32,
                        help=" Size of for attention models")
    parser.add_argument("--encoder-layer", type=int, default=2,
                        help='Number of Encoder Layers')
    parser.add_argument("--num-head", type=int, default=4,
                        help='Number of heads in MultiHeadAttention modules')
    parser.add_argument("--ff-size-actor", type=int, default=32,
                        help=" Size of fully connected Feed Forward Networks")
    parser.add_argument("--ff-size-critic", type=int, default=32,
                        help=" Size of fully connected Feed Forward Networks")
    parser.add_argument("--tanh-xplor", type=int, default=10)
    parser.add_argument("--edge_embedding_dim", type=int, default=64,
                        help = 'Edge embedding dimention for edge attributes')

    # PPO Agent Training related arguments
    parser.add_argument_group(" Training PPO Agnet ")
    parser.add_argument("--greedy", action='store_true', default=False,
                        help='weather to use greedy or smapling')
    parser.add_argument("--learning-rate", type=int, default=5e-5,
                        help='Learning rate for PPO agent')
    parser.add_argument("--ppo-epoch", type=int, default=2,
                        help='Epoch for PPO to run the sample and evaluate')
    parser.add_argument("--entropy-value", type=int, default=0.01)
    parser.add_argument("--epsilon-clip", type=int, default=0.2)
    parser.add_argument("--timestep", type=int, default=1)

    parser.add_argument("--epoch-count", "-e", type=int, default=15)
    parser.add_argument("--iter-count", "-i", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--rate-decay", '-d', type=float, default=0.96)
    parser.add_argument("--max-grad-norm", type=float, default=2)
    parser.add_argument("--grad-norm-decay", type=float, default=None)

    ### Testing Related arguments
    parser.add_argument("--test-batch-size", type=int, default=32)

    ### Saving paramters
    parser.add_argument_group("Checkpointing")
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    parser.add_argument("--checkpoint-period", "-c", type=int, default=1)
    parser.add_argument("--resume-state", type=str, default=None)

    args = parser.parse_args(argv)
    if args.config_file is not None:
        with open(args.config_file) as f:
            parser.set_defaults(**json.load(f))

    return parser.parse_args(argv)
