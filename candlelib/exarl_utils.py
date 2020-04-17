import default_utils

def get_default_exarl_parser(parser):
    parser.add_argument("--epsilon_rl", type=float, default= 1.0)
    parser.add_argument("--epsilon_min", type=float, default = 0.05)
    parser.add_argument("--epsilon_decay", type=float, default = 0.995)
    parser.add_argument("--gamma", type=float, default = 0.95)
    parser.add_argument("--tau", type=float, default = 0.5)

    parser.add_argument("--search_method", type=str, default = "epsilon")
    return parser
