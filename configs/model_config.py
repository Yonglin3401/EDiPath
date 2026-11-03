def add_model_config(parser):
    parser.add_argument('--dropout', help='drop out of gnn model', type=float, default=0.5)
    parser.add_argument('--hidden_dim', help='hidden units', type=int, default=128)
    parser.add_argument('--walk_time', help='walks per node', type=int, default=10)
    parser.add_argument('--walk_len', help='length per walk', type=int, default=3)
    parser.add_argument('--walk_cof', help='coefficient', type=float, default=0.75)
    parser.add_argument('--walk_delta', help='walk delta', type=float, default=1)
    parser.add_argument('--walk_q', help='walk delta', type=float, default=1)
    parser.add_argument('--hidden_att_dim', help='hidden_att_dim', type=int, default=64)
