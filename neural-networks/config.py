def add_arguments(parser):

    parser.add_argument('--name', default="default_name", type=str)
    parser.add_argument('--num_seeds', default=1, type=int)
    parser.add_argument('--no_cuda', default=1, type=int)

    parser.add_argument('--task', default='original', choices=['ts_regression','ts_classification'], type=str)
    parser.add_argument('--dataset', default='MNIST', type=str, choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'random'])
    parser.add_argument('--loss_type', default='nll', type=str)

    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument('--teacher_width', default=100, type=int)
    parser.add_argument('--teacher_depth', default=2, type=int)
    parser.add_argument('--width', default=20, type=int)
    parser.add_argument('--activation', default='relu', type=str)
    
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--d', default=None, type=int)
    parser.add_argument('--freeze', default=0, type=int)
    parser.add_argument('--pca', default=0, type=int)
    parser.add_argument('--r_phi', default=0.01, type=float)
    parser.add_argument('--r_c', default=1., type=float)
    parser.add_argument('--r_beta', default=1., type=float)
    parser.add_argument('--pca_normalized', default=1, type=int)
    parser.add_argument('--n', default=None, type=int)
    parser.add_argument('--n_test', default=None, type=int)
    parser.add_argument('--noise', default=0., type=float)
    parser.add_argument('--input_noise', default=0., type=float)
    parser.add_argument('--test_noise', default=0., type=int)
    parser.add_argument('--num_classes', default=None, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=0., type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--batch_size_test', default=1000, type=int)

    return parser
