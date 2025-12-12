import argparse

# this is for manually loadign the args
def load_args(overrides=None):
    args = parser.parse_args([])  # load defaults
    if overrides:
        for k, v in overrides.items():
            setattr(args, k, v)
    return args

parser = argparse.ArgumentParser(prog = 'Printed Spiking Neural Networks',
                                 description = 'Training for Printed Spiking Neural Networks')

# printing-related hyperparameters for pNNs
parser.add_argument('--gmin',                  type=float,     default=0.01,                  help='minimal printable conductance value')
parser.add_argument('--gmax',                  type=float,     default=10.,                   help='maximal printable conductance value')
parser.add_argument('--T',                     type=float,     default=0.1,                   help='measuring threshold')
parser.add_argument('--m',                     type=float,     default=0.3,                   help='measuring margin')
# learnable activation circuits 
parser.add_argument('--ACT_R1n',               type=float,     default=12.7865,               help='resistance in nonlinear circuit')
parser.add_argument('--ACT_R2n',               type=float,     default=-3.1871,               help='resistance in nonlinear circuit')
parser.add_argument('--ACT_W1n',               type=float,     default=-10.4537,              help='width of the transistor 1')
parser.add_argument('--ACT_L1n',               type=float,     default=10.5460,               help='length of the transistor 1')
parser.add_argument('--ACT_W2n',               type=float,     default=-5.8496,               help='width of the transistor 2')
parser.add_argument('--ACT_L2n',               type=float,     default=4.2337,                help='length of the transistor 2')
# learnable negative weight circuits 
parser.add_argument('--NEG_R1n',               type=float,     default=2.5041,                help='resistance in nonlinear circuit')
parser.add_argument('--NEG_k1',                type=float,     default=-1.1751,               help='ratio of R1/R2')
parser.add_argument('--NEG_R3n',               type=float,     default=-6.4452,               help='resistance in nonlinear circuit')
parser.add_argument('--NEG_k2',                type=float,     default=1.5129,                help='ratio of R3/R4')
parser.add_argument('--NEG_R5n',               type=float,     default=-3.4880,               help='resistance in nonlinear circuit')
parser.add_argument('--NEG_Wn',                type=float,     default=-1.8311,               help='width of the transistor')
parser.add_argument('--NEG_Ln',                type=float,     default=4.0129,                help='length of the transistor')

# machine-learning-related hyperparameters
# dataset-related
parser.add_argument('--task',                  type=str,       default='temporized',          help='train normal pNN or split manufacturing, or temporal information')
parser.add_argument('--DATASET',               type=int,       default=0,                     help='index of training dataset')
parser.add_argument('--DataPath',              type=str,       default='./dataset',           help='path to dataset')
parser.add_argument('--DS_VAR',                type=str,       default='none',                help='variation of dataset')
parser.add_argument('--augment',               type=bool,      default=False,                 help='if needed data augmentation')

# data augmentation
parser.add_argument('--InputNoise',            type=float,     default=0.,                    help='noise of input signal')
parser.add_argument('--IN_test',               type=float,     default=0.,                    help='noise of input signal for test')
parser.add_argument('--R_train',               type=int,       default=1,                     help='number of sampling for input noise in training')
parser.add_argument('--R_test',                type=int,       default=1,                     help='number of sampling for input noise in testing')
parser.add_argument('--NOISE_LEVEL',               type=float,
                    default=0.05,                    help='noise of input signal')
parser.add_argument('--WARP_FACTOR',               type=float,
                    default=0.1,                    help='noise of input signal')
parser.add_argument('--SFR_down',               type=float,
                    default=0.9,                    help='noise of input signal')
parser.add_argument('--SFR_up',               type=float,
                    default=1.1,                    help='noise of input signal')
parser.add_argument('--CROP_SIZE',               type=int,
                    default=50,                   help='noise of input signal')

# temporal information processing
parser.add_argument('--N_time',                type=int,       default=100,                   help='number of sampling for temporal information processing')
# regularization
parser.add_argument('--pathnorm',              type=bool,      default=False,                 help='path-norm as regularization for improving robustness against input noise')
# network-related
parser.add_argument('--hidden',                type=list,      default=[3],                   help='topology of the hidden layers')
# training-related
parser.add_argument('--SEED',                  type=int,       default=0,                     help='random seed')
parser.add_argument('--DEVICE',                type=str,       default='gpu',                 help='device for training')
parser.add_argument('--PATIENCE',              type=int,       default=75,                   help='patience for early-stopping')
parser.add_argument('--EPOCH',                 type=int,       default=1000,                help='maximal epochs')
parser.add_argument('--LR',                    type=float,     default=0.1,                   help='learning rate')
parser.add_argument('--PROGRESSIVE',           type=bool,      default=True,                  help='whether the learning rate will be adjusted')
parser.add_argument('--LR_PATIENCE',           type=int,       default=20,                   help='patience for updating learning rate')
parser.add_argument('--LR_DECAY',              type=float,     default=0.5,                   help='decay of learning rate for progressive lr')
parser.add_argument('--LR_MIN',                type=float,     default=5e-3,                  help='minimal learning rate for stop training')
# evaluation
parser.add_argument('--loss',                  type=str,       default='celoss',              help='loss function')
parser.add_argument('--metric',                type=str,       default='temporal_acc',        help='nominal accuracy or measuring-aware accuracy')
parser.add_argument('--SoftEva',               type=bool,      default=False,                 help='if True, evaluate only existing models, otherwise check all models')
# server-related
parser.add_argument('--TIMELIMITATION',        type=float,     default=45,                    help='maximal running time (in hour)')

# hardware-related hyperparameters
# aging-related hyperparameters
parser.add_argument('--MODE',                  type=str,       default='nominal',             help='training mode: aging, nominal')
parser.add_argument('--M_train',               type=int,       default=1,                     help='number of stochastic aging models during training')
parser.add_argument('--K_train',               type=int,       default=1,                     help='number of temporal sampling during training')
parser.add_argument('--M_test',                type=int,       default=1,                     help='number of stochastic aging models for testing')
parser.add_argument('--K_test',                type=int,       default=1,                     help='number of temporal sampling for testing')
parser.add_argument('--t_test_max',            type=int,       default=1,                     help='test time interval')
parser.add_argument('--integration',           type=str,       default='MC',                  help='method for integration: Monte-Carlo, Gaussian Quadrature')
# variation-related hyperparameters
parser.add_argument('--N_train',               type=int,       default=1,                     help='number of sampling for variation during training')
parser.add_argument('--e_train',               type=float,     default=0.,                    help='variation during training')
parser.add_argument('--N_test',                type=int,       default=1,                     help='number of sampling for variation for testing')
parser.add_argument('--e_test',                type=int,       default=0.,                    help='variation for testing')
# power
parser.add_argument('--powerestimator',        type=str,       default='power',               help='the penalty term for encouraging lower energy')
parser.add_argument('--powerbalance',          type=float,     default=0.,                    help='the scaling term for energy vs. accuracy')
parser.add_argument('--estimatorbalance',      type=float,     default=0.,                    help='the scaling term for energy & weight decay')
parser.add_argument('--pgmin',                 type=float,     default=1e-7,                  help='minimal printable conductance gmin')
# circuit learnability
parser.add_argument('--lnc',                   type=bool,      default=False,                  help='shared learnable nonlinear components')
parser.add_argument('--ilnc',                  type=bool,      default=False,                  help='independent learnable nonlinear components')


# log-file-related information
parser.add_argument('--projectname',           type=str,       default='project',             help='name of the project')
parser.add_argument('--temppath',              type=str,       default='/temp',               help='path to temp files')
parser.add_argument('--logfilepath',           type=str,       default='/log',                help='path to log files')
parser.add_argument('--report_freq',           type=int,       default=10,                    help='write log in every N epochs')
parser.add_argument('--recording',             type=bool,      default=False,                 help='save information in each epoch')
parser.add_argument('--recordpath',            type=str,       default='/record',              help='save information in each epoch')
parser.add_argument('--savepath',              type=str,       default='/experiment',         help='save information in each epoch')
parser.add_argument('--loglevel',              type=str,       default='info',                help='level of message logger')

'''# Variation
parser.add_argument('--N_train',               type=int,       default=20,                    help='number of sampling for variation during training')
parser.add_argument('--e_train',               type=float,     default=0.1,                   help='variation during training')
parser.add_argument('--N_test',                type=int,       default=20,                    help='number of sampling for variation for testing')
parser.add_argument('--e_test',                type=float,     default=0.1,                   help='variation for testing')'''