import argparse
import wandb

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-f', '--ff', type=str, default='delete this arg later.', help="Ignore this arg.")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument('--load_height', type=int, default=1024, help="Height of input image.")
    parser.add_argument('--load_width', type=int, default=768, help="Width of input image.")
    parser.add_argument('--shuffle', action='store_true', help="Shuffle the dataset.")
    parser.add_argument('--num_gpus', type=int, default=1, help="Use Distributed training with multi GPUs.")
    parser.add_argument('--sync_bn', action='store_true', help="Synchronize BatchNorm across all devices.")

    parser.add_argument('--use_wandb', action='store_true', help="Use wandb logger.")
    parser.add_argument('--log_interval', type=int, default=100, metavar="N", help="Log per N steps.")
    parser.add_argument('--seed', type=int, default=3407, metavar="N", help="Random seed.")
    parser.add_argument('--project_name', type=str, default='VITON-HD', help="Name of wandb project.")
    parser.add_argument('--use_amp', action='store_true', help="Use mixed precision training.")
    parser.add_argument('--memory_format', type=str, default='channels_last', help="Channels last or contiguous.")

    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--dataset_mode', type=str, default='train', help="train or test.")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')

    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth')

    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--no_lsgan', action='store_true', help="Not use Least Squares GAN loss.")

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5, help="For GMM.")

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')

    args = parser.parse_args()
    if args.use_wandb:
        if args.local_rank == 0:
            run = wandb.init(project=args.project_name)
            wandb.config.update(args)
            args = wandb.config
    return args