import argparse
import os
import torch
from model import GAN
from makelabel import makeDir, moveFiles


def parse_args():
    desc = 'GAN Pytorch'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['CACD2000', 'UTKFace'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs for training')
    parser.add_argument('--batch_size', type=str, default=1, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=224, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='D:/GAN/Pyramid-GAN/Dict/', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='D:/GAN/Pyramid-GAN/results', help='Directory name to save the generated images')
    parser.add_argument('--lrG', type=float, default=0.0001)
    parser.add_argument('--lrD', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)

    return check_args(parser.parse_args())


def check_args(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('the size of batch must be larger than or equal to one')

    return args


def main():
    makeDir()
    moveFiles()
    args = parse_args()

    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    gan = GAN(args)

    gan.train()
    print('Training finished')

    gan.visualize_results(args.epoch)
    print('Testing finished')


if __name__ == '__main__':
    main()
