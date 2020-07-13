import os
import cv2
import argparse
import torch

import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd

from loss import BalancedL1Loss, ReducedFocalLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from models.res_unet_regressor import ResUNet
from models.hrnet import HRNet
#from dataloader_regressor import SumitomoCADDS
from dataloader import SumitomoCADDS
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(args):

    if args.task == 'train':
        if not args.train_image_path:
            raise 'train data path should be specified !'
        train_dataset = SumitomoCADDS(file_path=args.train_image_path)
        #train_dataset = SumitomoCADDS(file_path=args.val_image_path)

        if args.val_image_path:
            val_dataset = SumitomoCADDS(file_path=args.val_image_path, val=True)

        model = HRNet(3, 32, 8).to(device)
        #model = ResUNet(3, 8).to(device)
        #model = R2AttU_Net(3, 1).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0003)

        if args.resume:
            if not os.path.isfile(args.resume):
                raise '=> no checkpoint found at %s' % args.resume
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s (epoch %d)' % (args.resume, args.start_epoch))

        train(args, model, optimizer, train_dataset, val_dataset)

    else: # test
        if not args.test_image_path: 
            raise '=> test data path should be specified'
        if not args.resume or not os.path.isfile(args.resume):
            raise '=> resume not specified or no checkpoint found'
        test_dataset = SumitomoCADDS(file_path=args.test_image_path, test=True)
        model = ResUNet(3, 8).to(device)
        #model = R2AttU_Net(3, 1).to(device)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Successfully loaded model from {args.resume}')
        test(args, model, test_dataset)

def train(args, model, optimizer, train_dataset, val_dataset):

    #criterion = BalancedL1Loss()
    criterion = ReducedFocalLoss()
    #criterion = nn.SmoothL1Loss()
    #criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    
    # epoch-wise losses
    print(args.batch_size, 'train')
    dataloader = DataLoader(batch_size=args.batch_size, shuffle=True,
                            dataset=train_dataset, num_workers=args.workers)
    writer = SummaryWriter('runs/rfl_hr')
    #dummy_input = torch.rand(16, 3, 256, 256).to(device)
    #writer.add_graph(model, dummy_input)
    best_loss = args.best_loss

    for epoch in range(args.epochs):
        print('training epoch %d/%s' % (args.start_epoch+epoch+1, args.start_epoch+args.epochs))
        batch_train_losses = []
        data_iterator = tqdm(dataloader, total=len(train_dataset) // args.batch_size + 1)
        model.train()
        for i, (images, labels) in enumerate(data_iterator):
            images = images.to(device).float()
            labels = labels.to(device).float()
            outputs = model(images)
            #print(outputs.max(), outputs.min())
            # loss
            train_loss = criterion(outputs, labels)
            #print(outputs.eq(1).sum(), train_loss.item())
            batch_train_losses.append(train_loss.item())
        
            # backward
            model.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # evaluation per epoch
        epo_train_loss = np.mean(batch_train_losses)
        print('epoch train loss: %.4f' % epo_train_loss)
        
        epo_eval_loss = evaluate(args, model, criterion, val_dataset, args.start_epoch+epoch+1, writer)
        print('epoch val loss: %.4f' % epo_eval_loss)
        
        writer.add_scalar('loss/train', epo_train_loss, args.start_epoch+epoch+1)
        writer.add_scalar('loss/val', epo_eval_loss, args.start_epoch+epoch+1)

        # save model
        if epo_eval_loss < best_loss:
            best_loss = epo_eval_loss
            state = {
                'epoch': args.start_epoch+epoch+1,
                'best_loss': best_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, 'best_eval_model.pth.tar')
        
        # update learning rate
        #if (epoch + 1) % 20 == 0:
            #curr_lr /= 3
            #update_lr(optimizer, curr_lr)
    writer.close()

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(args, model, criterion, val_dataset, epo_no, writer):
    
    losses = []
    print(args.batch_size, 'eval')
    dataloader = DataLoader(batch_size=args.batch_size, shuffle=True,
                            dataset=val_dataset, num_workers=args.workers)
    
    model.eval()
    with torch.no_grad():
        print('evaluating...')
        data_iterator = tqdm(dataloader, total=len(val_dataset) // args.batch_size + 1)
        for images, labels, fname in data_iterator:
            images = images.to(device).float()
            labels = labels.to(device).float()
            
            outputs = model(images)
            # loss
            loss = criterion(outputs, labels)
            losses.append(loss.item())
        
        mean_val_loss = np.mean(losses)

        if epo_no % 10 == 0:
            writer.add_images('images', images, epo_no)
            writer.add_images('labels', torch.sum(labels, dim=1, keepdim=True), epo_no)
            writer.add_images('outputs', torch.sum(outputs, dim=1, keepdim=True), epo_no)
            for i in range(8):
                F.to_pil_image(images[i].cpu().detach()).save(f'./{fname[i]}_{epo_no}_{i}.png')
                for j in range(8):
                    F.to_pil_image(labels[i, j].cpu().detach().float()).save(f'./val_lb_{epo_no}_{i}_{j}.png')
                    F.to_pil_image(outputs[i, j].cpu().detach().float()).save(f'./val_op_{epo_no}_{i}_{j}.png')
    
    return mean_val_loss

def stitch_test_results(outputs, fname):
    img = np.zeros((1024, 1024), dtype=np.uint8)
    for i in range(16):
        x = torch.sum(outputs[i], dim=0).cpu().detach().float()
        r, c = i // 4, i % 4
        img[r*256:(r+1)*256, c*256:(c+1)*256] = x.numpy()
    img = cv2.dilate(img, np.ones((15,15), dtype=np.uint8))
    cv2.imwrite(f'./{fname}', img)


# python train_regression.py --task test --test-image-path ../../data/sumitomo_cad/test.txt --resume res/rfl_channel_du/best_eval_model.pth.tar  --batch-size 1
def test(args, model, test_dataset):
    dataloader = DataLoader(batch_size=1, dataset=test_dataset, num_workers=args.workers)
    model.eval()
    with torch.no_grad():
        data_iterator = tqdm(dataloader, total=len(test_dataset) // args.batch_size + 1)
        for img_no, (images, fname) in enumerate(data_iterator):
            fname = fname[0]
            images = images
            images = images.to(device)
            outputs = model(images.float())
            #stitch_test_results(outputs, fname[0])
            torch.save(images[0], f'./{fname}_im.pt')
            torch.save(outputs[0], f'./{fname}_op.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch UNet Training')

    # Task setting
    parser.add_argument('--task', default='train', type=str,
                        choices=['train', 'test'], help='task')

    # Dataset setting
    parser.add_argument('--train-image-path', default='', type=str,
                        help='path to training images')

    parser.add_argument('--val-image-path', default='', type=str,
                        help='path to validation images')

    parser.add_argument('--test-image-path', default='', type=str,
                        help='path to test images')


    # Training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    # Data processing
    parser.add_argument('--augment', dest='augment', action='store_true',
                        help='augment data for training')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    parser.add_argument('--best-loss', type=float, default=np.float('inf'),
                        help='best (minimum) loss of current model.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='trained epoch of current model.')

    main(parser.parse_args())
