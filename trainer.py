import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

from utils import val_single_volume


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.00001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    epoch_loss = []
    epoch_ce_loss = []
    epoch_dice_loss = []

    for epoch_num in iterator:
        model.train()
        total_loss = 0
        total_ce_loss = 0
        total_dice_loss = 0
        num_batches = len(trainloader)
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            total_loss += loss.item()
            total_ce_loss += loss_ce.item()
            total_dice_loss += loss_dice.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, loss_ce: %f, lr: %f' % (iter_num, epoch_num, loss.item(), loss_ce.item(), lr_))
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
        if (epoch_num+1) % 1 == 0 and (epoch_num+1) >= 299:

            logging.info("{} test iterations per epoch".format(len(testloader)))
            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in tqdm(enumerate(testloader)):
                image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
                metric_i = val_single_volume(image, label, model, classes=args.num_classes,
                                             patch_size=[args.img_size, args.img_size],
                                             case=case_name, z_spacing=1)
                metric_list += np.array(metric_i)
            metric_list = metric_list / len(db_test)
            performance = np.mean(metric_list, axis=0)[0]
            logging.info(
                'Testing performance in val model: mean_dice : %f,  best_dice : %f' % (performance, best_performance))
            # performance = inference(args, model, best_performance)
            if (best_performance < performance):
                best_performance = performance
                save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        save_interval = 50

        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

        epoch_loss.append(total_loss / num_batches)
        epoch_ce_loss.append(total_ce_loss / num_batches)
        epoch_dice_loss.append(total_dice_loss / num_batches)

    epochs = range(1, max_epoch)  # 创建每个 epoch 的范围
    plt.figure(figsize=(16, 5))  # 增加图像的宽度

    # 绘制总损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, epoch_loss, label='Total Loss', color='blue', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, max_epoch, 50))  # 设置 x 轴刻度为每个 epoch
    plt.legend()

    # 绘制交叉熵损失
    plt.subplot(1, 3, 2)
    plt.plot(epochs, epoch_ce_loss, label='Cross Entropy Loss', color='orange', marker='o')
    plt.title('Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, max_epoch, 50))  # 设置 x 轴刻度为每个 epoch
    plt.legend()

    # 绘制Dice损失
    plt.subplot(1, 3, 3)
    plt.plot(epochs, epoch_dice_loss, label='Dice Loss', color='green', marker='o')
    plt.title('Dice Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, max_epoch, 50))  # 设置 x 轴刻度为每个 epoch
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(snapshot_path, 'training_curve.png'))
    plt.close()

    writer.close()
    return "Training Finished!"