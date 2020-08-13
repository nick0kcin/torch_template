import os
import torch.utils.data
from torch import device
from opts import opts
from models.models import  load_model, save_model
from trainers.classify_trainer import  ClassifyTrainer as Trainer
from history import History
from logger import Logger
from torch.utils.tensorboard import SummaryWriter
from albumentations import *
from copy import deepcopy
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
import json
from transforms.random_lines import random_lines, random_mm, random_microscope, AdvancedHairAugmentation
import pandas as pd


def ensemble(model, paths):
    models = [deepcopy(model) for paths in paths]
    for i, pth in enumerate(paths):
        models[i] = load_model(models[i], pth)
        models[i] = models[i].eval().cuda()
    def ens(x):
        res = sum([model(x).sigmoid() for model in models])
        res/=len(models)
        return res
    return ens

if __name__ == '__main__':
    opt = opts().parse()
    logger = Logger(opt.save_dir)
    history = History(opt.save_dir, opt.resume)
    # writer = SummaryWriter()
    torch.backends.cudnn.benchmark = True
    print(opt)
    transforms = {
        "train": Compose([
            # Rotate(),
            IAAAffine(shear=12, p=0.7),
            # RandomSizedCrop((96, 256), 128, 128),
            ShiftScaleRotate(rotate_limit=45, scale_limit=(-0.5, 0.5)),
            Flip(),
            Transpose(),
            ElasticTransform(alpha=100, sigma=25, p=0.5),
            # random_lines(),
            # random_mm(),
            AdvancedHairAugmentation(hairs=10, hairs_folder="hairs"),
            random_microscope(),
            # ImageCompression(quality_lower=80, always_apply=True),
            CoarseDropout(min_holes=8,  max_width=16, max_height=16, p=0.75),
            # OneOf([
            #     CoarseDropout(max_holes=40, min_holes=6),
            # ], p=0.2),
            OneOf([
                CLAHE(),
                GaussNoise(),
                GaussianBlur(),
                # MotionBlur(),
                # RGBShift(),
                RandomBrightnessContrast(),
                RandomGamma()
            ]),
            Normalize()
        ]
        ),
        "val": Compose([
            # Resize(128, 128),
            # Transpose(p=1),
            # CenterCrop(256, 256, always_apply=True),
            Normalize()
        ]),
        "test": Compose([
            IAAAffine(shear=12, p=0.7),
            # RandomSizedCrop((96, 256), 128, 128),
            ShiftScaleRotate(rotate_limit=45, scale_limit=(-0.5, 0.5)),
            Flip(),
            Transpose(),
            ElasticTransform(alpha=100, sigma=25, p=0.5),
            # random_lines(),
            # random_mm(),
            AdvancedHairAugmentation(hairs=10, hairs_folder="hairs"),
            # random_microscope(),
            # ImageCompression(quality_lower=80, always_apply=True),
            CoarseDropout(min_holes=8, max_width=4, max_height=4, p=0.75),
            # OneOf([
            #     CoarseDropout(max_holes=40, min_holes=6),
            # ], p=0.2),
            OneOf([
                CLAHE(),
                GaussNoise(),
                GaussianBlur(),
                # MotionBlur(),
                # RGBShift(),
                RandomBrightnessContrast(),
                RandomGamma()
            ]),
            Normalize()
        ]),
        "predict": Compose([
            IAAAffine(shear=12, p=0.7),
            # RandomSizedCrop((96, 256), 128, 128),
            ShiftScaleRotate(rotate_limit=45, scale_limit=(-0.5, 0.5)),
            Flip(),
            Transpose(),
            ElasticTransform(alpha=100, sigma=25, p=0.5),
            # random_lines(),
            # random_mm(),
            AdvancedHairAugmentation(hairs=10, hairs_folder="hairs"),
            random_microscope(),
            # ImageCompression(quality_lower=80, always_apply=True),
            CoarseDropout(min_holes=8, max_width=16, max_height=16, p=0.75),
            # OneOf([
            #     CoarseDropout(max_holes=40, min_holes=6),
            # ], p=0.2),
            OneOf([
                CLAHE(),
                GaussNoise(),
                GaussianBlur(),
                # MotionBlur(),
                # RGBShift(),
                RandomBrightnessContrast(),
                RandomGamma()
            ]),
            Normalize()
        ]),
    }
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    np.random.seed(0)
    folds = list(range(0, 15))
    np.random.shuffle(folds)
    print(folds)

    aucs = []
    targets = []
    predicts = []
    for it in range(5):
        print({"train":folds[:3*it] + folds[3*it+3:], "val":folds[3*it: 3*it+3]})
        losses, loss_weights = logger.loss
        model = logger.model
        # ens = ensemble(model, [f"/home/nick/PycharmProjects/airbus/exp/melanoma/model_best_{p}.pth" for p in ["012", "123"]])
        teacher = logger.teacher
        params = logger.parameters(model)
        optimizer = logger.optimizer(params)
        lr_schedule = logger.lr_scheduler(optimizer)
        start_epoch = 0
        if opt.load_model != '':
            try:
                model, optimizer, start_epoch, best = load_model(model, opt.load_model,
                                                                 optimizer, opt.resume)
                opt.resume = False
                opt.load_model = ''
            except:
                try:
                    model, optimizer, start_epoch, best = load_model(model, opt.load_model.replace(".", f"{it}."), optimizer, opt.resume)
                    opt.resume = False
                    opt.load_model = ''
                except:
                    pass
        if opt.load_teacher != '' and teacher is not None:
            teacher = load_model(teacher, opt.load_teacher)
        metrics = logger.metric
        trainer = Trainer(model, losses, loss_weights, metrics=metrics, teacher=teacher, optimizer=optimizer,
                          device=opt.device,
                          print_iter=opt.print_iter, num_iter=opt.num_iters, batches_per_update=opt.batches_per_update)
        trainer.set_device(opt.gpus, opt.device)
        loaders = logger.dataloaders(transforms=transforms, folds={"train": folds[:3*it] + folds[3*it+3:],
                                                                   "val": folds[3*it: 3*it+3],
                                                                   "test": folds[3*it: 3*it+3]})
        history.reset()

        if opt.predict:
            try:
                f= open(f"exp/{opt.exp_id}/predict{it}.csv", "r")
            except:
                trainer.predict_file(loaders["predict"], f"exp/{opt.exp_id}/predict{it}.csv")
                if not opt.test:
                    continue
            # continue


        if opt.test:
            # auc = trainer.test(loaders["test"], verbose=1)
            # print(auc)
            # aucs.append(auc)
            pred, gt = trainer.predict_partial(loaders["test"], f"exp/{opt.exp_id}/predict_val{it}.csv")
            targets.extend(gt)
            predicts.extend(pred)
            continue

        if lr_schedule:
            for i in range(start_epoch):
                lr_schedule.step()
            print([group_param["lr"] for group_param in optimizer.param_groups])
        for epoch in range(start_epoch + 1, opt.num_epochs + 1):
            log_dict_val, log_dict_test = None, None
            log_dict_train = trainer.train(epoch, loaders["train"])
            # writer.add_scalars("train", log_dict_train, 1)
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, -1, optimizer)
            if "val" in loaders and opt.val_intervals > 0 and not (epoch % opt.val_intervals):
                with torch.no_grad():
                    log_dict_val = trainer.val(epoch, loaders["val"])
                # writer.add_scalars("val", log_dict_val, opt.val_intervals)

            if "test" in loaders and opt.test_intervals > 0 and not (epoch % opt.test_intervals):
                log_dict_test = trainer.test(loaders["val"])
                # writer.add_scalars("test", log_dict_test, opt.test_intervals)

            need_save, timespamp = history.step(epoch, log_dict_train, log_dict_val, log_dict_test)
            if need_save:
                save_model(os.path.join(opt.save_dir, str(timespamp) + '.pth'),
                           epoch, model, log_dict_train["loss"], optimizer)

            if lr_schedule:
                lr_schedule.step()
                print([group_param["lr"] for group_param in optimizer.param_groups])
        os.rename(os.path.join(opt.save_dir, 'model_last.pth'), os.path.join(opt.save_dir, f'model_last{it}.pth'))
        os.rename(os.path.join(opt.save_dir, 'model_best.pth'), os.path.join(opt.save_dir, f'model_best{it}.pth'))
        aucs.append(history.best[0])

    # print(np.mean(aucs), np.std(aucs))
    val = roc_auc_score(targets, predicts)
    print(val)