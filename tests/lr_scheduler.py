import unittest
from trainingmanager import TrainingManager
from torch.optim import Adam
from torch.autograd import Variable
import torch
from torch.optim.lr_scheduler import *


class MyTestCase(unittest.TestCase):
    def check_schedule(self, path, gt_scheduler, gt_lr):
        logger = TrainingManager(path)
        w = Variable(torch.zeros((1,)))
        optimizer = Adam([w])
        lr_scheduler = logger.lr_scheduler(optimizer)
        for epoch, (gold_class, gold_lr) in enumerate(zip(gt_scheduler, gt_lr)):
            self.assertEqual(lr_scheduler.current_scheduler.__class__,
                             gold_class)
            self.assertEqual(lr_scheduler.get_lr(), [gold_lr])
            lr_scheduler.step()

    def check_schedule_groups(self, path, gt_scheduler, gt_lr):
        logger = TrainingManager(path)
        w1 = Variable(torch.zeros((1,)))
        w2 = Variable(torch.zeros((1,)))
        optimizer = Adam([{"params": [w1], "lr":0.0}, {"params": [w2], "lr":0.0}])
        lr_scheduler = logger.lr_scheduler(optimizer)
        for epoch, (gold_class, gold_lr) in enumerate(zip(gt_scheduler, gt_lr)):
            print(epoch)
            self.assertEqual(lr_scheduler.current_scheduler.__class__,
                             gold_class)
            self.assertEqual(lr_scheduler.get_lr(), gold_lr)
            lr_scheduler.step()

    def test1(self):
        self.check_schedule("configs/1", [StepLR] * 20,
                            [0.000003] * 5 + [0.000003 * 1.2] * 5 + [0.000265] * 10)

    def test2(self):
        self.check_schedule("configs/2",
                            [StepLR] * 2 + [MultiStepLR] * 6 + [ExponentialLR] * 10,
                            ([0.000003, 0.000003 * 1.2]
                             + [0.000265] * 3 + [0.000265 * 0.75] * 2 + [0.000265 * 0.75 ** 2] +
                             [0.0009 * 0.8 ** i for i in range(12)])
                            )

    def test3(self):
        lr1 = [0.000003, 0.0000032]
        lr2 = [0.000265, 0.0002652]
        lr3 = [0.0009, 0.00092]
        self.check_schedule_groups("configs/3", [StepLR] * 2 + [MultiStepLR] * 6 + [ExponentialLR] * 10,
                                   ([lr1, [ v * 1.2 for v in lr1]]
                                    + [lr2] * 3 + [[v * 0.75 for v in lr2]] * 2 + [[v * 0.75 ** 2 for v in lr2]] +
                                    [[v * 0.8 ** i for v in lr3] for i in range(12)])
                                   )


if __name__ == '__main__':
    unittest.main()
