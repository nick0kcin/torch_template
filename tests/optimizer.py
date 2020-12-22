import unittest
from trainingmanager import TrainingManager


class MyTestCase(unittest.TestCase):

    def check_optimizer(self, path):
        logger = TrainingManager(path)
        model = logger.model()
        params = logger.parameters(model)
        optimizer = logger.optimizer(params)
        scheduler = logger.lr_scheduler(optimizer)
        return optimizer

    def test1(self):
        self.check_optimizer("configs/1")

    def test2(self):
        self.check_optimizer("configs/2")

    def test3(self):
        self.check_optimizer("configs/3")

if __name__ == '__main__':
    unittest.main()
