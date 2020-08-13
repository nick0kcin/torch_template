import yaml
import time
import os

class History:
    def __init__(self, path, resume, eps=1e-04):
        self.path = path + "/"
        self.log_file = path + "/" + "log.txt"
        if not resume:
            try:
                f = open(self.log_file, "r")
                f.close()
                raise ValueError("log already exist")
            except FileNotFoundError:
                pass
            # f = open(self.log_file, "w")
            # f.close()
            self.history = {}
        else:
            self.history = {}
            with open(self.log_file, "r") as file:
                for line in file:
                    self.history.update(yaml.load(line.replace(",", "\n")))
        if self.history:
            self.best = min([(item["val"]["loss"], timestamp) for timestamp, item in self.history.items()])
            self._epoch = self.history[self.best[1]]
        else:
            self.best = 0, None
            self._epoch = 0
        self.eps = eps

    @property
    def epoch(self):
        return next(reversed(self.history.values()))["epoch"] if self.history else 0

    def reset(self):
        self.best =0, None
        self._epoch = 0

    def step(self, epoch, train_info, val_info=None, test_info=None):
        epoch_info = {"epoch": epoch, "train": train_info}
        if val_info:
            epoch_info.update({"val": val_info})
        if test_info:
            epoch_info.update({"test": test_info})
        timestamp = time.strftime("%b %d %Y %H:%M:%S", time.gmtime(time.time()))
        stamp = timestamp
        self.history.update({timestamp: epoch_info})
        is_save = False
        if test_info and test_info > self.best[0]:
            is_save = True
            try:
                os.rename(self.path + "model_best.pth", self.path + str(timestamp) + ".pth")
            except FileNotFoundError:
                pass
            self.best = test_info, timestamp
            timestamp = "model_best"
        # elif val_info and val_info["loss"] - self.best[0] < self.eps:
        #     is_save = True

        for type, info in epoch_info.items():
            if isinstance(info, dict):
                for key, loss in info.items():
                    if key != "loss":
                        epoch_info[type][key] = float("{0:.3f}".format(loss))
                    else:
                        epoch_info[type][key] = float("{0:.8f}".format(loss))

        with open(self.log_file, "a") as file:
            file.write(yaml.dump({stamp: epoch_info}).replace("\n", ",") + "\n")

        return is_save, timestamp
