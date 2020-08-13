import yaml
from losses.losses import create_loss
from models.models import create_model
from lr_schedulers.scheduler import create_lr_schedule, get_lr_schedule_class
from lr_schedulers.compose import compose_wrapper
from optimizers.optimizer import create_optimizer
from metrics.metrics import create_metric, MetricsHandler
from datasets.datasets import create_dataset, create_dataloader, create_sampler
from albumentations.core.serialization import from_dict
from collators.collators import get_collator


class Logger:
    def __init__(self, path):
        self.path = path
        with open(path + "/" + "losses.txt") as file:
            self.losses = yaml.load(file.read(), Loader=yaml.FullLoader)
        with open(path + "/" + "model.txt") as file:
            self.models = yaml.load(file.read(), Loader=yaml.FullLoader)
        with open(path + "/" + "optimizer.txt") as file:
            self.optimizers = yaml.load(file.read(), Loader=yaml.FullLoader)
        with open(path + "/" + "params.txt") as file:
            self.params = yaml.load(file.read(), Loader=yaml.FullLoader)
        with open(path + "/" + "datasets.txt") as file:
            self.datasets = yaml.load(file.read(), Loader=yaml.FullLoader)
        try:
            with open(path + "/" + "trainer.txt") as file:
                self.trainer = yaml.load(file.read(), Loader=yaml.FullLoader)
        except FileNotFoundError:
            self.trainer = {}
            print(" no trainer params info")
        try:
            with open(path + "/" + "lr_scheduler.txt") as file:
                self.lr_schedulers = yaml.load(file.read(), Loader=yaml.FullLoader)
        except FileNotFoundError:
            print("no lr schedule info")
            self.lr_schedulers = None
        try:
            with open(path + "/" + "metrics.txt") as file:
                self.metrics = yaml.load(file.read(), Loader=yaml.FullLoader)
        except FileNotFoundError:
            print("no metrics info")
            self.metrics = None

    @property
    def loss(self):
        losses = {}
        loss_weights = {}
        for i, (loss, data) in enumerate(self.losses.items()):
            losses.update({loss: create_loss(data["func"], **data.get("kwargs", {}))})
            loss_weights.update({loss: data.get("weight", 1)})
        return losses, loss_weights

    @property
    def metric(self):
        if self.metrics:
            return MetricsHandler({name: create_metric(name, **data.get("kwargs", {}) if data["kwargs"] else {}) for name, data in self.metrics.items()})
        return MetricsHandler({})

    @property
    def model(self):
        if len(self.models) > 2:
            raise ValueError("more than one model in info")
        model_name, kwargs = next(iter(self.models.items()))
        return create_model(model_name, **kwargs)

    @property
    def teacher(self):
        if len(self.models) > 2:
            raise ValueError("more than one model in info")
        if len(self.models) < 2:
            return None
        model_name, kwargs = list(self.models.items())[1]
        return create_model(model_name, **kwargs)

    def lr_scheduler(self, optimizer):
        if self.lr_schedulers:
            if len(self.lr_schedulers) > 1:
                schedulers_named_args = []
                schedulers_args = []
                steps = []
                classes = []
                lrs = []
                for data in self.lr_schedulers:
                    if "start" not in data:
                        raise ValueError("lr schedule not starts")
                    elif "class" not in data:
                        raise ValueError("lr schedule class missing")
                    else:
                        schedulers_args.append(data.get("args", []))
                        schedulers_named_args.append(data.get("kwargs", {}))
                        steps.append(data["start"])
                        classes.append(get_lr_schedule_class(data["class"]))
                        lrs.append(data.get("lr", []))
                        lrs[-1] = [lrs[-1][p["lr"]] for p in optimizer.param_groups]
                        if lrs[-1] and len(optimizer.param_groups) != len(lrs[-1]):
                            raise ValueError("non consistent size of optimizer group params")
                return compose_wrapper(classes, schedulers_args, schedulers_named_args, steps, lrs)(optimizer)
            else:
                scheduler_name, kwargs = next(iter(self.lr_schedulers.items()))
                return create_lr_schedule(scheduler_name, optimizer, **kwargs)
        return None

    def parameters(self, model):
        param = []
        if self.params:
            for name in self.params:
                try:
                    a = model
                    for p in name.split("."):
                        a = getattr(a, p)
                    param.append({"params": dict(a.named_parameters()), "lr": 0.0})
                except AttributeError:
                    param.append({"params": getattr(model, name)(), "lr": 0.0})
            return param
        return model.parameters()

    def optimizer(self, parameters):
        if len(self.optimizers) > 1:
            raise ValueError("more than one optimizer in info")
        model_name, kwargs = next(iter(self.optimizers.items()))
        # parameters = dict(parameters)
        # config_params = []
        # if isinstance(kwargs["weight_decay"], list):
        #     for i, param in enumerate(parameters):
        #         p = dict(param["params"])
        #         p_with_bias = [param for name, param in p.items() if not( not name.endswith("bias") and param.requires_grad and len(param.shape) > 1)]
        #         p_without_bias = [param for name, param in p.items() if not name.endswith("bias") and param.requires_grad and len(param.shape) > 1]
        #         config_params.append([
        #             {"params": p_without_bias,
        #              "weight_decay": kwargs["weight_decay"][i]},
        #             {"params": p_with_bias,
        #              "weight_decay": 0.0}
        #             ]
        #         )

        if isinstance(kwargs["weight_decay"], list):
            parameters = [{"params": param,
                           "weight_decay": kwargs["weight_decay"][i]
                           if not name.endswith("bias") and param.requires_grad else 0.0,
                           "lr": i}
                          for i, params in enumerate(parameters) for name, param in params["params"].items()
            ]
        return create_optimizer(model_name, parameters, **kwargs)

    @property
    def trainer_params(self):
        return self.trainer

    def dataloaders(self, transforms=None, folds=None):
        dataloaders = {}
        for type, data in self.datasets.items():
            if "dataset" not in data or "dataloader" not in data:
                raise ValueError("dataset info missing for {%type}")
            if "transform" in data:
                transform = from_dict(data["transform"])
            else:
                transform = transforms[type] if type in transforms else None
            if folds and type in folds:
                fold = folds[type]
            elif "folds" in data:
                fold = data["folds"]

            dataset = create_dataset(data["dataset"]["class"], transform, folds=fold, **data["dataset"].get("kwargs", {}))
            if "sampler" in data:
                sampler = create_sampler(data["sampler"]["class"], dataset,  data["sampler"].get("kwargs", {}))
            else:
                sampler = None
            if "collate_fn" in data:
                collate = get_collator(data["collate_fn"])
            else:
                collate = None
            dataloader = create_dataloader(data["dataloader"]["class"], dataset, sampler, collate,
                                           **data["dataloader"].get("kwargs", {}))
            dataloaders.update({type: dataloader})
        return dataloaders
