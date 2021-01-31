


import torch
import torch.nn as nn
from .utils import AverageMeter, ProgressMeter



def enter_attack_exit(func):
    def wrapper(attacker, *args, **kwargs):
        attacker.model.attack(True)
        results = func(attacker, *args, **kwargs)
        attacker.model.attack(False)
        return results
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper



class Coach:

    def __init__(
        self, model, device,
        normalizer, optimizer, 
        learning_policy   
    ):
        self.model = model
        self.device = device
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.learning_policy = learning_policy
        self.loss = AverageMeter("Loss")
        self.acc = AverageMeter("Acc.")
        self.progress = ProgressMeter(self.loss, self.acc)

    def save(self, path):
        torch.save(self.model.state_dict(), path + "/paras.pt")

    
    def train(self, trainlaoder, *, epoch=8888):
        self.progress.step() # reset the meter
        self.model.train()
        for inputs, labels in trainlaoder:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outs, loss = self.model(self.normalizer(inputs), labels, epoch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc_count = (outs.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.acc.update(acc_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step()

        return self.loss.avg

class Valider:

    def __init__(self, model, device, normalizer):
        self.model = model
        self.device = device
        self.normalizer = normalizer
        self.acc = AverageMeter("Acc.")

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.acc.reset()
        self.model.eval()
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outs = self.model(self.normalizer(inputs))

            acc_count = (outs.argmax(-1) == labels).sum().item()
            self.acc.update(acc_count, inputs.size(0), mode="sum")
            return self.acc.avg 


