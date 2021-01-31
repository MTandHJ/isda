
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from tqdm import tqdm


from .dict2obj import Config
from .config import *


class ModelNotDefineError(Exception): pass
class LossNotDefineError(Exception): pass
class OptimNotIncludeError(Exception): pass
class DatasetNotIncludeError(Exception): pass


# return the num_classes of corresponding data set
def get_num_classes(dataset_type: str):
    if dataset_type in ('mnist', 'cifar10'):
        return 10
    elif dataset_type in ('cifar100', ):
        return 100
    else:
        raise DatasetNotIncludeError("Dataset {0} is not included." \
                        "Refer to the following: {1}".format(dataset_type, _dataset.__doc__))


def load_model(model_type: str):
    """
    cifar: 256
    resnet32: 64 
    """
    if model_type == "cifar":
        from models.cifar import CIFAR
        model = CIFAR
        dim_feature = 256
    elif model_type == "resnet32":
        from models.resnet import resnet32
        model = resnet32
        dim_feature = 64
    else:
        raise ModelNotDefineError(f"model {model_type} is not defined.\n" \
                    f"Refer to the following: {load_model.__doc__}\n")
    return model, dim_feature



class _Normalize:

    def __init__(self, mean=None, std=None):
        self.set_normalizer(mean, std)

    def set_normalizer(self, mean, std):
        if mean is None or std is None:
            self.flag = False
            return 0
        self.flag = True
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.nat_normalize = T.Normalize(
            mean=mean, std=std
        )
        self.inv_normalize = T.Normalize(
            mean=-mean/std, std=1/std
        )

    def _normalize(self, imgs, inv):
        if not self.flag:
            return imgs
        if inv:
            normalizer = self.inv_normalize
        else:
            normalizer = self.nat_normalize
        new_imgs = [normalizer(img) for img in imgs]
        return torch.stack(new_imgs)

    def __call__(self, imgs, inv=False):
        # normalizer will set device automatically.
        return self._normalize(imgs, inv)


def _get_normalizer(dataset_type: str):
    mean = MEANS[dataset_type]
    std = STDS[dataset_type]
    return _Normalize(mean, std)


def _get_transform(dataset_type: str, transform: str, train=True):
    """
    Transform:
    default: the default transform for each data set
    simclr: the transform introduced in SimCLR
    """
    try:
        if train:
            print(f"Augmentation: {transform}")
            return TRANSFORMS[dataset_type][transform]
        else:
            print(f"Augmentation: T.ToTensor")
            return T.ToTensor()
    except KeyError:
        raise DatasetNotIncludeError(f"Dataset {dataset_type} or transform {transform} is not included.\n" \
                        f"Refer to the following: {_get_transform.__doc__}")


def _dataset(dataset_type: str, train=True):
    """
    Dataset:
    mnist: MNIST
    cifar10: CIFAR-10
    cifar100: CIFAR-100
    """

    if dataset_type == "mnist":
        dataset = torchvision.datasets.MNIST(
            root=ROOT, train=train, download=False,
            transform=None
        )
    elif dataset_type == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=ROOT, train=train, download=False,
            transform=None
        )
    elif dataset_type == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=ROOT, train=train, download=False,
            transform=None
        )
        
    return dataset


def load_normalizer(dataset_type: str):
    normalizer = _get_normalizer(dataset_type)
    return normalizer


def load_dataset(
    dataset_type: str, transform='default', 
    train=True, double=False
):  
    dataset = _dataset(dataset_type, train)
    transform = _get_transform(dataset_type, transform, train)
    if double:
        from .datasets import DoubleSet
        print("Dataset: DoubleSet")
        dataset = DoubleSet(dataset, transform)
    else:
        from .datasets import SingleSet
        print("Dataset: SingleSet")
        dataset = SingleSet(dataset, transform)

    return dataset


class _TQDMDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        return iter(
            tqdm(
                super(_TQDMDataLoader, self).__iter__(), 
                leave=False, desc="վ'ᴗ' ի-"
            )
        )


def load_dataloader(dataset, batch_size: int, train=True, show_progress=False):
    loader = _TQDMDataLoader if show_progress else torch.utils.data.DataLoader
    if train:
        dataloader = loader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY)
    else:
        dataloader = loader(dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY)
    return dataloader


def load_optimizer(
    model: torch.nn.Module, 
    optim_type: str, *,
    lr=0.1, momentum=0.9,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    nesterov=False,
    **kwargs
):
    """
    sgd: SGD
    adam: Adam
    """
    try:
        cfg = OPTIMS[optim_type]
    except KeyError:
        raise OptimNotIncludeError(f"Optim {optim_type} is not included.\n" \
                        f"Refer to the following: {load_optimizer.__doc__}")
    
    kwargs.update(lr=lr, momentum=momentum, betas=betas, 
                weight_decay=weight_decay, nesterov=nesterov)
    
    cfg.update(**kwargs) # update the kwargs needed automatically
    print(optim_type, cfg)
    if optim_type == "sgd":
        optim = torch.optim.SGD(model.parameters(), **cfg)
    elif optim_type == "adam":
        optim = torch.optim.Adam(model.parameters(), **cfg)

    return optim


def load_learning_policy(
    optimizer: torch.optim.Optimizer,
    learning_policy_type: str,
    **kwargs
):
    """
    default: 10x decay at 80, 120， epochs=160, weight_decay=1e-4
    cosine: CosineAnnealingLR, kwargs: T_max, eta_min, last_epoch
    """
        
    try:
        learning_policy_ = LEARNING_POLICY[learning_policy_type]
    except KeyError:
        raise NotImplementedError(f"Learning_policy {learning_policy_type} is not defined.\n" \
            f"Refer to the following: {load_learning_policy.__doc__}")

    lp_type = learning_policy_[0]
    lp_cfg = learning_policy_[1]
    lp_description = learning_policy_[2]
    lp_cfg.update(**kwargs) # update the kwargs needed automatically
    print(lp_type, lp_cfg)
    print(lp_description)
    learning_policy = getattr(
        torch.optim.lr_scheduler, 
        lp_type
    )(optimizer, **lp_cfg)
    
    return learning_policy


def generate_path(method: str, dataset_type: str, model:str,  description: str):
    info_path = INFO_PATH.format(
        method=method,
        dataset=dataset_type,
        model=model,
        description=description
    )
    log_path = LOG_PATH.format(
        method=method,
        dataset=dataset_type,
        model=model,
        description=description
    )
    return info_path, log_path


def load_valider(model: nn.Module, device, dataset_type: str):
    from .base import Valider
    valider = Valider(
        model=model,
        device=device,
        normalizer=load_normalizer(dataset_type)
    )
    return valider


    