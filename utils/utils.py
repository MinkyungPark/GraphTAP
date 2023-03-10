import os
import yaml
import time
import datetime
import random
import torch
import numpy as np


class DotConfig:
    def __init__(self, configs):
        if type(configs) is str:
            configs = self._get_config(configs)
            
        for k, v in configs.items():
            if isinstance(v, dict):
                setattr(self, k, DotConfig(v))
            else:
                setattr(self, k, v)
                
    def _get_config(self, path):
        with open(path, "r") as f:
            configs = yaml.safe_load(f)
        return configs
    
    def overwrite(self, args):
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))
        
def set_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

DTYPE = torch.float
DEVICE = torch.cuda.current_device()
def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	return torch.tensor(x, dtype=dtype, device=device)

def to_device(*xs, device=DEVICE):
	return [x.to(device) for x in xs]

def to(xs, device=DEVICE):
    return [x.to(device) for x in xs]
	
def normalize(x):
	"""
		scales `x` to [0, 1]
	"""
	x = x - x.min()
	x = x / x.max()
	return x

def to_img(x):
    normalized = normalize(x)
    array = to_np(normalized)
    array = np.transpose(array, (1,2,0))
    return (array * 255).astype(np.uint8)

def set_device(device):
	DEVICE = device
	if 'cuda' in device:
		torch.set_default_tensor_type(torch.cuda.FloatTensor)

def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        dir_path = dir_path + '_' + suffix
        os.makedirs(dir_path)
    return dir_path

class Timer:
	def __init__(self):
		self._start = time.time()

	def __call__(self, reset=True):
		now = time.time()
		diff = now - self._start
		if reset:
			self._start = now
		return diff

