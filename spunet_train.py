import os
import torch

from unet_pytorch.train import UNetTrainer
from unet_pytorch.data import Channel_Conv


total_cores = os.cpu_count()

torch.set_num_threads(total_cores - 4)
torch.set_num_interop_threads(total_cores - 4)

INPUTS = "./img/Input"
MASKS = "./img/Input_Target"
TARGETS = "./img/Target"    
CHECKPOINTS = "./Model"  

# Run to convert MASKS to 1 channel
if not len(os.listdir(TARGETS)):
    Channel_Conv(MASKS, TARGETS)

trainer = UNetTrainer(
    inputs_path=INPUTS,
    targets_path=TARGETS,
    model_path=CHECKPOINTS,
    batch_size=8
)

trainer.train(epochs=50, save_interval=10)