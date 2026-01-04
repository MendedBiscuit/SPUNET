import os
import torch

from the_spunet.train import UNetTrainer
from the_spunet.data import Channel_Conv


# total_cores = os.cpu_count()

# torch.set_num_threads(total_cores - 4)
# torch.set_num_interop_threads(total_cores - 4)

INPUTS = "./img/Input"
MASKS = "./img/Input_Masks"
TARGETS = "./img/Masks"    
CHECKPOINTS = "./Checkpoints"  

# Run to convert MASKS to 1 channel
if not len(os.listdir(TARGETS)):
    Channel_Conv(MASKS, TARGETS)

trainer = UNetTrainer(
    inputs_path=INPUTS,
    targets_path=TARGETS,
    model_path=CHECKPOINTS,
    batch_size=8
)

trainer.train(epochs=100, save_interval=10)