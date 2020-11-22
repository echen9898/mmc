
from PIL import Image
import os, sys
import numpy as np
import torch
from torch import nn

class MMC(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            action_size
            ):
        # Initialize any models here
        pass

    def forward(self):
        # usually called in compute_bonus or compute_loss to generate state predictions etc. Not usually called
        # anywhere else but in this file. 
        pass

    def compute_bonus(self, observations, actions):
        # called by another method in the codebase - can take in observations and actions. If we need
        # access to other things we can pass them up from the sampler. Returns the intrinsic reward,
        # which is either a single reward (if we are processing samples one at a time while sampling)
        # or a batch of rewards for the whole rollout (recommended). This method is called with
        # torch.no_grad, so no worries about anything backpropagating through these variables.
        
        pass
        # return r_int


    def compute_loss(self, observations, actions):
        # called by another method in the codebase - can take in observations and actions. If we need
        # access to other things we can pass them up from the sampler. Returns the loss over the entire
        # recently sampled batch. "loss.backprop" will be called in the method that calls this method. 
        # Variables involved in the loss will go through backprop.
        
        pass
        # return loss



