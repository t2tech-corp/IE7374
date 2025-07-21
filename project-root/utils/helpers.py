# # project-root/utils/helpers.py

# Returns the appropriate device (cuda or cpu)

import torch

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Additional utility functions added here