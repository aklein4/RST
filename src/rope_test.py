import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from utils.model_utils import RotaryEmbedding


def main():
    
    rope = RotaryEmbedding(64, 1024, 10000)

    plt.matshow(rope.sin_emb.weight.data.T.cpu().detach().numpy())
    plt.show()


if __name__ == "__main__":

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()