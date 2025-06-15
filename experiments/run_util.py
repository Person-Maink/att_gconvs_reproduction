import os
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d   # pip install scipy
import numpy as np

def model_directory(args):
    # Create name from arguments
    comment = "model_{}_optim_{}_lr_{}_wd_{}_seed_{}/".format(args.model, args.optim, args.lr, args.weight_decay, args.seed)
    if args.extra_comment is not "": comment = comment[:-1] + "_" + args.extra_comment + comment[-1]
    # Create directory
    modeldir = "./saved/" + comment
    os.makedirs(modeldir, exist_ok=True)
    # Add the path to the args
    args.path = modeldir + "model.pth"

def test(model, test_loader, device):
    # send model to device
    model.eval()
    model.to(device)

    # Summarize results
    lbls = []
    pred = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results
    print('Accuracy of the network on the {} test images: {}'.format(total, (100 * correct / total)))
    # Return results
    return correct/total, lbls, pred


def plot_curves(path, smooth=5):
    """Plot (val_acc, val_loss, train_acc, train_loss) stored in a (N,4) array."""
    hist = np.load(path, allow_pickle=True)
    val_acc, val_loss, train_acc, train_loss = hist
    epochs = np.arange(len(val_acc))
    plt.rcParams.update({'font.size': 14})


    # optional smoothing
    if smooth and smooth > 1:
        val_acc   = uniform_filter1d(val_acc,   size=smooth)
        train_acc = uniform_filter1d(train_acc, size=smooth)
        val_loss  = uniform_filter1d(val_loss,  size=smooth)
        train_loss= uniform_filter1d(train_loss,size=smooth)

    fig, ax_acc = plt.subplots(figsize=(10, 6))
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")

    ax_acc.plot(epochs, train_acc, label="train acc", linewidth=4.5)
    ax_acc.plot(epochs, val_acc,   label="val acc",   linewidth=4.5)
    ax_acc.legend(loc="upper left")
    ax_acc.grid(True, linestyle="--", alpha=0.3)

    ax_loss = ax_acc.twinx()
    ax_loss.set_ylabel("loss")
    ax_loss.plot(epochs, train_loss, label="train loss",
                 linewidth=4.5, linestyle="--")
    ax_loss.plot(epochs, val_loss,   label="val loss",
                 linewidth=4.5, linestyle="--")
    ax_loss.legend(loc="upper right")

    fig.savefig(path + "_plot.png", dpi=300, bbox_inches='tight')

    fig.tight_layout()
    plt.show()

