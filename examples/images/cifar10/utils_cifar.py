import copy

import torch

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_samples(model, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

     Parameters
     ----------
     model:
         represents the neural network that we want to generate samples from
     savedir: str
         represents the path where we want to save the generated images
     step: int
    @@ -26,24 +26,65 @@ def generate_samples(model, parallel, savedir, step, net_="normal"):
    """
    model.eval()


    def norm(z):
        return (z**2).mean().sqrt().item()

    images = [ ]  # a series of images for each num_steps value, one per step.
    # always use the same seed; the hope is that this will make
    # different generated images easier to compare across runs, since in
    # principle the ideal mapping from noise to image given a certain
    # training set should be deterministic.
    x0 = torch.randn(64, 3, 32, 32, generator=torch.Generator().manual_seed(110)).to(device)

    NS = [ 80, 40, 20, 10]
    with torch.no_grad():
        for num_steps in NS:
            x = x0
            this_steps = [ x ]
            delta_t = 1. / num_steps
            for i in range(num_steps):
                t = torch.tensor(i / num_steps, device=device).unsqueeze(0).expand(64)
                t_next = t + delta_t
                model_output = model(t, x).reshape(64, 3, 32, 32)
                x = x + delta_t * model_output
                this_steps.append(x)
                print(f"step={step}, net={net_}, iter={i+1}/{num_steps}: x norm={norm(x)}, model norm={norm(model_output)}")
            images.append(this_steps)

    def rel_diff(x, y):
        return norm(x - y) / (0.5 * (norm(x) + norm(y)))
    def list_rel_diff(x, y):
        ratio = (len(x) - 1) / (len(y) - 1)
        diffs = [ ]
        for i in range(len(y)):
            diffs.append(rel_diff(x[round(i * ratio)], y[i]))
        return diffs
    for i in range(1, len(NS)):
        print(f"step={step}, net={net_}: Rel-diff between samples={NS[0]} and samples={NS[i]} is {list_rel_diff(images[0], images[i])}")
    # for most recent training step, show outputs with all the
    # num-inference-steps that we did inference with.  these get overwritten.
    for i in range(len(NS)):
        img = images[i][-1].clip(-1, 1) / 2 + 0.5
        save_image(img, savedir + f"{net_}_generated_FM_images_infstep{NS[i]}.png", nrow=8)

    for i in [ 0, len(NS) - 1 ]:
        img = images[i][-1].clip(-1, 1) / 2 + 0.5
        # we don't want to fill up the disk with too many inference outputs, so we write
        # just for the largest and smallest num-steps, e.g. 80 and 1.
        save_image(img, savedir + f"{net_}_generated_FM_images_step_{step}_infstep{NS[i]}.png", nrow=8)

    model.train()



def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x
