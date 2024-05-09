# env to use: source /star-ly/env/py39/bin/activate
# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import logging
import torch
from torch import Tensor
from typing import Tuple
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from utils_cifar import ema, generate_samples, infiniteloop, setup_logger
import torch.autograd.forward_ad as fwAD

from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "./results_t_mod_center_1g/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup



def get_xt_and_ut(t: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Returns a warped version of x and the derivative of the result with
    respect to t.  By "warped" we mean, given a random smooth
    warp like that of a clock in a Salvador Dali painting.

    Args:
       t: the time-step, between 0 and 1, where 0 means fully warped
          and 1 means not warped at all.  Of shape (B, 1, 1, 1)
      x1: the target image.   Note, this will be the clean image.
          Of shape (B, 3, H, W) actually (B, 3, 32, 32).
    Returns: (xt, dx_dt)
    """

    # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    device = x1.device
    # First compute a random relative-warp amount.
    (B, _, K, _) = x1.shape
    assert x1.shape[-1] == K  # we're assuming image is square.  this is not really necessary
    assert t.shape == (B, 1, 1, 1)

    x0 = torch.randn_like(x1)

    # smooth out the warp, we can adjust this factor.
    smooth_distance = 5
    num_smooth_iters = 3

    # first get a random gaussian warp amount.  the 2 is for the 2 directions,
    # x and y.
    warp = torch.randn(B, 2,
                       K - num_smooth_iters * (smooth_distance - 1),
                       K - num_smooth_iters * (smooth_distance - 1),
                       device=device)

    # ... and convolve it with a square of ones.
    weight = torch.ones(2, 1, smooth_distance, smooth_distance, device=device)

    for _ in range(num_smooth_iters):
        warp = torch.nn.functional.conv2d(warp, weight, groups=2,
                                          padding=(smooth_distance-1, smooth_distance-1))
    # now warp is spatially smoothed and is close to zero near the edges.
    assert warp.shape == (B, 2, K, K), warp.shape

    warp_rms = (warp ** 2).mean().sqrt()
    warp = warp / warp_rms

    # this dictates the RMS how much the images are warped by, in pixels away from the current
    # point.  We can change this.


    # this warp amount is expressed as a faction of [half the image height].
    warp_amount = 0.1

    # The values given to grid_sample are normalized to between -1 and 1 corresponding
    # to the e.g. top & bottom of the image,
    # so we have to multiply warp_amount by 2 / K since the range from, say, 0 to 1 corresponds
    # to  K / 2 pixels.
    warp = warp * warp_amount
    warp = warp.permute(0, 2, 3, 1)

    theta = torch.eye(3, device=device)[:2]
    theta = theta.unsqueeze(0)
    # this theta is a one-to-one affine map, saying "there is no shift or rotation"
    grid = torch.nn.functional.affine_grid(theta, (1, 1, K, K))
    # grid shape: (1, K, K, 2)
    grid = grid.expand(B, K, K, 2)

    # warp shape now: (B, K, K, 2).  This is the affine grid plus the
    # random spatially-smoothed warping amount.


    # I'm getting some not-implemented errors with fwAD.  It may be a torch version issue,
    # so I'll just compute the derivatives with a small delta.
    delta = 0.001

    x_vals = [ ]
    for n in [ 0, 1 ]:
        if n == 1:
            t = t + torch.full_like(t, delta)

        delta_t = torch.full_like(t, delta) if n == 1 else 0.0
        this_t = t + delta_t
        x1_warped_grid = grid + (1 - this_t) * warp
        this_x1 = torch.nn.functional.grid_sample(x1, x1_warped_grid,
                                                  mode='bilinear', padding_mode='reflection',
                                                  align_corners=False)

        # x0_warped_grid is like x1_warped_grid but we only keep the delta-t-related
        # part of the warping, not the part related to the baseline t.
        # the reason for this is to avoid any effects whereby the magnitude of x0
        # gets smaller due to warping.  The idea is that x0 is random anyway, so we
        # don't really need to worry about the non-changing part of the warp.
        x0_warped_grid = grid - delta_t * warp
        this_x0 = torch.nn.functional.grid_sample(x0, x0_warped_grid,
                                                  mode='bilinear', padding_mode='reflection',
                                                  align_corners=False)

        x = (1 - this_t) * this_x0  + this_t * this_x1
        x_vals.append(x)

    return x_vals[0], (x_vals[1] - x_vals[0]) / delta




def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
    )  # new dropout + bs of 128


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################


    savedir = FLAGS.output_dir + "/"
    os.makedirs(savedir, exist_ok=True)
    writer = SummaryWriter(f"{savedir}/tensorboard")
    setup_logger(f"{savedir}/log")

    def get_time_shape(x):
        for n in range(1, x.ndim):
            x = x.narrow(n, 0, 1)
        return x

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x = next(datalooper).to(device)
            t = torch.rand_like(get_time_shape(x))

            # ut is d/dt (xt)
            xt, ut = get_xt_and_ut(t, x)

            if step == 0:
                save_image(x / 2 + 0.5, savedir + f"orig.png", nrow=8)
                save_image(xt / 2 + 0.5, savedir + f"train.png", nrow=8)

            ut = ut.reshape(FLAGS.batch_size, -1)

            #t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            # import pdb; pdb.set_trace()
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new


            if step % 50 == 0:
                message = f"step={step}, loss={loss.item()}"
                logging.info(message)
                writer.add_scalar('loss', loss, step)

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")

                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"cifar10_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
