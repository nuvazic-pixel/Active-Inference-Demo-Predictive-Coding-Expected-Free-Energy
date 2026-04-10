from __future__ import annotations

from . import mini_pc_pytorch as linear_pc
from . import mini_pc_active_inference as active_pc
from . import mnist_foveated_active_inference_lite as mnist_pc


def linear_demo() -> None:
    linear_pc.train(linear_pc.build_argparser().parse_args())


def active_demo() -> None:
    active_pc.main()


def mnist_demo() -> None:
    mnist_pc.main()
