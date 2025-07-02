import custom_rasterizer_kernel
import torch


def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    assert pos.device == tri.device
    findices, barycentric = custom_rasterizer_kernel.rasterize_image(
        pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
    )
    return findices, barycentric


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
