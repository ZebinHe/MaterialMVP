import numpy as np
from PIL import Image


class imageSuperNet:
    def __init__(self, config) -> None:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=config.realesrgan_ckpt_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=None,
        )
        self.upsampler = upsampler

    def __call__(self, image):
        output, _ = self.upsampler.enhance(np.array(image))
        output = Image.fromarray(output)
        return output
