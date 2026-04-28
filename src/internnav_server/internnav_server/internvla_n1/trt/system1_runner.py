from typing import Optional

import torch
import tensorrt as trt

class TRTSystem1Runner:
    """Full-model TRT runner.

    The BF16 engine was built from an FP32 ONNX model, so this runner
    casts inputs → fp32 before TRT, and casts the fp32 output
    back to the input dtype.
    """

    def __init__(self, engine_path, logger_level=trt.Logger.INFO):
        self.logger = trt.Logger(logger_level)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream_pt = torch.cuda.Stream()
        self.stream = self.stream_pt.cuda_stream

    def generate_traj(
        self,
        traj_latents: torch.Tensor,
        images_dp: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        # Cast input → fp32 for the FP32-ONNX TRT engine
        latents_fp32 = traj_latents.float().contiguous()
        images_fp32 = images_dp.float().contiguous()
        noise_fp32 = noise.float().contiguous() if noise is not None else None

        # Output is FP32, shape (batch, 32, 3)
        output = torch.empty(
            (latents_fp32.shape[0], 32, 3),
            dtype=torch.float32,
            device=traj_latents.device,
        )

        self.context.set_input_shape('traj_latents', tuple(latents_fp32.shape))
        self.context.set_input_shape('images_dp', tuple(images_fp32.shape))
        if noise_fp32 is not None:
            self.context.set_input_shape('noise', tuple(noise_fp32.shape))
        self.context.set_tensor_address('trajectory', output.data_ptr())

        self.context.set_tensor_address('traj_latents', latents_fp32.data_ptr())
        self.context.set_tensor_address('images_dp', images_fp32.data_ptr())
        if noise_fp32 is not None:
            self.context.set_tensor_address('noise', noise_fp32.data_ptr())
        self.context.execute_async_v3(self.stream)
        self.stream_pt.synchronize()

        # Cast back to caller's dtype
        return output.to(traj_latents.dtype)
