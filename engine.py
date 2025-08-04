from iree import compiler as ireec
from iree import runtime as ireert
import numpy as np


class Engine:
    def __init__(self, model):
        self.compiled_vmfb = ireec.tools.compile_file(
            model,
            input_type="auto",
            target_backends=["vulkan-spirv"],
            extra_args=["--iree-vulkan-target=gfx1201", "--iree-opt-level=O3"],
        )
        self.config = ireert.Config("vulkan")
        self.ctx = ireert.SystemContext(config=self.config)
        self.vm_module = ireert.VmModule.copy_buffer(
            self.ctx.instance, self.compiled_vmfb
        )
        self.ctx.add_vm_module(self.vm_module)

    def fix_input_shape(self, input):
        if input.ndim == 2:
            input = np.expand_dims(input, axis=0)
            input = np.expand_dims(input, axis=-1)
        elif input.ndim == 3:
            input = np.expand_dims(input, axis=0)
        return input

    def fix_output_shape(self, output):
        output = np.squeeze(np.array(output))
        if output.ndim == 2:
            output = np.expand_dims(output, axis=-1)
        return output

    def run(self, input):
        input = self.fix_input_shape(input)
        f = self.ctx.modules.module["serve"]
        pred = f(input).to_host()
        pred = self.fix_output_shape(pred)
        return pred
