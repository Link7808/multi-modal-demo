import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM
from typing import List

class DiffCategoryTarget:
    def __init__(self, class1_idx: int, class_k_idx: int, alpha: float):
        self.class1_idx = class1_idx  
        self.class_k_idx = class_k_idx  
        self.alpha = alpha

    def __call__(self, model_output):
        w1 = model_output[..., self.class1_idx]
        wk = model_output[..., self.class_k_idx]
        return (w1 - wk) + self.alpha * wk

class DiffCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None, compute_input_gradient=False):
        super(DiffCAM, self).__init__(model, target_layers, use_cuda,
                                      reshape_transform, compute_input_gradient)
    
    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[DiffCategoryTarget] = None,
                 eigen_smooth: bool = False,
                 aug_smooth: bool = False,
                 **kwargs):
        if aug_smooth:
            return self.forward_augmentation_smoothing(input_tensor,
                                                       targets,
                                                       eigen_smooth=eigen_smooth,
                                                       **kwargs)
        return self.forward(input_tensor,
                            targets,
                            eigen_smooth=eigen_smooth,
                            **kwargs)
    
    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[DiffCategoryTarget] = None,
                target_size=None,
                eigen_smooth: bool = False,
                alpha: float = 0.0,
                k: int = 2) -> np.ndarray:
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        W, H = self.get_target_width_height(input_tensor)
        outputs = self.activations_and_grads(input_tensor, H, W)

        if targets is None:
            if isinstance(outputs, (list, tuple)):
                output_data = outputs[0].detach().cpu().numpy()
            else:
                output_data = outputs.detach().cpu().numpy()

            sorted_indices = np.argsort(-output_data, axis=-1)
            targets = []
            for i in range(sorted_indices.shape[0]):
                class1_idx = int(sorted_indices[i, 0])  # 最高预测类别
                class_k_idx = int(sorted_indices[i, k - 1])  # 第 k 高预测类别
                target = DiffCategoryTarget(class1_idx, class_k_idx, alpha)
                targets.append(target)
                print(class1_idx, class_k_idx)

        if self.uses_gradients:
            self.model.zero_grad()
            if isinstance(outputs, (list, tuple)):
                loss = sum([target(output) for target, output in zip(targets, outputs)])
            else:
                loss = sum([target(output) for target, output in zip(targets, [outputs])])
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   target_size,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer), outputs[0], outputs[1]
    
    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))
