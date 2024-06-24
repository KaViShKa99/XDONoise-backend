from torchcam.methods import GradCAMpp
import torch 
import numpy as np
from torch import nn
from captum.attr import IntegratedGradients, GradientShap
from xaiMask import xai_mask

class XDONoiseMethods:
    
    def __init__(self, model, last_layer):
        self.model = model
        self.last_layer = last_layer
        self.criterion = nn.CrossEntropyLoss()

    def pgd_attack_grad_pp(self, images, eps=0.3, alpha=0.01, iters=5, sensitivity=0.5):
        ori_images = images.clone().detach()  # Clone original images for computing eta
        labels = torch.tensor([0])  # Assuming binary classification
        loss = nn.CrossEntropyLoss()
        
        for i in range(iters):
            with torch.enable_grad():
                images.requires_grad_(True)  # Ensure gradients are enabled for images
                with GradCAMpp(self.model, self.last_layer) as cam_extractor:
                    output = self.model(images.unsqueeze(0))
                    class_index = output.squeeze(0).argmax().item()
                    activation_map = cam_extractor(class_index, output)
    
                captured_area = xai_mask(activation_map, images, sensitivity)  # Assuming xai_mask is defined
                outputs = self.model(images.unsqueeze(0))
                cost = loss(outputs, labels)
                gradient_loss = torch.autograd.grad(cost, images, retain_graph=True)[0]
                masked_area = gradient_loss * captured_area
            
            # PGD step
            adv_images = images + alpha * masked_area.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()  # Clamp and detach
        
        return images

    def pgd_attack_IG(self, images, eps=0.3, alpha=0.01, iters=5, threshold=0.01, steps=20):
        ori_images = images.clone().detach()  # Clone original images for computing eta
        labels = torch.tensor([0])  # Assuming binary classification
        loss = nn.CrossEntropyLoss()

        for i in range(iters):
            with torch.enable_grad():
                images.requires_grad_(True)  # Ensure gradients are enabled for images
                integrated_gradients = IntegratedGradients(self.model)
                attributions_ig = integrated_gradients.attribute(images.unsqueeze(0), target=labels, n_steps=steps)
                binary_mask = (attributions_ig.squeeze().cpu().detach().numpy() > threshold).astype(np.uint8)
                outputs = self.model(images.unsqueeze(0))
                cost = loss(outputs, labels)
                gradient_loss = torch.autograd.grad(cost, images, retain_graph=True)[0]
                masked_area = gradient_loss * binary_mask[0]
            
            # PGD step
            adv_images = images + alpha * masked_area.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()  # Clamp and detach
        
        return images

    def pgd_attack_gradient_shap(self, images, eps=0.3, alpha=0.01, iters=5, threshold=0.01, num_samples=20):
        ori_images = images.clone().detach()  # Clone original images for computing eta
        labels = torch.tensor([0])  # Assuming binary classification
        loss = nn.CrossEntropyLoss()

        for i in range(iters):
            with torch.enable_grad():
                images.requires_grad_(True)  # Ensure gradients are enabled for images
                gradient_shap = GradientShap(self.model)
                baseline_tensor = torch.zeros_like(images.unsqueeze(0))

                attributions, delta = gradient_shap.attribute(images.unsqueeze(0),
                                                               n_samples=num_samples,
                                                               baselines=baseline_tensor,
                                                               target=0,
                                                               return_convergence_delta=True)
                binary_mask = (attributions.squeeze().cpu().detach().numpy() > threshold).astype(np.uint8)
                outputs = self.model(images.unsqueeze(0))
                cost = loss(outputs, labels)
                gradient_loss = torch.autograd.grad(cost, images, retain_graph=True)[0]
                masked_area = gradient_loss * binary_mask[0]
            
            # PGD step
            adv_images = images + alpha * masked_area.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()  # Clamp and detach
        
        return images

    def fgsm_attack_grad_pp(self, input_tensor,learning_rate, sensitivity):
        with GradCAMpp(self.model, self.last_layer) as cam_extractor:
            output = self.model(input_tensor.unsqueeze(0))
            class_index = output.squeeze(0).argmax().item()
            activation_map = cam_extractor(class_index, output)

        modified_tensor_smoothed = xai_mask(activation_map, input_tensor, sensitivity)

        output = self.model(input_tensor.unsqueeze(0))
        target_class = torch.tensor([0])
        loss = self.criterion(output, target_class)

        gradient = torch.autograd.grad(loss, input_tensor)[0]
        gradient_mask = gradient * modified_tensor_smoothed

        modified_tensor = input_tensor + learning_rate *torch.sign(gradient_mask)

        return modified_tensor  

    def fgsm_attack_IG(self, input_tensor, threshold, steps ,learning_rate):
        target_class = torch.tensor([0])
        integrated_gradients = IntegratedGradients(self.model)

        attributions_ig = integrated_gradients.attribute(input_tensor.unsqueeze(0), target=target_class, n_steps=steps)
        binary_mask = (attributions_ig.squeeze().cpu().detach().numpy() > threshold).astype(np.uint8)

        output = self.model(input_tensor.unsqueeze(0))
        loss = self.criterion(output, target_class)

        gradient = torch.autograd.grad(loss, input_tensor)[0]
        gradient_mask = gradient * binary_mask[0]

        modified_tensor = input_tensor + learning_rate *torch.sign(gradient_mask)

        return modified_tensor

    def fgsm_attack_gradient_shap(self, input_tensor, threshold, num_samples,learning_rate):
        target_class = torch.tensor([0])
        gradient_shap = GradientShap(self.model)
        baseline_tensor = torch.zeros_like(input_tensor.unsqueeze(0)) 

        attributions, _ = gradient_shap.attribute(input_tensor.unsqueeze(0),
                                                   n_samples=num_samples,
                                                   baselines=baseline_tensor,
                                                   target=0,
                                                   return_convergence_delta=True)

        binary_mask = (attributions.squeeze().cpu().detach().numpy() > threshold).astype(np.uint8)

        output = self.model(input_tensor.unsqueeze(0))
        loss = self.criterion(output, target_class)

        gradient = torch.autograd.grad(loss, input_tensor)[0]
        gradient_mask = gradient * binary_mask[0]

        modified_tensor = input_tensor + learning_rate *torch.sign(gradient_mask)

        return modified_tensor
