import numpy as np
import open_clip
import timm
import torch
from open_clip.factory import HF_HUB_PREFIX
from open_clip.modified_resnet import ModifiedResNet
from open_clip.timm_model import TimmModel
from open_clip.transformer import VisionTransformer
from PIL import Image
from torch import Tensor, nn

from gradcam.hook import Hook


def reshape_transform(
    model: nn.Module, tensor: Tensor, grid_size: tuple[int, int]
) -> Tensor:
    tensor.squeeze()
    result = tensor
    if isinstance(model.visual, VisionTransformer):
        result = tensor[:, 1:, :].reshape(
            tensor.size(0), grid_size[0], grid_size[1], tensor.size(2)
        )
    if isinstance(model.visual, TimmModel):
        if isinstance(
            model.visual.trunk, timm.models.vision_transformer.VisionTransformer
        ):
            result = tensor.reshape(
                tensor.size(0), grid_size[0], grid_size[1], tensor.size(2)
            )

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_gradient(model: nn.Module, hook: Hook) -> Tensor:
    if isinstance(model.visual, VisionTransformer):
        return reshape_transform(model, hook.gradient.float(), model.visual.grid_size)
    if isinstance(model.visual, TimmModel):
        if isinstance(
            model.visual.trunk, timm.models.vision_transformer.VisionTransformer
        ):
            return reshape_transform(model, hook.gradient.float(), (14, 14))
    return hook.gradient.float()


def get_activation(model: nn.Module, hook: Hook) -> Tensor:
    if isinstance(model.visual, VisionTransformer):
        return reshape_transform(model, hook.activation.float(), model.visual.grid_size)
    if isinstance(model.visual, TimmModel):
        if isinstance(
            model.visual.trunk, timm.models.vision_transformer.VisionTransformer
        ):
            return reshape_transform(model, hook.activation.float(), (14, 14))
    return hook.activation.float()


# https://arxiv.org/abs/1610.02391
def get_attention_map_for_layer(
    model: nn.Module, input: Tensor, target: Tensor, layer: nn.Module
) -> Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)

    with torch.cuda.amp.autocast(), torch.autograd.set_detect_anomaly(
        True
    ), torch.set_grad_enabled(True), Hook(layer) as hook:
        image_features = model.encode_image(input)
        text_features = model.encode_text(target)

        normalized_image_features = image_features / torch.linalg.norm(
            image_features, dim=-1, keepdim=True
        )
        normalized_text_features = text_features / torch.linalg.norm(
            text_features, dim=-1, keepdim=True
        )
        text_probs = 100.0 * normalized_image_features @ normalized_text_features.T
        text_probs[:, 0].backward()

        grad = get_gradient(model, hook)
        act = get_activation(model, hook)

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        attention_map = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        attention_map = torch.clamp(attention_map, min=0)
        # Normalize the attention_map
        attention_map /= torch.max(attention_map)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return attention_map


def get_layer(model: nn.Module) -> nn.Module | None:
    if isinstance(model.visual, ModifiedResNet):
        return model.visual.layer4
    if isinstance(model.visual, TimmModel):
        if isinstance(
            model.visual.trunk, timm.models.vision_transformer.VisionTransformer
        ):
            return model.visual.trunk.blocks[-1].norm1
        return model.visual.trunk.stages[-1]
    if isinstance(model.visual, VisionTransformer):
        return model.visual.transformer.resblocks[-1].ln_1
    return None


def get_attention_map(
    image_path: str, caption_text: str, model_name: str, pretrain_tag: str | None = None
) -> np.ndarray:
    if model_name.startswith(HF_HUB_PREFIX):
        model, preprocess = open_clip.create_model_from_pretrained(model_name)
        print(
            f"Executing gradcam for model '{model_name}' with image '{image_path}' and caption '{caption_text}'"
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrain_tag
        )
        print(
            f"Executing gradcam for model '{model_name}' - '{pretrain_tag}' with image '{image_path}' and caption '{caption_text}'"
        )

    tokenizer = open_clip.get_tokenizer(model_name)
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    caption = tokenizer([caption_text])

    layer = get_layer(model)
    if layer is None:
        raise Exception("Model type is not supported")

    attention_map = get_attention_map_for_layer(
        model,
        image,
        caption,
        layer,
    )
    return attention_map.squeeze().detach().cpu().numpy()
