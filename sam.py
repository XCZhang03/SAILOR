from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests
import numpy as np
import matplotlib

device_count = torch.cuda.device_count()
print(f"Number of CUDA devices available: {device_count}")
device = f"cuda:{device_count-1}" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("/n/home01/xczhang/.cache/modelscope/hub/models/facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("/n/home01/xczhang/.cache/modelscope/hub/models/facebook/sam3")

def segment_image(image: Image.Image, prompt: str):
    # Segment using text prompt
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.3,
        mask_threshold=0.3,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    print(f"Found {len(results['masks'])} objects")
    # Results contain:
    # - masks: Binary masks resized to original image size
    # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
    # - scores: Confidence scores
    results = {k: v.cpu().numpy() for k, v in results.items()}  # Move results to CPU for further processing
    return results

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

def get_mask_center_pixel(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))
    width = mask.shape[1]
    height = mask.shape[0]
    return int(center_x / width * 1000), int(center_y / height * 1000)

if __name__ == "__main__":
    image = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/sideview_image.png"
    image = Image.open(image).convert("RGB")
    prompt = "a black bottle"
    results = segment_image(image, prompt)
    image_with_masks = overlay_masks(image, results["masks"])
    image_with_masks.save("segmentation_result.png")
