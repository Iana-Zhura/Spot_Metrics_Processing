#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tempfile
from PIL import Image

def overlay_patch_scores(image, patch_scores, terrain_location, show=False, logging=False, saving_folder=None, cluster=False, alpha=0.6):
    """
    Overlay patch scores on the image with specified colors and transparency.

    Args:
        image (np.ndarray): The input image.
        patch_scores (np.ndarray or torch.Tensor): The patch scores (assumed in the range [0,1]).
        terrain_type (str): A label for the type of terrain.
        show (bool): Whether to display the overlay image.
        logging (bool): If True, saves a high DPI temporary file for logging.
        saving_folder (str): Folder to save the overlay image.
        cluster (bool): If True, assume patch_scores are clustered.
        alpha (float): Transparency for blending.
        
    Returns:
        np.ndarray or PIL.Image: The overlayed image (or a PIL image if logging is enabled).
    """
    if not show:
        plt.ioff()
    
    # Create a heatmap from patch scores.
    if cluster:
        heatmap = patch_scores.permute(1, 2, 0).numpy().astype(np.uint8)
    else:
        # If patch_scores is a torch tensor, detach and convert to numpy.
        if hasattr(patch_scores, "detach"):
            patch_scores_np = patch_scores.detach().cpu().numpy()
        else:
            patch_scores_np = patch_scores
        # Scale scores (assumed 0-1) to 0-255 and apply a colormap.
        heatmap = cv2.applyColorMap((patch_scores_np * 255).astype(np.uint8),
                                    cv2.COLORMAP_RAINBOW)
    
    # Resize heatmap to match the input image dimensions if necessary.
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Blend the heatmap with the original image.
    overlay = cv2.addWeighted(heatmap, alpha, image.astype(np.uint8), 1 - alpha, 0)
    
    # Optionally, save the overlay image.
    if saving_folder:
        save_path = f"{saving_folder}/overlay_image_{terrain_location}.png"
        plt.imsave(save_path, overlay)
    
    # Create a figure to display the result.
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    norm = colors.Normalize(vmin=0, vmax=1)
    if not cluster:
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Relative Traversability Map (0-1)', rotation=270, labelpad=15)
    
    if logging:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format='png', dpi=200)
            tmpfile_path = tmpfile.name
        pil_image = Image.open(tmpfile_path)
        plt.close()
        return pil_image

    if show:
        if saving_folder:
            print("Saving image to", saving_folder)
            plt.savefig(f"{saving_folder}/overlay_image_{terrain_location}.png")
        plt.show()
    
    return overlay

def main():
    # --- Load predicted semantic patch scores ---
    semantic_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/patch_scores.npy"
    patch_scores = np.load(semantic_path, allow_pickle=True)
    print("Patch scores shape:", patch_scores.shape)
    
    # Display the raw patch scores.
    plt.imshow(patch_scores)
    plt.title("Raw Semantic Patch Scores")
    plt.show()
    
    # --- Load an example image ---
    image_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/images/pair_0000/image.png"
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image from", image_path)
        return
    
    # Convert image from BGR to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # --- Overlay patch scores onto the image ---
    terrain_location = "greenhouse"
    overlayed_image = overlay_patch_scores(image_rgb, patch_scores, terrain_location, show=True, saving_folder="./", alpha=0.6)
    
if __name__ == "__main__":
    main()
