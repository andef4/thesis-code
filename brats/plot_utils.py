import matplotlib.pyplot as plt
import numpy as np


def plot_image_row(images, *, title=None, labels=None, color_map=None, color_maps=None,
                   overlay=None, overlay_alpha=0.8, overlay_color_map='Reds', colorbar=False):
    image_count = len(images)
    if image_count != 1 and colorbar:
        raise ValueError('colorbar only works reliable with a single image')

    if image_count == 1:
        figsize = None
    elif image_count == 2:
        figsize = (10, 5)
    elif image_count == 3:
        figsize = (15, 5)
    elif image_count == 4:
        figsize = (20, 5)
    else:
        figsize = (20, 4)

    figure, plots = plt.subplots(1, len(images), figsize=figsize)
    # when there is only one subplot, matplotlib returns the plot directly instead of an numpy array
    if not isinstance(plots, np.ndarray):
        plots = [plots]

    if title:
        figure.suptitle(title, fontsize=16)

    first_image = None
    for i, plot in enumerate(plots):
        if color_map:
            img = plot.imshow(images[i], cmap=color_map)
        elif color_maps:
            img = plot.imshow(images[i], cmap=color_maps[i])
        else:
            img = plot.imshow(images[i])

        if labels:
            plot.set_title(labels[i])

        if overlay is not None:
            plot.imshow(overlay, alpha=overlay_alpha, cmap=overlay_color_map)
        if i == 0:
            first_image = img
    if colorbar:
        figure.colorbar(first_image)
    plt.show()
