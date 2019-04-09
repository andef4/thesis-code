import matplotlib.pyplot as plt
import numpy as np


def plot_image_row(images, *, title=None, labels=None, color_map=None, color_maps=None):
    image_count = len(images)

    if image_count == 1:
        figsize = (4, 4)
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

    for i, plot in enumerate(plots):
        if color_map:
            plot.imshow(images[i], cmap=color_map)
        elif color_maps:
            plot.imshow(images[i], cmap=color_maps[i])
        else:
            plot.imshow(images[i])
        if labels:
            plot.set_title(labels[i])

    plt.show()
