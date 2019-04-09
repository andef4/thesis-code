import matplotlib.pyplot as plt


def plot_image_row(images, *, title=None, labels=None, color_maps=None):
    image_count = len(images)
    assert image_count > 1

    if image_count >= 5:
        figsize = (20, 4)
    elif image_count == 4:
        figsize = (20, 5)
    elif image_count == 3:
        figsize = (15, 5)
    elif image_count == 2:
        figsize = (10, 5)

    figure, plots = plt.subplots(1, len(images), figsize=figsize)
    if title:
        figure.suptitle(title, fontsize=16)

    for i, plot in enumerate(plots):
        if color_maps:
            plot.imshow(images[i], cmap=color_maps[i])
        else:
            plot.imshow(images[i])
        if labels:
            plot.set_title(labels[i])

    plt.show()
