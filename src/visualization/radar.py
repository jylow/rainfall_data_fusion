import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs


def improved_visualise_radar_grid(
    data: pd.Series,
    ax=None,
    zoom=None,
    vmin=0,
    vmax=None,
    cmap=plt.get_cmap("turbo").copy(),
    mask_threshold=0.1,
    add_basemap=True,
    title=None,
    colorbar=True,
    norm=None,
):
    """
    Visualize weather radar data with proper geographic context.

    Parameters:
    -----------
    data : pd.Series with keys 'data', 'bounds', 'crs', 'transform'
    ax : matplotlib/cartopy axis (optional)
    zoom : dict with 'left', 'right', 'bottom', 'top' (optional)
    vmin, vmax : color scale limits
    cmap : colormap name or object
    mask_threshold : minimum value to display (skip light rain)
    add_basemap : whether to add coastlines and borders
    title : plot title
    colorbar : whether to add colorbar
    """

    d = data["data"]
    bounds = data["bounds"]
    if zoom is None:
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        grid_to_plot = d.copy()
    else:
        extent = [zoom["left"], zoom["right"], zoom["bottom"], zoom["top"]]

        clip_left = round(
            (zoom["left"] - bounds.left) / (bounds.right - bounds.left) * d.shape[1]
        )
        clip_right = round(
            (zoom["right"] - bounds.left) / (bounds.right - bounds.left) * d.shape[1]
        )
        clip_top = round(
            (bounds.top - zoom["top"]) / (bounds.top - bounds.bottom) * d.shape[0]
        )
        clip_bottom = round(
            (bounds.top - zoom["bottom"]) / (bounds.top - bounds.bottom) * d.shape[0]
        )

        grid_to_plot = d[clip_top:clip_bottom, clip_left:clip_right]

    # Mask low values (like kriging does)
    # data_arr = np.ma.masked_where(d < mask_threshold, d)
    data_arr = np.array(grid_to_plot)

    # Setup extent
    if zoom is not None:
        print(bounds)

    print(data_arr.shape)

    # Plot raster data
    im = ax.imshow(
        data_arr,
        extent=extent,
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
        transform=ccrs.PlateCarree(),
        alpha=1,
        norm=norm,
    )

    # Add geographic features
    if add_basemap:
        ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=1, linestyle="--"
        )

    # Add title
    if title:
        ax.set_title(title, fontsize=12, pad=10)

    # Add colorbar
    # if colorbar:
    # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label('Rainfall (mm)', rotation=270, labelpad=15)

    return im, ax
