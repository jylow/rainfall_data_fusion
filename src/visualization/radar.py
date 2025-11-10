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

def visualize_one_radar_image_with_cropping(radar_df, sg_bounds):
    # pick one radar frame
    radar_row = radar_df.iloc[0]   # first frame
    data = radar_row['data']       # 2D numpy array
    bounds = radar_row['bounds']   # (left, bottom, right, top)

    # extract bounds
    left, bottom, right, top = bounds
    nrows, ncols = data.shape

    # DEBUG: Check data statistics
    print(f"Data shape: {data.shape}")
    print(f"Data min: {data.min()}, max: {data.max()}, mean: {data.mean()}")
    print(f"Non-zero values: {(data > 0).sum()} out of {data.size}")
    
    # make coordinate grids
    x = np.linspace(left, right, ncols)
    y = np.linspace(bottom, top, nrows)
    X, Y = np.meshgrid(x, y)

    # boolean mask for the coordinate grids
    mask_lon = (X[0,:] >= sg_bounds["left"]) & (X[0,:] <= sg_bounds["right"])
    mask_lat = (Y[:,0] >= sg_bounds["bottom"]) & (Y[:,0] <= sg_bounds["top"])

    # crop data and grids
    data_crop = data[np.ix_(mask_lat, mask_lon)]
    X_crop = X[np.ix_(mask_lat, mask_lon)]
    Y_crop = Y[np.ix_(mask_lat, mask_lon)]

    # DEBUG: Check cropped data
    print(f"\nCropped data shape: {data_crop.shape}")
    print(f"Cropped data min: {data_crop.min()}, max: {data_crop.max()}")
    print(f"Cropped non-zero values: {(data_crop > 0).sum()}")

    # Mask zero/no-rain values
    data_crop_masked = np.ma.masked_where(data_crop <= 0, data_crop)

    # plot cropped radar
    fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([sg_bounds["left"], sg_bounds["right"], sg_bounds["bottom"], sg_bounds["top"]],
                crs=ccrs.PlateCarree())
    
    # Use better colormap and set vmin/vmax explicitly
    cmap = plt.get_cmap("Blues")
    cmap.set_bad(color='white')  # Show masked values as white
    
    im = ax.pcolormesh(X_crop, Y_crop, data_crop_masked, 
                       cmap=cmap, 
                       vmin=0.1,  # Start colormap at 0.1 mm/hr
                       vmax=data_crop.max() if data_crop.max() > 0 else 10)
    
    plt.colorbar(im, ax=ax, orientation='vertical', label="Rainfall (mm/hr)")
    plt.title(f"Radar frame (Singapore crop): {radar_row['time_sgt']}")
    
    # Add coastlines for reference
    ax.coastlines()
    
    plt.show()

def visualize_one_radar_image(radar_df, n = 1):
    """
    Visualizes the first radar frame from the DataFrame without cropping.
    
    Assumes 'data', 'bounds', and 'time_sgt' columns exist.
    """
    # pick one radar frame
    for i in range(n):
        radar_row = radar_df.iloc[i]   # first frame
        data = radar_row['data']       # 2D numpy array
        bounds = radar_row['bounds']   # (left, bottom, right, top)

        # extract bounds
        left, bottom, right, top = bounds
        nrows, ncols = data.shape

        # DEBUG: Check data statistics
        print(f"Data shape: {data.shape}")
        print(f"Data min: {data.min()}, max: {data.max()}, mean: {data.mean()}")
        print(f"Non-zero values: {(data > 0).sum()} out of {data.size}")
        
        # make coordinate grids for the full image
        x = np.linspace(left, right, ncols)
        y = np.linspace(bottom, top, nrows)
        X, Y = np.meshgrid(x, y)

        # Mask zero/no-rain values
        data_masked = np.ma.masked_where(data <= 0, data)

        # plot full radar
        fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Set extent to the full image bounds
        ax.set_extent([left, right, bottom, top],
                    crs=ccrs.PlateCarree())
        
        # Use better colormap and set vmin/vmax explicitly
        cmap = plt.get_cmap("Blues")
        cmap.set_bad(color='white')  # Show masked values as white
        
        im = ax.pcolormesh(X, Y, data_masked, 
                        cmap=cmap, 
                        vmin=0.1,  # Start colormap at 0.1 mm/hr
                        vmax=data.max() if data.max() > 0 else 10)
        
        plt.colorbar(im, ax=ax, orientation='vertical', label="Rainfall (mm/hr)")
        plt.title(f"Radar frame (Full): {radar_row['time_sgt']}")
        
        # Add coastlines for reference
        ax.coastlines()
        
        plt.show()