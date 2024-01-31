#!/home/wolp/miniconda3/envs/iagos-viz/bin/python

# This script is based on the notebook /home/wolp/projects/iagos/jupyter-scripts/iagos-footprints.ipynb

import pathlib
import numpy as np
import pandas as pd
import xarray as xr
import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.path as mpath
import cartopy.crs as ccrs
from shapely.geometry.linestring import LineString

import fpout
from common import longitude
from iagos_viz import toar_viz


def releases_trajectory(ds):
    lons = longitude.geodesic_longitude_midpoint(
        ds['release_lon'].isel({'pointspec': slice(0, -1)}).values,
        ds['release_lon'].isel({'pointspec': slice(1, None)}).values
    )
    lons = np.concatenate(([ds['release_lon'].isel({'pointspec': 0})], lons, [ds['release_lon'].isel({'pointspec': -1})]))
    lats = (ds['release_lat'].isel({'pointspec': slice(0, -1)}).values + ds['release_lat'].isel({'pointspec': slice(1, None)}).values) / 2
    lats = np.concatenate(([ds['release_lat'].isel({'pointspec': 0})], lats, [ds['release_lat'].isel({'pointspec': -1})]))
    return ds.assign_coords({
        'release_lon_begin': ('pointspec', lons[:-1]),
        'release_lat_begin': ('pointspec', lats[:-1]),
        'release_lon_end': ('pointspec', lons[1:]),
        'release_lat_end': ('pointspec', lats[1:]),
    })


def get_agg_residencetime_and_traj(ds, t0=None, t1=None, days=None):
    if t0 is not None:
        t0 = np.datetime64(t0)
        ds = ds.where(ds['release_time'] >= t0, drop=True)
    if t1 is not None:
        t1 = np.datetime64(t1)
        ds = ds.where(ds['release_time'] <= t1, drop=True)

    ds2 = ds[['res_time']].sum('height').mean('pointspec').squeeze()
    ds3 = xr.merge((ds2, ds[['release_lon']]))

    ds3 = ds3.rename({'longitude': 'lon', 'latitude': 'lat'})
    ds3 = ds3.assign_coords({
        'lon_lb': ds3['lon'] - 0.5,
        'lon_ub': ds3['lon'] + 0.5,
        'lat_lb': ds3['lat'] - 0.5,
        'lat_ub': ds3['lat'] + 0.5,
    })
    if days is not None and days >= 1:
        ds3 = ds3.isel({'time': slice(-days, None)})
    da = ds3['res_time'].sum('time') / ds3['area'] * 1e6
    da = da.compute()
    da = da.where(da > 0)

    traj = list(zip(
        np.concatenate((ds3['release_lon_begin'].values, [ds3['release_lon_end'].values[-1]])),
        np.concatenate((ds3['release_lat_end'].values, [ds3['release_lat_end'].values[-1]])),
    ))

    return da, traj


def get_footprint_fig(
        da, traj,
        title,
        plot_type,
        cmap,
        vmin, vmax, norm,
        fig_width, fig_height,
        projection, projection_kwargs,
        cmap_levels, cmap_extend,
        extent, cutout_circle,
        lon_grids, lat_grids,
        show_releases, releases_color,
        plotting_kwargs,
):
    if norm == 'linear':
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm == 'sqrt':
        norm = colors.PowerNorm(0.5, vmin=vmin, vmax=vmax)
    elif norm == 'log':
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        raise ValueError(norm)

    try:
        projection = getattr(ccrs, projection)
    except AttributeError:
        raise ValueError(f'Unknown projection: {projection}')

    if plot_type == 'contour':
        _plotting_kwargs = dict(
            levels=cmap_levels,
            extend=cmap_extend
        )
    else:
        _plotting_kwargs = {}
    if plotting_kwargs is not None:
        _plotting_kwargs.update(plotting_kwargs)

    fig = toar_viz.get_lon_lat_plots(
        da.where(da > 0).expand_dims({'_': [' ']}),
        plot_type=plot_type,
        colorbar_extend=cmap_extend,
        cmap=cmap,
        norm=norm,
        projection=projection(**projection_kwargs),
        extent=extent,
        figsize=(fig_width, fig_height),
        set_xticks=None,
        set_yticks=None,
        title=title,
        plotting_kwargs=_plotting_kwargs
    )

    ax, colorscale_ax = fig.get_axes()
    colorscale_ax.set_title('residence time (s km**-2)')

    ax.gridlines(draw_labels=True, xlocs=lon_grids, ylocs=lat_grids)

    if cutout_circle:
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    if show_releases:
        polyline = LineString(traj)
        ax.add_geometries([polyline], crs=ccrs.PlateCarree(), edgecolor=releases_color, facecolor='none',
                          linewidth=3)  # facecolor='b', edgecolor='red', alpha=0.8)

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='footprint',
        description='Produces a footprint plot',
        epilog='Usage example: ./footprint.py -i "2023070500501606" -o footprint_2023070500501606_bis.png --tmin \'2023-07-05T01:54\' --tmax \'2023-07-05T02:18\' --days 12 --plot_type colormesh --projection NorthPolarStereo --extent "[-180, 180, 30, 90]" --cutout_circle --lon_grids "[-180, -120, -60, 0, 60, 120]" --lat_grids "[0, 10, 20, 30, 40, 50, 60, 70, 80]" --show_releases --savefig_kwargs \'{"dpi": 200}\''
    )
    parser.add_argument('-i', '--flight', type=str, help='flight id')
    parser.add_argument('-d', '--fpdir', default='/o3p/iagos/flexpart/V9.2',
                        help='the directory with FLEXPART outputs; the footprint data is supposed to be in <dir>/<year-month>/<flight_id>/grid_time.nc; the default is /o3p/iagos/flexpart/V9.2')
    parser.add_argument('-f', '--fpfile', help='path to grid_time.nc; if provided, flight and fpdir is ignored')
    parser.add_argument('-o', '--output', required=True, help='output filename; extension determines a format')
    parser.add_argument('--tmin', help='if given, filter releases with release time >= tmin')
    parser.add_argument('--tmax', help='if given, filter releases with release time <= tmax')
    parser.add_argument('--days', type=int, help='number of days (backward from release) along which integrate the residence time')
    parser.add_argument('--plot_type', choices=['contour', 'colormesh'], default='contour')
    parser.add_argument('--cmap', default='YlOrBr')
    parser.add_argument('--vmin', type=float)
    parser.add_argument('--vmax', type=float)
    parser.add_argument('--norm', choices=['linear', 'sqrt', 'log'], default='log')
    parser.add_argument('--fig_title')
    parser.add_argument('--fig_width', type=float, default=6)
    parser.add_argument('--fig_height', type=float, default=8)
    parser.add_argument('--projection', default='PlateCarree', help='cartopy CRS projection')
    parser.add_argument('--projection_kwargs', type=json.loads, default='{}', help='cartopy CRS projection arguments as JSON')
    parser.add_argument('--cmap_levels', type=json.loads, default='[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1]',
                        help='a list of contour levels as JSON; used only when plot_type=\"contour\"')
    parser.add_argument('--cmap_extend', choices=['neither', 'max', 'min', 'both'], default='both')
    parser.add_argument('--extent', type=json.loads, help='geographical extent as JSON; can be \"auto\" or [lon_min, lon_max, lat_min, lat_max]')
    parser.add_argument('--cutout_circle', action='store_true'),
    parser.add_argument('--lon_grids', type=json.loads, help='a list with lon coordinates of grid lines as JSON'),
    parser.add_argument('--lat_grids', type=json.loads, help='a list with lat coordinates of grid lines as JSON'),
    parser.add_argument('--show_releases', action='store_true'),
    parser.add_argument('--releases_color', default='xkcd:blue'),
    parser.add_argument('--plotting_kwargs', type=json.loads, help='extra arguments passed to a plotting function')
    parser.add_argument('--savefig_kwargs', type=json.loads, help='arguments passed to savefig function')
    args = parser.parse_args()

    if args.fpfile is None:
        fpdir = pathlib.PurePath(args.fpdir)
        flight_id = args.flight
        fpfile = fpdir / flight_id[:6] / flight_id / 'grid_time.nc'
    else:
        flight_id = '???'
        fpfile = args.fpfile

    ds = fpout.open_dataset(fpfile, max_chunk_size=1e8)
    ds = releases_trajectory(ds)
    res_time, traj = get_agg_residencetime_and_traj(ds, args.tmin, args.tmax, args.days)

    fig_title = args.fig_title
    if fig_title is None:
        fig_title = f'Footprint for {flight_id}\nt0={args.tmin}, t1={args.tmax}'

    try:
        fig = get_footprint_fig(
            res_time, traj,
            title=fig_title,
            plot_type=args.plot_type,
            cmap=args.cmap,
            vmin=args.vmin, vmax=args.vmax, norm=args.norm,
            fig_width=args.fig_width, fig_height=args.fig_height,
            projection=args.projection, projection_kwargs=args.projection_kwargs,
            cmap_levels=args.cmap_levels, cmap_extend=args.cmap_extend,
            extent=args.extent, cutout_circle=args.cutout_circle,
            lon_grids=args.lon_grids, lat_grids=args.lat_grids,
            show_releases=args.show_releases, releases_color=args.releases_color,
            plotting_kwargs=args.plotting_kwargs,
        )
        savefig_kwargs = args.savefig_kwargs
        if savefig_kwargs is None:
            savefig_kwargs = {}
        plt.savefig(args.output, **savefig_kwargs)
    finally:
        plt.close()
