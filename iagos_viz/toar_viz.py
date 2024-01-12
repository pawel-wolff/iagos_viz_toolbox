import numpy as np
import pandas as pd
import xarray as xr
import xarray_extras  # noq

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

import cartopy.feature as cf
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker


LON = 'lon'
LAT = 'lat'
LON_LB = 'lon_lb'
LON_UB = 'lon_ub'
LAT_LB = 'lat_lb'
LAT_UB = 'lat_ub'


def get_nrows_ncols(facet_coords, nrows, ncols, orient):
    # find out ncols / nrows, if necessary
    if nrows is None and ncols is None:
        if orient == 'column':
            ncols = 1
        else:
            nrows = 1
    if nrows is None:
        nrows = (len(facet_coords) + ncols - 1) // ncols
    elif ncols is None:
        ncols = (len(facet_coords) + nrows - 1) // nrows
    return nrows, ncols


def plot_data_cube(
        x, y,
        title,
        ncols=None, nrows=None,
        orient='column',
        subplots=None,
        color=(None, 'blue'),
        linestyle=(None, ('solid', ',')),
        xmin=None, xmax=None,
        ymin=None, ymax=None,
        figsize=(10.7, 7.3),
        linewidth=1,
):
    ds = xr.Dataset({'x': x, 'y': y})
    x, y = ds['x'], ds['y']
    if any(map(lambda _: _ is None, [xmin, xmax, ymin, ymax])):
        _ds = ds.where(x.notnull() & y.notnull(), drop=True)
        _x, _y = _ds['x'], _ds['y']
        if xmin is None:
            xmin = _x.min().item()
        if xmax is None:
            xmax = _x.max().item()
        if ymin is None:
            ymin = _y.min().item()
        if ymax is None:
            ymax = _y.max().item()

    if subplots is None:
        facet_coords = [None]
        nrows, ncols = 1, 1
    else:
        facet_coords = ds[subplots].values
        nrows, ncols = get_nrows_ncols(facet_coords, nrows, ncols, orient)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        layout='constrained',
        figsize=figsize,
    )
    axs = np.asanyarray(axs)
    if orient == 'column':
        axs = axs.T
    axs = axs.flatten()

    line_df = []
    for i, (facet_coord, ax) in enumerate(zip(facet_coords, axs)):
        if False and not (orient == 'column' and (i + 1) % nrows == 0 or orient == 'row' and i >= (nrows - 1) * ncols):
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,  # labels along the bottom edge are off
                left=False,
                labelleft=False,
            )
        if not (orient == 'column' and i < nrows or orient == 'row' and i % ncols == 0):
            ax.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,  # labels along the bottom edge are off
                left=False,
                labelleft=False,
            )

        if xmin is not None or xmax is not None:
            ax.set(xlim=(xmin, xmax))
        if ymin is not None or ymax is not None:
            ax.set(ylim=(ymin, ymax))

        if subplots is not None:
            ds_for_facet = ds.sel({subplots: facet_coord})
            ax.set_title(facet_coord)
        else:
            ds_for_facet = ds

        ax.grid(linewidth=0.5, color='grey', alpha=0.4)  # xlocs=[50], ylocs=[2,4,6,8,10], linewidth=0.2, color='gray', alpha=0.5)

        color_dim, colors = color
        if not isinstance(colors, (list, tuple)):
            colors = [colors]
        if color_dim is not None:
            dss_for_colors = [ds_for_facet.sel({color_dim: color_coord}) for color_coord in ds[color_dim].values]
        else:
            dss_for_colors = [ds_for_facet]

        for ds_for_color, c in zip(dss_for_colors, colors):
            linestyle_dim, linestyles_and_markers = linestyle
            if linestyle_dim is None:
                linestyles_and_markers = [linestyles_and_markers]
            if linestyle_dim is not None:
                dss = [ds_for_color.sel({linestyle_dim: linestyle_coord}) for linestyle_coord in ds[linestyle_dim].values]
            else:
                dss = [ds_for_color]

            for _ds, (ls, marker) in zip(dss, linestyles_and_markers):
                print(type(ax))
                _line, = ax.plot(_ds['x'].values, _ds['y'].values, linestyle=ls, linewidth=linewidth, marker=marker, color=c)
                _record = {}
                if subplots is not None:
                    _record[subplots] = facet_coord
                if color_dim is not None:
                    _record[color_dim] = ds_for_color[color_dim].item()
                if linestyle_dim is not None:
                    _record[linestyle_dim] = _ds[linestyle_dim].item()
                _record['_line'] = _line
                line_df.append(_record)

    plt.suptitle(title)
    return fig, pd.DataFrame.from_dict(line_df)


def _hide_axes_ticks_and_labels(ax, axes='xy'):
    assert axes in ['x', 'y', 'xy']
    for axis in ['x', 'y']:
        if axis in axes:
            ax.tick_params(
                axis=axis,  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,  # labels along the bottom edge are off
                left=False,
                labelleft=False,
            )


def plot_profiles(fig, ds, facet_dim, multiline_dim, specie, v_ranges, h_ranges, grid_spans):
    assert len(v_ranges) == len(h_ranges) == len(grid_spans) - 1
    grid_spans = np.concatenate([[0], np.asarray(grid_spans)])
    grid_spans_start, grid_spans_stop = np.cumsum(grid_spans)[:-1], np.cumsum(grid_spans)[1:]
    grid_spans_slice = [slice(start, stop) for start, stop in zip(grid_spans_start, grid_spans_stop)]

    facets = len(ds[facet_dim])
    ncols = (facets + 1) // 2
    gs = gridspec.GridSpec(2, ncols, hspace=0.3, figure=fig)
    legend_items = []

    for i, (facet_label, ds_facet) in enumerate(ds.xrx.iterate(facet_dim)):
        row = i // ncols
        col = i % ncols
        gs_facet = gridspec.GridSpecFromSubplotSpec(nrows=np.sum(grid_spans), ncols=1, subplot_spec=gs[row, col])
        axs = [
            fig.add_subplot(gs_facet[grid_span_slice, 0])
            for grid_span_slice in grid_spans_slice
        ]
        *axs, ax2 = axs  # last axis goes to ax2

        _hide_axes_ticks_and_labels(ax2, axes='y')
        if col > 0:
            for ax in axs:
                _hide_axes_ticks_and_labels(ax, axes='y')

        axs[0].set_title(facet_label, fontsize=8)
        for ax, v_range, h_range in zip(axs, v_ranges, h_ranges):
            h_min, h_max = h_range
            if h_min is None:
                h_min = ds['height_km'].min().item() - 0.5
            if h_max is None:
                h_max = ds['height_km'].max().item() + 0.5
            h_range = (h_min, h_max)

            ax.grid(linewidth=0.5, color='grey', alpha=0.4)
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            ax.set(xlim=v_range, ylim=h_range)

        for _, ds_singleline in ds_facet.xrx.iterate(multiline_dim):
            c = ds_singleline['color'].item()
            lm = ds_singleline['linemarker'].item()
            ls = ds_singleline['percentiles_linestyle'].item()
            for j, ax in enumerate(axs):
                _line, = ax.plot(
                    ds_singleline[f'{specie}_mean'],
                    ds_singleline['height_km'],
                    color=c,
                    marker=lm,
                    linewidth=1
                )
                if i == 0 and j == 0:
                    legend_items.append(_line)
                for stat in ['p5', 'p95']:
                    ax.plot(
                        ds_singleline[f'{specie}_{stat}'],
                        ds_singleline['height_km'],
                        color=c,
                        linestyle=ls,
                        linewidth=1
                    )

        ax2.tick_params(axis='x', which='both', labelsize=5)
        ax2.set(xlim=(0.9, max(ds[f'{specie}_flights'].max().values, 10)))
        if col == 0:
            ax2.set_ylabel('profiles', fontsize=6)
        ax2.barh(
            y=ds_facet[multiline_dim],
            width=ds_facet[f'{specie}_flights'],
            color=ds_facet['color'].values,
            log=True
        )
        ax2.invert_yaxis()

    fig.legend(legend_items, ds[multiline_dim].values, loc='outside lower center', ncols=len(legend_items))
    return fig


def plot_seasonal_cycles(
        fig, ds_by_specie, facet_dim, multiline_dim, x_dim, v_range_by_specie,
        single_legend=False,
        primary_linewidth=0.75, secondary_linewidth=0.75,
        margins=None
):
    if margins is None:
        margins = {}
    ds_any = list(ds_by_specie.values())[0]
    nfacets = len(ds_any[facet_dim])
    gs = gridspec.GridSpec(
        len(ds_by_specie),
        nfacets,
        figure=fig,
        #wspace=0.3,
        **margins
    )
    legend_items_by_specie = {}

    for row, (specie, ds) in enumerate(ds_by_specie.items()):
        flight_max = max(ds[f'{specie}_flights'].max().item(), 5)
        legend_items_by_specie[specie] = {}
        nlines = len(ds[multiline_dim])

        for col, (facet_label, ds_facet) in enumerate(ds.xrx.iterate(facet_dim)):
            gs_facet = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs[row, col], height_ratios=[0.6, 0.4])
            ax_lineplot = fig.add_subplot(gs_facet[0, 0])
            gs_barplots = gridspec.GridSpecFromSubplotSpec(nrows=nlines, ncols=1, subplot_spec=gs_facet[1, 0], hspace=0.5)
            axs_barplot = [
                fig.add_subplot(gs_barplots[i, 0], sharex=ax_lineplot)
                for i in range(nlines)
            ]

            # _hide_axes_ticks_and_labels(ax_lineplot, axes='y')
            # if col > 0:
                # for ax_barplot in axs_barplot:
                    # _hide_axes_ticks_and_labels(ax_barplot, axes='y')

            for i, ax_barplot in enumerate(axs_barplot):
                _hide_axes_ticks_and_labels(ax_barplot, axes='x')

            ax_lineplot.set_title(f'{specie}: {facet_label}', fontsize=8)

            ax_lineplot.grid(linewidth=0.5, color='grey', alpha=0.4)
            ax_lineplot.tick_params(axis='x', labelsize=6)
            ax_lineplot.tick_params(axis='y', labelsize=6)
            ax_lineplot.set(ylim=v_range_by_specie[specie][col])
            _x_labels = ds[x_dim].values
            if _x_labels.dtype == str:
                ax_lineplot.xaxis.set_major_formatter(lambda tl, pos: _x_labels[tl][0])
            elif _x_labels.dtype == int:
                pass

            for i, ax_barplot in enumerate(axs_barplot):
                ax_barplot.grid(axis='x', linewidth=0.5, color='grey', alpha=0.4)
                ax_barplot.tick_params(axis='x', labelsize=6)
                ax_barplot.tick_params(axis='y', labelsize=6)
                ax_barplot.set(ylim=(0, flight_max))

            for i, (_, ds_singleline) in enumerate(ds_facet.xrx.iterate(multiline_dim)):
                c = ds_singleline['color'].item()
                lm = ds_singleline['linemarker'].item()
                ls = ds_singleline['percentiles_linestyle'].item()

                _line, = ax_lineplot.plot(
                    ds_singleline[x_dim],
                    ds_singleline[f'{specie}_mean'],
                    color=c,
                    marker=lm,
                    linewidth=primary_linewidth,
                )
                if col == 0:
                    legend_items_by_specie[specie][_] = _line
                for stat in ['p5', 'p95']:
                    ax_lineplot.plot(
                        ds_singleline[x_dim],
                        ds_singleline[f'{specie}_{stat}'],
                        color=c,
                        linestyle=ls,
                        linewidth=secondary_linewidth,
                    )

                axs_barplot[i].bar(
                    x=ds_singleline[x_dim],
                    height=ds_singleline[f'{specie}_flights'],
                    color=c,
                )

    if single_legend:
        legend_loc = ['lower center']
        title = None
    else:
        legend_loc = ['lower left', 'lower right']
        title = f'{multiline_dim.capitalize()} for {specie} observations:'
    for specie, loc in zip(ds_by_specie, legend_loc):
        fig.legend(
            list(legend_items_by_specie[specie].values()),
            list(legend_items_by_specie[specie].keys()),
            loc=loc,
            fontsize=6,
            title=title,
            title_fontsize=6,
            alignment='left',
            ncols=len(legend_items_by_specie[specie])
        )
    return fig


def multifacet_plot(fig, ds, facet_dim, multiline_dim, x_dim, specie, v_range):
    facets = len(ds[facet_dim])
    ncols = (facets + 1) // 2
    gs = gridspec.GridSpec(2, ncols, hspace=0.3, figure=fig)
    legend_items = []

    for i, (facet_label, ds_facet) in enumerate(ds.xrx.iterate(facet_dim)):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs)
        if col > 0:
            _hide_axes_ticks_and_labels(ax, axes='y')

        ax.set_title(facet_label, fontsize=8)
        ax.grid(linewidth=0.5, color='grey', alpha=0.4)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.set(ylim=v_range)

        for _, ds_singleline in ds_facet.xrx.iterate(multiline_dim):
            c = ds_singleline['color'].item()
            lm = ds_singleline['linemarker'].item()
            ls = ds_singleline['percentiles_linestyle'].item()
            _line, = ax.plot(
                ds_singleline[f'{specie}_mean'],
                ds_singleline[x_dim],
                color=c,
                marker=lm,
                linewidth=1
            )
            if i == 0:
                legend_items.append(_line)
            for stat in ['p5', 'p95']:
                ax.plot(
                    ds_singleline[f'{specie}_{stat}'],
                    ds_singleline['height_km'],
                    color=c,
                    linestyle=ls,
                    linewidth=1
                )

        ax2.tick_params(axis='x', which='both', labelsize=5)
        ax2.set(xlim=(0.9, max(ds[f'{specie}_flights'].max().values, 10)))
        if col == 0:
            ax2.set_ylabel('profiles', fontsize=6)
        ax2.barh(
            y=ds_facet[multiline_dim],
            width=ds_facet[f'{specie}_flights'],
            color=ds_facet['color'].values,
            log=True
        )
        ax2.invert_yaxis()

    fig.legend(legend_items, ds[multiline_dim].values, loc='outside lower center', ncols=len(legend_items))
    return fig


def init_default_axis(ax, data, facet_dims):
    ax.set_title(' '.join(str(data[d].values) for d in facet_dims), fontsize=10)
    ax.gridlines(
        xlocs=np.linspace(-120, 120, 5),
        ylocs=[0],
        linewidth=0.2,
        color='gray',
        alpha=0.5,
    )
    ax.coastlines(resolution='110m', linewidth=0.2)
    ax.add_feature(cf.BORDERS, linewidth=0.1)
    # ax.set_facecolor((0.7, 0.7, 0.7))
    return ax


def set_default_xticks(ax, data):
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(axis='x', labelsize=6)
    return ax


def set_default_yticks(ax, data):
    ax.set_yticks(np.arange(-20, 21, 20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(axis='y', labelsize=6)
    return ax


def get_lon_lat_plots(
        da,
        vmin=None, vmax=None,
        cmap=None,
        norm=None,
        plot_type='colormesh',
        shading='flat',
        ncols=None, nrows=None,
        facet_dims=None,
        projection=None,
        set_extent=True,
        figsize=(10.7, 7.3),
        orient='column',
        title='default',
        title_font_size=14,
        init_axis='default',
        init_axis_kwargs=None,
        set_xticks='default',
        set_xticks_kwargs=None,
        set_yticks='default',
        set_yticks_kwargs=None,
        plotting_kwargs=None
):
    """
    See:
    https://stackoverflow.com/questions/30030328/correct-placement-of-colorbar-relative-to-geo-axes-cartopy
    Multi panel plots with single color bar and title
    https://kpegion.github.io/Pangeo-at-AOES/examples/multi-panel-cartopy.html
    :param da:
    :param vmin:
    :param vmax:
    :param cmap:
    :param norm: an instance of colors.Normalize, colors.LogNorm or colors.PowerNorm
    :param shading:
    :param ncols:
    :param nrows:
    :param facet_dims:
    :param projection:
    :param figsize:
    :param orient:
    :param title:
    :param title_font_size:
    :param init_axis:
    :param init_axis_kwargs:
    :param set_xticks:
    :param set_xticks_kwargs:
    :param set_yticks:
    :param set_yticks_kwargs:
    :return:
    """
    assert orient in ['column', 'row']
    assert plot_type in ['colormesh', 'contour']
    assert shading in ['flat', 'nearest', 'gouraud'] or shading is None and plot_type == 'contour'

    if projection is None:
        projection = ccrs.PlateCarree()
    if plotting_kwargs is None:
        plotting_kwargs = {}

    # setup facet_dim
    if facet_dims is None:
        facet_dims = [d for d in da.dims if d not in [LON, LAT]]
    elif not isinstance(facet_dims, (list, tuple)):
        facet_dims = list(facet_dims)
    if len(facet_dims) == 0:
        raise ValueError('for a single lon/lat plot, use another function')
    if len(facet_dims) > 1:
        _facet_dim = '_facet'
        _da = da.stack({_facet_dim: facet_dims})
    else:
        _facet_dim, = facet_dims
        _da = da

    _da = _da.xrx.make_coordinates_increasing([LON, LAT])

    if norm is None:
        norm = colors.Normalize
        _vmin = _da.min().item() if vmin is None else vmin
        _vmax = _da.max().item() if vmax is None else vmax
        norm = norm(vmin=_vmin, vmax=_vmax)

    if plot_type == 'colormesh' and shading == 'flat':
        # check if lon/lat cells are adjacent
        if not np.allclose(_da[LON_LB].values[1:], _da[LON_UB].values[:-1]):
            raise ValueError(f'{LON} cells are not adjacent')
        if not np.allclose(_da[LAT_LB].values[1:], _da[LAT_UB].values[:-1]):
            raise ValueError(f'{LAT} cells are not adjacent')

        # prepare mesh and extent
        lon_grid = np.concatenate((_da[LON_LB].values, [float(_da[LON_UB].max())]))
        lat_grid = np.concatenate((_da[LAT_LB].values, [float(_da[LAT_UB].max())]))
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        extent = None  # data extent applies
    elif plot_type == 'contour' or shading in ['nearest', 'gouraud']:
        lon_mesh, lat_mesh = np.meshgrid(_da[LON].values, _da[LAT].values)
        extent = float(_da[LON_LB].min()), float(_da[LON_UB].max()), float(_da[LAT_LB].min()), float(_da[LAT_UB].max())
    else:
        raise ValueError(f'Invalid parameters combination: plot_type={plot_type}, shading={shading}')

    nrows, ncols = get_nrows_ncols(da[_facet_dim], nrows, ncols, orient)

    # init fig and axes
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw={'projection': projection},
        layout='constrained',
        figsize=figsize
    )
    if nrows == 1 and ncols == 1:
        axs = np.asanyarray([axs])
    if orient == 'column':
        axs = axs.T
    axs = axs.flatten()

    facet_labels = _da[_facet_dim].values
    # delete empty subplots
    for ax in axs[len(facet_labels):]:
        fig.delaxes(ax)

    # generate figures for each facet
    for i, (ax, _facet_label) in enumerate(zip(axs, _da[_facet_dim].values)):
        data = _da.sel({_facet_dim: _facet_label})
        if set_extent and extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        if init_axis == 'default':
            init_default_axis(ax, data, facet_dims=facet_dims)
        elif init_axis is None:
            pass
        else:
            if init_axis_kwargs is None:
                init_axis_kwargs = {}
            init_axis(ax, data, **init_axis_kwargs)

        # define the xticks for longitude
        if orient == 'column' and (i + 1) % nrows == 0 or orient == 'row' and i >= (nrows - 1) * ncols:
            if set_xticks == 'default':
                set_default_xticks(ax, data)
            elif set_xticks is None:
                pass
            else:
                if set_xticks_kwargs is None:
                    set_xticks_kwargs = {}
                set_xticks(ax, data, **set_xticks_kwargs)

        # define the yticks for latitude
        if orient == 'column' and i < nrows or orient == 'row' and i % ncols == 0:
            if set_yticks == 'default':
                set_default_yticks(ax, data)
            elif set_yticks is None:
                pass
            else:
                if set_yticks_kwargs is None:
                    set_yticks_kwargs = {}
                set_yticks(ax, data, **set_yticks_kwargs)

        # if data has only NaN's and vmin or vmax is not given, matplotlib will crash when calc. min/max, so skip plotting in such case:
        #if vmin is not None and vmax is not None or data.notnull().any():

        if plot_type == 'colormesh':
            pcolormesh_kwargs = dict(
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                edgecolors='none',
                shading=shading,
                snap=True,
            )
            pcolormesh_kwargs.update(plotting_kwargs)
            _plot = ax.pcolormesh(
                lon_mesh, lat_mesh,
                data.transpose(LAT, LON).values,
                **pcolormesh_kwargs
            )
        elif plot_type == 'contour':
            contourf_kwargs = dict(
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
            )
            contourf_kwargs.update(plotting_kwargs)
            _plot = ax.contourf(
                lon_mesh, lat_mesh,
                data.transpose(LAT, LON).values,
                **contourf_kwargs
            )
        else:
            raise ValueError(plot_type)

    # Draw the colorbar
    if vmin is not None:
        colorbar_extend = 'both' if vmax is not None else 'min'
    else:
        colorbar_extend = 'max' if vmax is not None else 'neither'
    cbar = fig.colorbar(_plot, ax=axs, orientation='horizontal', location='bottom', extend=colorbar_extend, shrink=0.4)

    # Add a big title at the top
    if title == 'default':
        title = da.name
    if title:
        plt.suptitle(title, fontsize=title_font_size)

    return fig
