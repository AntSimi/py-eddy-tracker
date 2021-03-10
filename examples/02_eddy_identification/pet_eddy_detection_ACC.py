"""
Eddy detection : Antartic circum polar
======================================

Script will detect eddies on adt field, and compute u,v with method add_uv(which could use, only if equator is avoid)

Two ones with filtering adt and another without

"""
from datetime import datetime

from matplotlib import pyplot as plt

from py_eddy_tracker import data
from py_eddy_tracker.dataset.grid import RegularGridDataset


def quad_axes(title):
    fig = plt.figure(figsize=(13, 8.5))
    fig.suptitle(title, weight="bold")
    axes = list()
    for position in (
        [0.05, 0.53, 0.44, 0.44],
        [0.53, 0.53, 0.44, 0.44],
        [0.05, 0.03, 0.44, 0.44],
        [0.53, 0.03, 0.44, 0.44],
    ):
        ax = fig.add_axes(position)
        ax.set_xlim(5, 45), ax.set_ylim(-60, -37)
        ax.set_aspect("equal"), ax.grid(True)
        axes.append(ax)
    return axes


# %%
# Load Input grid, ADT is used to detect eddies
margin = 30

kw_data = dict(
    filename=data.get_path("nrt_global_allsat_phy_l4_20190223_20190226.nc"),
    x_name="longitude",
    y_name="latitude",
    # Manual area subset
    indexs=dict(
        latitude=slice(100 - margin, 220 + margin),
        longitude=slice(0, 230 + margin),
    ),
)
g_raw = RegularGridDataset(**kw_data)
g_raw.add_uv("adt")
g = RegularGridDataset(**kw_data)
g.copy("adt", "adt_low")
g.bessel_high_filter("adt", 700)
g.bessel_low_filter("adt_low", 700)
g.add_uv("adt")

# %%
# Identification
# ^^^^^^^^^^^^^^
# Run the identification step with slices of 2 mm
date = datetime(2016, 5, 15)
kw_ident = dict(
    date=date, step=0.002, shape_error=70, sampling=30, uname="u", vname="v"
)
a, c = g.eddy_identification("adt", **kw_ident)
a_, c_ = g_raw.eddy_identification("adt", **kw_ident)


# %%
# Figures
# -------
axs = quad_axes("General properties field")
m = g_raw.display(axs[0], "adt", vmin=-1, vmax=1, cmap="RdBu_r")
axs[0].set_title("ADT(m)")
m = g.display(axs[1], "adt_low", vmin=-1, vmax=1, cmap="RdBu_r")
axs[1].set_title("ADT (m) large scale with cut at 700 km")
m = g.display(axs[2], "adt", vmin=-1, vmax=1, cmap="RdBu_r")
axs[2].set_title("ADT (m) high scale with cut at 700 km")
cb = plt.colorbar(
    m, cax=axs[0].figure.add_axes([0.03, 0.51, 0.94, 0.01]), orientation="horizontal"
)
cb.set_label("ADT(m)", labelpad=-2)

# %%
axs = quad_axes("")
axs[0].set_title("Without filter")
axs[0].set_ylabel("Contours used in eddies")
axs[1].set_title("With filter")
axs[2].set_ylabel("Closed contours but not used")
g_raw.contours.display(axs[0], lw=0.5, only_used=True)
g.contours.display(axs[1], lw=0.5, only_used=True)
g_raw.contours.display(axs[2], lw=0.5, only_unused=True)
g.contours.display(axs[3], lw=0.5, only_unused=True)

# %%
kw = dict(ref=-10, linewidth=0.75)
kw_a = dict(color="r", label="Anticyclonic ({nb_obs} eddies)")
kw_c = dict(color="b", label="Cyclonic ({nb_obs} eddies)")
kw_filled = dict(vmin=0, vmax=100, cmap="Spectral_r", lut=20, intern=True, factor=100)
axs = quad_axes("Comparison between two detection")
# Match with intern/inner contour
i_a, j_a, s_a = a_.match(a, intern=True, cmin=0.15)
i_c, j_c, s_c = c_.match(c, intern=True, cmin=0.15)

a_.index(i_a).filled(axs[0], s_a, **kw_filled)
a.index(j_a).filled(axs[1], s_a, **kw_filled)
c_.index(i_c).filled(axs[0], s_c, **kw_filled)
m = c.index(j_c).filled(axs[1], s_c, **kw_filled)

cb = plt.colorbar(
    m, cax=axs[0].figure.add_axes([0.03, 0.51, 0.94, 0.01]), orientation="horizontal"
)
cb.set_label("Similarity index", labelpad=-5)
a_.display(axs[0], **kw, **kw_a), c_.display(axs[0], **kw, **kw_c)
a.display(axs[1], **kw, **kw_a), c.display(axs[1], **kw, **kw_c)

axs[0].set_title("Without filter")
axs[0].set_ylabel("Detection")
axs[1].set_title("With filter")
axs[2].set_ylabel("Contours' rejection criteria")

g_raw.contours.display(axs[2], lw=0.5, only_unused=True, display_criterion=True)
g.contours.display(axs[3], lw=0.5, only_unused=True, display_criterion=True)

for ax in axs:
    ax.legend()
# %%
# Criteria for rejecting a contour :
#  0. Accepted (green)
#  1. Rejection for shape error (red)
#  2. Masked value within contour (blue)
#  3. Under or over the pixel limit bounds (black)
#  4. Amplitude criterion (yellow)

# %%
i_a, j_a = i_a[s_a >= 0.4], j_a[s_a >= 0.4]
i_c, j_c = i_c[s_c >= 0.4], j_c[s_c >= 0.4]
fig = plt.figure(figsize=(12, 12))
fig.suptitle(f"Scatter plot (A : {i_a.shape[0]}, C : {i_c.shape[0]} matches)")

for i, (label, field, factor, stop) in enumerate(
    (
        ("speed radius (km)", "radius_s", 0.001, 120),
        ("outter radius (km)", "radius_e", 0.001, 120),
        ("amplitude (cm)", "amplitude", 100, 25),
        ("speed max (cm/s)", "speed_average", 100, 25),
    )
):
    ax = fig.add_subplot(2, 2, i + 1, title=label)
    ax.set_xlabel("Without filter")
    ax.set_ylabel("With filter")

    ax.plot(
        a_[field][i_a] * factor,
        a[field][j_a] * factor,
        "r.",
        label="Anticyclonic",
    )
    ax.plot(
        c_[field][i_c] * factor,
        c[field][j_c] * factor,
        "b.",
        label="Cyclonic",
    )
    ax.set_aspect("equal"), ax.grid()
    ax.plot((0, 1000), (0, 1000), "g")
    ax.set_xlim(0, stop), ax.set_ylim(0, stop)
    ax.legend()
