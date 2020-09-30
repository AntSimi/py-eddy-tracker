"""
Collocating external data
==========================

Script will use py-eddy-tracker methods to upload external data (sea surface temperature, SST) in a common structure with altimetry.

Figures higlights the different steps.

"""

from matplotlib import pyplot as plt
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker import data
import cartopy.crs as ccrs
from datetime import datetime
from numpy import ma, meshgrid

datest = '20160707'

filename_alt = "../l4_cmems/dt_blacksea_allsat_phy_l4_"+datest+"_20200801.nc" 
lon_name_alt = 'longitude'
lat_name_alt = 'latitude'

filename_sst = "../SST/"+datest+"000000-GOS-L4_GHRSST-SSTfnd-OISST_HR_REP-BLK-v02.0-fv01.0.nc4" 
lon_name_sst = 'lon'
lat_name_sst = 'lat'
var_name_sst = 'analysed_sst'

extent = [27, 42, 40.5, 47]

# %%
# Functions to initiate figure axes
def start_axes(title, extent=extent, fig=None, sp=None):
    if fig is None:
        fig = plt.figure(figsize=(13, 5))
        ax = fig.add_axes([0.03, 0.03, 0.90, 0.94],projection=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(sp,projection=ccrs.PlateCarree())
            
    ax.set_extent(extent)
    ax.gridlines()
    ax.coastlines(resolution='50m')
    ax.set_title(title)
    return ax

def update_axes(ax, mappable=None, unit=''):
    ax.grid()
    if mappable:  
        plt.colorbar(mappable, cax=ax.figure.add_axes([0.95, 0.05, 0.01, 0.9],title=unit))

# %%
# Loading SLA and first display
# -----------------------------
g = RegularGridDataset(data.get_path(filename_alt), lon_name_alt,lat_name_alt)
ax = start_axes("SLA", extent=extent)
m = g.display(ax, "sla", vmin=0.05, vmax=0.35)
u,v = g.grid("ugosa").T,g.grid("vgosa").T
ax.quiver(g.x_c, g.y_c, u, v, scale=10)
update_axes(ax, m, unit='[m]')

# %%
# Loading SST and first display
# -----------------------------
t = RegularGridDataset(filename=data.get_path(filename_sst),
                       x_name=lon_name_sst,
                       y_name=lat_name_sst)

# The following now load the corresponding variables from the SST netcdf (it's not needed to load it beforehand, so not executed.)
# t.grid(var_name_sst)

# %% 
# We can now plot SST from `t`
ax = start_axes("SST title")
m = t.display(ax, 'analysed_sst', vmin=295, vmax=300)
update_axes(ax, m, unit='[°K]')

# %% 
# Including SST in the Altimetry grid
# -----------------------------------
# We can use `Grid` tools to interpolate SST on the altimetry grid

lons, lats = meshgrid(g.x_c, g.y_c)
shape = lats.shape

# flat grid before interp
lons, lats = lons.reshape(-1), lats.reshape(-1)

# interp and reshape
ti = t.interp('analysed_sst', lons, lats).reshape(shape).T
ti = ma.masked_invalid(ti)

# %% 
# and add it to `g`
g.add_grid('sst',ti)

# %%
ax = start_axes("SST")
m = g.display(ax, "sst", vmin=295, vmax=300)
u,v = g.grid("ugosa").T,g.grid("vgosa").T
ax.quiver(g.x_c, g.y_c, u, v, scale=10)
update_axes(ax, m, unit='[°K]')

# %%
# Now, with eddy contours, and displaying SST anomaly
# ! lazy patch since add_grid isn't currently completing g.variables_description
g.variables_description['sst'] = t.variables_description[var_name_sst]
g.copy("sst", "sst_high")
g.bessel_high_filter('sst_high',200)

# %% 
# Eddy detection
date = datetime.strptime(datest,'%Y%m%d')
a, c = g.eddy_identification("sla", "ugosa", "vgosa", date, 0.002)

# %%
kwargs_a = dict(lw=2, label="Anticyclonic", ref=-10, color="b")
kwargs_c = dict(lw=2, label="Cyclonic", ref=-10, color="r")
ax = start_axes("SST anomaly")
m = g.display(ax, "sst_high", vmin=-1, vmax=1)
ax.quiver(g.x_c, g.y_c, u, v, scale=20)
a.display(ax, **kwargs_a), c.display(ax, **kwargs_c)
update_axes(ax, m, unit='[°K]')

# %% 
# Example of post-processing
# --------------------------
# Get mean of sst anomaly_high in each internal contour
anom_a = a.interp_grid(g, "sst_high", method="mean", intern=True)
anom_c = c.interp_grid(g, "sst_high", method="mean", intern=True)

# %% 
# Are cyclonic (resp. anticyclonic) eddies generally associated with positive (resp. negative) SST anomaly ? 
fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0.03, 0.03, 0.90, 0.90])
ax.set_xlabel("SST anomaly")
ax.set_xlim([-1,1])
ax.set_title('Histograms of SST anomalies')
ax.hist(anom_a,5, alpha=0.5, label = 'Anticyclonic (mean:%s)'%(anom_a.mean()))
ax.hist(anom_c,5, alpha=0.5, label = 'Cyclonic (mean:%s)'%(anom_c.mean()))
ax.legend()

# %% 
# Not clearly so in that case ..
