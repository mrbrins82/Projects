import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import os, sys
# import get_locations # uncomment if new jobs have been added

from matplotlib.axes import Axes
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
from geopy.geocoders import Nominatim
from matplotlib.colors import LinearSegmentedColormap

# load the job csv files and the csv with coordinates
ai_df = pd.read_csv('new_artificial_intelligence_jobs.csv')
ds_df = pd.read_csv('new_data_scientist_jobs.csv')
ml_df = pd.read_csv('new_machine_learning_jobs.csv')
locations_df = pd.read_csv('locations.csv')

# let's put all of the data frames together and drop duplicates
temp_df = pd.concat((ai_df, ds_df))
all_df = pd.concat((temp_df, ml_df))

df = all_df.drop_duplicates()
locations_df = locations_df.drop_duplicates()

df.location = df.location.replace(to_replace='Santa Clara Valley, CA', value='Santa Clara, CA')
locations_df.city = locations_df.city.replace(to_replace='Santa Clara Valley, CA', value='Santa Clara, CA')

del(ai_df, ds_df, ml_df, temp_df, all_df) # don't need these anymore


# create longitude/latitude features in the jobs dataframe
df['longitude'] = 0.
df['latitude'] = 0.

# fill in the lon/lat features
no_coords = 0
n_jobs = df.shape[0]
# give all of the cities coordinates
for ii in xrange(n_jobs):
    sys.stdout.write('\r')
    sys.stdout.write("Pct. completed: %.2f"%(100 * (ii+1)/float(n_jobs)))
    sys.stdout.flush()

    city = df.location.iloc[ii]
    city_stem = city.partition('(')[0].rstrip()

    longitude = locations_df[locations_df.city == city_stem].longitude.max()
    latitude = locations_df[locations_df.city == city_stem].latitude.max()
    
    try:
       df.longitude.iloc[ii] = longitude
       df.latitude.iloc[ii] = latitude
    except:
        print 'Coordinates not found for %s'%city_stem
        no_coords += 1

print 'Jobs with no coordinates --> %d'%no_coords

###############################################################
# begin making the job histograms/scatter plots
lons = df.longitude.dropna()
lats = df.latitude.dropna()
    
    
# Use orthographic projection centered on California with corners
# defined by number of meters from center position:
m  = Basemap(projection='ortho',lon_0=-98,lat_0=42,resolution='l',
             llcrnrx=-3000*1000,llcrnry=-2000*1000,
             urcrnrx=+3000*1000,urcrnry=+1700*1000)#1150, 1700
m.drawcoastlines()
m.drawcountries()
m.drawstates()
     
    
# compute appropriate bins to chop up the data:
db = 1 # bin padding
lon_bins = np.linspace(min(lons)-db, max(lons)+db, 40+1) # 40 bins
lat_bins = np.linspace(min(lats)-db, max(lats)+db, 30+1) # 30 bins
    
density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    
# Turn the lon/lat of the bins into 2 dimensional arrays ready
# for conversion into projected coordinates
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    
# convert the bin mesh to map coordinates:
xs, ys = m(lon_bins_2d, lat_bins_2d) # will be plotted using pcolormesh
# ######################################################################
    
    
    
# define custom colormap, white -> nicered, #E6072A = RGB(0.9,0.03,0.16)
cdict = {'red':  ( (0.0,  1.0,  1.0),
                     (1.0,  0.9,  1.0) ),
         'green':( (0.0,  1.0,  1.0),
                     (1.0,  0.03, 0.0) ),
         'blue': ( (0.0,  1.0,  1.0),
                     (1.0,  0.16, 0.0) ) }
custom_map = LinearSegmentedColormap('custom_map', cdict)
plt.register_cmap(cmap=custom_map)
    
    
# add histogram squares and a corresponding colorbar to the map:
plt.pcolormesh(xs, ys, density, cmap="custom_map")

cbar = plt.colorbar(orientation='horizontal', shrink=0.625, aspect=20, fraction=0.2,pad=0.02)
cbar.set_label('Number of Jobs',size=22)
#plt.clim([0,100])
    
    
# translucent blue scatter plot of jobs above histogram:    
x,y = m(list(lons), list(lats))
m.plot(x, y, 'o', markersize=8,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.33)
     
        
## http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap.drawmapscale
#m.drawmapscale(-119-6, 37-7.2, -119-6, 37-7.2, 500, barstyle='fancy', yoffset=20000)
        
        
## make image bigger:
#plt.gcf().set_size_inches(15,15)

plt.savefig('heat_scatter.png')
plt.clf()
os.system('open heat_scatter.png')


###########################
# ZOOM IN ON SAN FRANISCO #
###########################
m  = Basemap(projection='ortho',lon_0=-119,lat_0=37,resolution='l',
             llcrnrx=-1000*1000,llcrnry=-1000*1000,
             urcrnrx=+1150*1000,urcrnry=+1700*1000)#1150, 1700
m.drawcoastlines()
m.drawcountries()
m.drawstates()
     
    
# compute appropriate bins to chop up the data:
db = 1 # bin padding
lon_bins = np.linspace(min(lons)-db, max(lons)+db, 50+1) # 50 bins
lat_bins = np.linspace(min(lats)-db, max(lats)+db, 40+1) # 40 bins
    
density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    
# Turn the lon/lat of the bins into 2 dimensional arrays ready
# for conversion into projected coordinates
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    
# convert the bin mesh to map coordinates:
xs, ys = m(lon_bins_2d, lat_bins_2d) # will be plotted using pcolormesh
# ######################################################################
    
    
    
# define custom colormap, white -> nicered, #E6072A = RGB(0.9,0.03,0.16)
cdict = {'red':  ( (0.0,  1.0,  1.0),
                     (1.0,  0.9,  1.0) ),
         'green':( (0.0,  1.0,  1.0),
                     (1.0,  0.03, 0.0) ),
         'blue': ( (0.0,  1.0,  1.0),
                     (1.0,  0.16, 0.0) ) }
custom_map = LinearSegmentedColormap('custom_map', cdict)
plt.register_cmap(cmap=custom_map)
    
    
# add histogram squares and a corresponding colorbar to the map:
plt.pcolormesh(xs, ys, density, cmap="custom_map")

cbar = plt.colorbar(orientation='horizontal', shrink=0.625, aspect=20, fraction=0.2,pad=0.02)
cbar.set_label('Number of Jobs',size=22)
#plt.clim([0,100])
    
    
# translucent blue scatter plot of jobs above histogram:    
x,y = m(list(lons), list(lats))
m.plot(x, y, 'o', markersize=8,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.33)
     
        
## http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap.drawmapscale
#m.drawmapscale(-119-6, 37-7.2, -119-6, 37-7.2, 500, barstyle='fancy', yoffset=20000)
        
        
# make image bigger:
plt.gcf().set_size_inches(15,15)

plt.savefig('heat_scatter_westcoast_zoom.png')
plt.clf()
os.system('open heat_scatter_westcoast_zoom.png')

#########################
# ZOOM IN ON EAST COAST #
#########################
m  = Basemap(projection='ortho',lon_0=-73,lat_0=42,resolution='l',
             llcrnrx=-1000*1000,llcrnry=-1000*1000,
             urcrnrx=+1150*1000,urcrnry=+1500*1000)#1150, 1700
m.drawcoastlines()
m.drawcountries()
m.drawstates()
    
# compute appropriate bins to chop up the data:
db = 1 # bin padding
lon_bins = np.linspace(min(lons)-db, max(lons)+db, 40+1) # 40 bins
lat_bins = np.linspace(min(lats)-db, max(lats)+db, 30+1) # 30 bins
    
density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    
# Turn the lon/lat of the bins into 2 dimensional arrays ready
# for conversion into projected coordinates
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    
# convert the bin mesh to map coordinates:
xs, ys = m(lon_bins_2d, lat_bins_2d) # will be plotted using pcolormesh
# ######################################################################
    
    
    
# define custom colormap, white -> nicered, #E6072A = RGB(0.9,0.03,0.16)
cdict = {'red':  ( (0.0,  1.0,  1.0),
                     (1.0,  0.9,  1.0) ),
         'green':( (0.0,  1.0,  1.0),
                     (1.0,  0.03, 0.0) ),
         'blue': ( (0.0,  1.0,  1.0),
                     (1.0,  0.16, 0.0) ) }
custom_map = LinearSegmentedColormap('custom_map', cdict)
plt.register_cmap(cmap=custom_map)
    
    
# add histogram squares and a corresponding colorbar to the map:
plt.pcolormesh(xs, ys, density, cmap="custom_map")

cbar = plt.colorbar(orientation='horizontal', shrink=0.625, aspect=20, fraction=0.2,pad=0.02)
cbar.set_label('Number of Jobs',size=22)
#plt.clim([0,100])
    
    
# translucent blue scatter plot of jobs above histogram:    
x,y = m(list(lons), list(lats))
m.plot(x, y, 'o', markersize=8,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.33)
     
        
## http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap.drawmapscale
#m.drawmapscale(-119-6, 37-7.2, -119-6, 37-7.2, 500, barstyle='fancy', yoffset=20000)
        
        
# make image bigger:
plt.gcf().set_size_inches(12,8)

plt.savefig('heat_scatter_eastcoast_zoom.png')
plt.clf()
os.system('open heat_scatter_eastcoast_zoom.png')

