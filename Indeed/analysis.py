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

del(ai_df, ds_df, ml_df, temp_df, all_df)

print 'df.count()'
print df.count()
print ''


df['longitude'] = 0.
df['latitude'] = 0.

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




# load earthquake epicenters:
# http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/1.0_week.csv
#lats, lons = [], []
lons = df.longitude.dropna()
lats = df.latitude.dropna()

#with open('earthquake_data.csv') as f:
#    reader = csv.reader(f)
#        next(reader) # Ignore the header row.
#            for row in reader:
#                    lat = float(row[1])
#            lon = float(row[2])
#            # filter lat,lons to (approximate) map view:
#            if -130 <= lon <= -100 and 25 <= lat <= 55:
#                        lats.append( lat )
#                lons.append( lon )
    
    
# Use orthographic projection centered on California with corners
# defined by number of meters from center position:
m  = Basemap(projection='ortho',lon_0=-98,lat_0=42,resolution='l',
             llcrnrx=-3000*1000,llcrnry=-2000*1000,
             urcrnrx=+3000*1000,urcrnry=+1700*1000)#1150, 1700
m.drawcoastlines()
m.drawcountries()
m.drawstates()
     
    
        
# ######################################################################
# bin the epicenters (adapted from 
# http://stackoverflow.com/questions/11507575/basemap-and-density-plots)
    
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
cbar.set_label('Number of Jobs',size=18)
#plt.clim([0,100])
    
    
# translucent blue scatter plot of epicenters above histogram:    
x,y = m(list(lons), list(lats))
m.plot(x, y, 'o', markersize=8,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.33)
     
        
# http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap.drawmapscale
m.drawmapscale(-119-6, 37-7.2, -119-6, 37-7.2, 500, barstyle='fancy', yoffset=20000)
        
        
# make image bigger:
plt.gcf().set_size_inches(15,15)

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
     
    
        
# ######################################################################
# bin the epicenters (adapted from 
# http://stackoverflow.com/questions/11507575/basemap-and-density-plots)
    
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
cbar.set_label('Number of Jobs',size=18)
#plt.clim([0,100])
    
    
# translucent blue scatter plot of epicenters above histogram:    
x,y = m(list(lons), list(lats))
m.plot(x, y, 'o', markersize=8,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.33)
     
        
# http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap.drawmapscale
m.drawmapscale(-119-6, 37-7.2, -119-6, 37-7.2, 500, barstyle='fancy', yoffset=20000)
        
        
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
     
    
        
# ######################################################################
# bin the epicenters (adapted from 
# http://stackoverflow.com/questions/11507575/basemap-and-density-plots)
    
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
cbar.set_label('Number of Jobs',size=18)
#plt.clim([0,100])
    
    
# translucent blue scatter plot of epicenters above histogram:    
x,y = m(list(lons), list(lats))
m.plot(x, y, 'o', markersize=8,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.33)
     
        
# http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap.drawmapscale
m.drawmapscale(-119-6, 37-7.2, -119-6, 37-7.2, 500, barstyle='fancy', yoffset=20000)
        
        
# make image bigger:
plt.gcf().set_size_inches(15,15)

plt.savefig('heat_scatter_eastcoast_zoom.png')
plt.clf()
os.system('open heat_scatter_eastcoast_zoom.png')


exit()
#####################################
# Make a "heatmap" for job locations
#####################################
plot_lon = df.longitude.dropna()
plot_lat = df.latitude.dropna()

Map = Basemap(llcrnrlon=-119, llcrnrlat=20, urcrnrlon=-64, urcrnrlat=49,
              resolution='i', projection='lcc', lat_1=32, lat_2=45, lon_0=-95)
##              resolution='i', projection='cass', lat_0=35.5, lon_0=-91.5)

#Map = Basemap(projection='merc', resolution='h',
#              llcrnrlon=-128, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=52)

##############################################################################

# makes a scatterplot with marker sizes and transparencies related to
# number of jobs at the location
for ii in xrange(n_jobs):
        
#    print city, count
#    city_stem = city.partition('(')[0].rstrip()
    try:
#        loc = geolocator.geocode(city_stem)
#        print city_stem
#        print loc.longitude, loc.latitude

        longitude = df.longitude.iloc[ii]
        latitude = df.latitude.iloc[ii]

        x, y = Map(longitude, latitude)
        Map.plot(x, y, marker='o', color='Red', alpha=0.1, markersize=5)
    except:
        print 'There was a problem'

#Map.plot(plot_lon, plot_lat, marker='o', color='red', alpha=0.3)

Map.drawcoastlines()
Map.drawcountries()
Map.drawstates()
plt.savefig('job_scatter.png')
plt.clf()
os.system('open job_scatter.png')
exit()


######################################################################
# make a 2d histogram over the US
# remove points outside projection limb.
"""
fig = plt.figure()#figsize=(8,5))
ax = fig.add_subplot(111)
bins=50
#bincount, xedges, yedges = np.histogram2d(plot_lon, plot_lat, bins=bins)
#mask = bincount == 0
## reset zero values to one to avoid divide-by-zero
#bincount = np.where(bincount == 0, 1, bincount)
#H, xedges, yedges = np.histogram2d(plot_lon, plot_lat, bins=bins)
#H = np.ma.masked_where(mask, H/bincount)

plt.hist2d(plot_lon, plot_lat, bins=80)
plt.xlim(xmin=plot_lon.min() - 1, xmax=plot_lon.max() + 1)
plt.ylim(ymin=plot_lat.min() - 1, ymax=plot_lat.max() + 1)
## set color of masked values to axes background (hexbin does this by default)
#palette = plt.cm.jet
#palette.set_bad(ax.get_axis_bgcolor(), 1.0)
#CS = Map.pcolormesh(xedges,yedges,H.T,shading='flat',cmap=palette)
# draw coastlines, lat/lon lines.
Map.drawcoastlines()
Map.drawstates()
Map.colorbar(location="bottom",label="# jobs") # draw colorbar
#plt.title('histogram2d', fontsize=20)

#plt.gcf().set_size_inches(18,10)
plt.show()
plt.clf()
exit()
"""

Map.scatter(plot_lon, plot_lat)
plt.xlim(xmin=-130, xmax=-60)
plt.ylim(ymin=20, ymax=50)
plt.colorbar()
plt.savefig('simple_scatter.png')
plt.clf()
os.system('open simple_scatter.png')
exit()

###############################################################
lons = np.arange(-135, -60, 1)
print 'lons: ',lons
lats = np.arange(20, 55, 1)
print 'lats: ',lats

data = np.indices((lats.shape[0], lons.shape[0]))
print 'data: ',data
data = data[0] + data[1]
print 'data: ',data
print 'data.shape: ',data.shape

data_interp, x, y = Map.transform_scalar(data, lons, lats, 100, 50, returnxy=True, masked=True)

Map.pcolormesh(x, y, data_interp, cmap='summer')
#Map.bluemarble()
##Fill the globe with a blue color 
#Map.drawmapboundary(fill_color='aqua')
##Fill the continents with the land color
#Map.fillcontinents(color='coral',lake_color='aqua')
Map.drawcountries()
Map.drawstates(color='0.5')
Map.drawcoastlines()


#temp = [['San Francisco',100],
#        ['Pittsburgh',69],
#        ['Pittsburgh, PA',25],
#        ['Cleveland, OH', 9]]

#geolocator = Nominatim()
#for (city,count) in cities.iteritems():
#    print city,count
#    if count > 1:
#        try:
#            loc = geolocator.geocode(city)
#            x, y = Map(loc.longitude, loc.latitude)
#            Map.plot(x, y, marker='o', color='Red', alpha=1/np.sqrt(count), markersize=3*np.sqrt(count))
#        except:
#            print 'Could not geolocate for --> %s'%city


plt.savefig('jobmap.png')
plt.clf()

os.system('open jobmap.png')
