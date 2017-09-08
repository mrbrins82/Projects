import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import os, sys
#import get_locations

from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim
from matplotlib.colors import LinearSegmentedColormap

# load the job csv files and the csv with coordinates
ai_df = pd.read_csv('new_artificial_intelligence_jobs.csv')
ai_df['keyword'] = 'AI'
ds_df = pd.read_csv('new_data_scientist_jobs.csv')
ds_df['keyword'] = 'DS'
ml_df = pd.read_csv('new_machine_learning_jobs.csv')
ml_df['keyword'] = 'ML'
locations_df = pd.read_csv('locations.csv')


print 'ai_df.shape'
print 'ds_df.shape'
print 'ml_df.shape'
print ''
print ai_df.shape
print ds_df.shape
print ml_df.shape
print ''
print 'Total jobs: ',ai_df.shape[0] + ds_df.shape[0] + ml_df.shape[0]
print ''
# let's put all of the data frames together and drop duplicates
temp_df = pd.concat((ai_df, ds_df))
all_df = pd.concat((temp_df, ml_df))

df = all_df.drop_duplicates()
locations_df = locations_df.drop_duplicates()

df.location = df.location.replace(to_replace='Santa Clara Valley, CA', value='Santa Clara, CA')
locations_df.city = locations_df.city.replace(to_replace='Santa Clara Valley, CA', value='Santa Clara, CA')

print 'Dropping %d duplicates.'%(all_df.shape[0] - df.shape[0])
print 'df.shape'
print 'df.count()'
print ''
print df.shape
print df.count()
print ''

###############
#  COMPANIES  #
###############
n_top_companies = 25
print 'df.company.nunique()'
print 'df.company.value_counts()[:%d]'%n_top_companies
print df.company.nunique()
print df.company.value_counts()[:n_top_companies]
print ''

top_companies_df = pd.DataFrame(index=np.arange(n_top_companies), columns=['Company', 'Cum_AI', 'Cum_ML', 'Cum_DS'])


for (ii, (key, value)) in enumerate(df.company.value_counts()[:n_top_companies].iteritems()):

    temp_df = df[df.company == key]
    temp_ai_df = temp_df[temp_df.keyword == 'AI']
    ai_count = temp_ai_df.shape[0]
    temp_ml_df = temp_df[temp_df.keyword == 'ML']
    ml_count = temp_ml_df.shape[0]
    temp_ds_df = temp_df[temp_df.keyword == 'DS']
    ds_count = temp_ds_df.shape[0]

    top_companies_df.Company.iloc[ii] = key
    top_companies_df.Cum_AI.iloc[ii] = ai_count
    top_companies_df.Cum_ML.iloc[ii] = ai_count + ml_count
    top_companies_df.Cum_DS.iloc[ii] = ai_count + ml_count + ds_count
#    top_companies_df.Total.iloc[ii] = ai_count + ml_count + ds_count

sns.set_style(style='darkgrid')
f, ax = plt.subplots(figsize=(12, 6))

top_companies_df = top_companies_df.sort_values(by='Cum_DS', ascending=False)

sns.set_color_codes('muted')
sns.barplot(x="Cum_DS", y="Company", data=top_companies_df,
                    label="DS Jobs", color="b")
sns.barplot(x="Cum_ML", y="Company", data=top_companies_df,
                    label="ML Jobs", color="r")
sns.barplot(x="Cum_AI", y="Company", data=top_companies_df,
                    label="AI Jobs", color="purple")


# Add a legend and informative axis label
ax.legend(ncol=3, loc="lower right", frameon=True)
ax.set(xlim=(0, 80), ylabel="",
               xlabel="Total Number of Job Listings")
sns.despine(left=True, bottom=True)

## Draw a count plot to show the number of planets discovered each year
#g = sns.factorplot(x="company", y="count", data=df, #kind="count",
#                           palette="BuPu", size=6, aspect=1.5)
ax.set_yticklabels(labels=top_companies_df.Company, rotation=30)

plt.tight_layout()
plt.savefig('top_companies.png')
plt.clf()
os.system('open top_companies.png')
exit()
################
#  JOB TITLES  #
################
n_top_titles = 25
print 'df.jobtitle.nunique()'
print 'df.jobtitle.value_counts()[:%d]'%n_top_titles
print df.jobtitle.nunique()
print df.jobtitle.value_counts()[:n_top_titles]
print ''

top_titles = []
top_title_counts = []
for (key, value) in df.jobtitle.value_counts()[:n_top_titles].iteritems():
    top_titles.append(key)
    top_title_counts.append(value)

top_title_df = pd.DataFrame()
top_title_df['title'] = top_titles
top_title_df['count'] = top_title_counts


f, ax = plt.subplots(figsize=(12, 6))

sns.set_color_codes('muted')
sns.barplot(x="count", y="title", data=top_title_df,
                    label="Total Job Titles", color="b")

## Plot the crashes where alcohol was involved
#sns.set_color_codes("muted")
#sns.barplot(x="alcohol", y="abbrev", data=crashes,
#                    label="Alcohol-involved", color="b")

# Add a legend and informative axis label
#ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 525), ylabel="",
               xlabel="Total Number of Job Titles")
sns.despine(left=True, bottom=True)

## Draw a count plot to show the number of planets discovered each year
#g = sns.factorplot(x="company", y="count", data=df, #kind="count",
#                           palette="BuPu", size=6, aspect=1.5)
ax.set_yticklabels(labels=top_title_df.title, rotation=30)

plt.tight_layout()
plt.savefig('top_titles.png')
plt.clf()
os.system('open top_titles.png')
exit()



##################
print "apple_df = df[df.company == 'Apple']"
apple_df = df[df.company == 'Apple']
print ''
print 'apple_df.count()'
print apple_df.count()
print ''
print 'Apple locations:'
print apple_df.location.unique()
print ''
print "google_df = df[df.company == 'Google']"
google_df = df[df.company == 'Google']
print ''
print 'google_df.count()'
print google_df.count()
print ''
print 'Google locations:'
print google_df.location.unique()
print ''
print 'salaries_df = df[df.salary.notnull()]'
salaries_df = df[df.salary.notnull()]
print 'salaries_df'
print salaries_df.salary.unique()



exit()

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
plt.figure(figsize=(8,8))
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

cbar = plt.colorbar(orientation='horizontal', shrink=0.625, aspect=20, fraction=0.5,pad=0.02)
cbar.set_label('Number of Jobs',size=26)
#plt.clim([0,100])
    
    
# translucent blue scatter plot of jobs above histogram:    
x,y = m(list(lons), list(lats))
m.plot(x, y, 'o', markersize=8,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.33)
     
        
## http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap.drawmapscale
#m.drawmapscale(-119-6, 37-7.2, -119-6, 37-7.2, 500, barstyle='fancy', yoffset=20000)
        
        
## make image bigger:
plt.gcf().set_size_inches(20,20)

plt.savefig('heat_scatter_us.png')
plt.clf()
os.system('open heat_scatter_us.png')



#########################
# ZOOM IN ON WEST COAST #
#########################
m  = Basemap(projection='ortho',lon_0=-119,lat_0=37,resolution='l',
             llcrnrx=-1000*1000,llcrnry=-1000*1000,
             urcrnrx=+1150*1000,urcrnry=+1700*1000)#1150, 1700
m.drawcoastlines()
m.drawcountries()
m.drawstates()
    
# compute appropriate bins to chop up the data:
db = 1 # bin padding
lon_bins = np.linspace(min(lons)-db, max(lons)+db, 60+1) # 60 bins
lat_bins = np.linspace(min(lats)-db, max(lats)+db, 45+1) # 45 bins
    
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
             llcrnrx=-1250*1000,llcrnry=-1850*1000,
             urcrnrx=+800*1000,urcrnry=+800*1000)#1150, 1700
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# compute appropriate bins to chop up the data:
db = 1 # bin padding
lon_bins = np.linspace(min(lons)-db, max(lons)+db, 60+1) # 60 bins
lat_bins = np.linspace(min(lats)-db, max(lats)+db, 45+1) # 45 bins
    
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


#######################
# ZOOM IN ON NEW YORK #
#######################
m  = Basemap(projection='ortho',lon_0=-74,lat_0=41,resolution='h',
             llcrnrx=-350*1000,llcrnry=-350*1000,
             urcrnrx=+350*1000,urcrnry=+350*1000)#1150, 1700
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# compute appropriate bins to chop up the data:
db = 1 # bin padding
lon_bins = np.linspace(min(lons)-db, max(lons)+db, 225+1) # 225 bins
lat_bins = np.linspace(min(lats)-db, max(lats)+db, 150+1) # 150 bins
    
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

plt.savefig('heat_scatter_nyc_zoom.png')
plt.clf()
os.system('open heat_scatter_nyc_zoom.png')


#######################
# ZOOM IN ON SAN FRAN #
#######################
m  = Basemap(projection='ortho',lon_0=-122.42,lat_0=37.77,resolution='h',
             llcrnrx=-50*1000,llcrnry=-100*1000,
             urcrnrx=+100*1000,urcrnry=+50*1000)#1150, 1700
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# compute appropriate bins to chop up the data:
db = 1 # bin padding
lon_bins = np.linspace(min(lons)-db, max(lons)+db, 650+1) # 500 bins
lat_bins = np.linspace(min(lats)-db, max(lats)+db, 400+1) # 400 bins
    
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

plt.savefig('heat_scatter_sanfran_zoom.png')
plt.clf()
os.system('open heat_scatter_sanfran_zoom.png')

