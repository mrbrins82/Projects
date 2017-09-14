import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import os, sys

from sklearn.feature_extraction.text import CountVectorizer

"""
get_locations is my own script that get coordinates from 
the city locations in the job csv files. Uncomment the 
'import get_locations' line if new jobs/locations have
been added.
"""
#import get_locations

from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap

# load the job csv files and the csv with coordinates
ai_df = pd.read_csv('jobs_artificial_intelligence.csv')
ds_df = pd.read_csv('jobs_data_scientist.csv')
ml_df = pd.read_csv('jobs_machine_learning.csv')
locations_df = pd.read_csv('locations.csv')

# tag each data frame by it's job keyword search
ai_df['keyword'] = 'AI'
ds_df['keyword'] = 'DS'
ml_df['keyword'] = 'ML'

# put all of the data frames together
all_df = pd.concat((ai_df, ds_df, ml_df))

# see how many total jobs we have
print 'Total jobs: ', all_df.shape[0]

# drop duplicates
df = all_df.drop_duplicates()

# see how many duplicates we have
print 'Duplicate jobs: ', all_df.shape[0] - df.shape[0]
print 'Total unique jobs: ', df.shape[0]

# replace Santa Clara Valley with Santa Clara since geolocator can't find coords for it
df.location = df.location.replace(to_replace='Santa Clara Valley, CA', value='Santa Clara, CA')
locations_df.city = locations_df.city.replace(to_replace='Santa Clara Valley, CA', value='Santa Clara, CA')

#####################################
#  PART 1 - WHOS HIRING & FOR WHAT  #
#####################################

#
# COMPANIES
#

n_top_companies = 25

# create a new data frame for top 25 job listing companies
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
ax.set_yticklabels(labels=top_companies_df.Company, rotation=30)

plt.tight_layout()
plt.savefig('top_companies.png')
plt.clf()
os.system('open top_companies.png')


# Make dist plot for company size
sns.set_style(style='darkgrid')
companysize_bins = int(df.companysize.max() / 50.)
sns.distplot(df.companysize.dropna(), bins=companysize_bins, norm_hist=False, color='blue')
plt.xlabel('Company Size', size=18)
plt.savefig('companysize_dist_plot.png')
plt.clf()
os.system('open companysize_dist_plot.png')

# Make dist plot for companyrating
sns.set_style(style='darkgrid')
sns.distplot(df.companyrating.dropna(), bins=30, norm_hist=False, color='blue')
plt.xlabel('Company Rating', size=18)
plt.savefig('companyrating_dist_plot.png')
plt.clf()
os.system('open companyrating_dist_plot.png')

# Make hex plot for companysize vs companyrating
temp_df = df[df.companysize.notnull()]
temp_df = temp_df[temp_df.companyrating.notnull()]
print temp_df.shape
sns.set_style(style='ticks')
sns.jointplot(temp_df.companysize, temp_df.companyrating, kind="hex", stat_func=None, joint_kws={'gridsize':10}, color="#4CB391")

plt.tight_layout()
plt.savefig('size_rating_hex.png')
plt.clf()
os.system('open size_rating_hex.png')


"""
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(temp_df.companysize, temp_df.companyrating, cmap=cmap, n_levels=60, shade=True);
plt.savefig('size_rating_heatmap.png')
plt.clf()
os.system('open size_rating_heatmap.png')
"""


#
#  JOB TITLES  
#

n_top_titles = 25

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
ax.set(xlim=(0, 525), ylabel="",
               xlabel="Total Number of Job Titles")
sns.despine(left=True, bottom=True)
ax.set_yticklabels(labels=top_title_df.title, rotation=30)

plt.tight_layout()
plt.savefig('top_titles.png')
plt.clf()
os.system('open top_titles.png')


#
# JOB SUMMARIES
#

vectorizer = CountVectorizer(stop_words='english')
word_array = vectorizer.fit_transform(df.summary.dropna()).todense()

word_counts = {} # create dict that has <word> as the key, and <count> as value
for (word, index) in vectorizer.vocabulary_.iteritems():
    word_freq = int(sum(word_array[:,index]))
    word_counts[word] = word_freq

min_word_count = 50
x, y = [], []
garbage_words = ['looking', 'using', 'role', 'seeking', 'join', 'working', 'work', 'areas', 'use', 'including'] # list of irrelevant throw away words
for (word, count) in word_counts.iteritems():
    if ((count >= min_word_count) and not (word in garbage_words)):
        x.append(count)
        y.append(word)

barplot_df = pd.DataFrame()
barplot_df['word'] = pd.Series(y)
barplot_df['count'] = pd.Series(x)

barplot_df = barplot_df.sort_values('count', ascending=False)
print barplot_df

sns.set(style='darkgrid')
f, ax = plt.subplots(figsize=(15,30))

sns.set_color_codes('muted')
sns.barplot(x="count", y="word", data=barplot_df, color='b')
ax.set_xlabel(r'$Count$', size=28)
ax.set_ylabel(r'$Word$', size=28)
ax.tick_params(axis='both', labelsize=18)
#ax.set_yticks(fontsize=14)
plt.tight_layout()
plt.savefig('summary_word_count.png')
plt.clf()
os.system('open summary_word_count.png')



############################
#  PART 2 - JOB LOCATIONS  #
############################

#
# US HEAT/SCATTER MAP
#

# create longitude/latitude features in the jobs dataframe
df['longitude'] = 0.
df['latitude'] = 0.

# fill in the lon/lat features
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


# begin making the job histograms/scatter plots
lons = df.longitude.dropna()
lats = df.latitude.dropna()
    

plt.figure(figsize=(8,8))
m  = Basemap(projection='ortho',lon_0=-98,lat_0=42,resolution='l',
             llcrnrx=-3000*1000,llcrnry=-2000*1000,
             urcrnrx=+3000*1000,urcrnry=+1700*1000)
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


#
# SAN FRANCISCO ZOOM HEAT/SCATTER
#

m  = Basemap(projection='ortho',lon_0=-122.42,lat_0=37.77,resolution='h',
             llcrnrx=-50*1000,llcrnry=-100*1000,
             urcrnrx=+100*1000,urcrnry=+50*1000)
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# compute appropriate bins to chop up the data:
db = 1 # bin padding
lon_bins = np.linspace(min(lons)-db, max(lons)+db, 650+1) 
lat_bins = np.linspace(min(lats)-db, max(lats)+db, 400+1)
    
density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    
# Turn the lon/lat of the bins into 2 dimensional arrays ready
# for conversion into projected coordinates
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    
# convert the bin mesh to map coordinates:
xs, ys = m(lon_bins_2d, lat_bins_2d) # will be plotted using pcolormesh
    
    
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


#
# NYC ZOOM HEAT/SCATTER
#

m  = Basemap(projection='ortho',lon_0=-74,lat_0=41,resolution='h',
             llcrnrx=-350*1000,llcrnry=-350*1000,
             urcrnrx=+350*1000,urcrnry=+350*1000)
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# compute appropriate bins to chop up the data:
db = 1 # bin padding
lon_bins = np.linspace(min(lons)-db, max(lons)+db, 225+1)
lat_bins = np.linspace(min(lats)-db, max(lats)+db, 150+1)
    
density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    
# Turn the lon/lat of the bins into 2 dimensional arrays ready
# for conversion into projected coordinates
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    
# convert the bin mesh to map coordinates:
xs, ys = m(lon_bins_2d, lat_bins_2d) # will be plotted using pcolormesh
    
    
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


#####################
#  PART 3 - SALARY  #
#####################

# we will create new features for salary minimum and salary maximum
# some jobs list salary in terms of yearly, monthly, or hourly rates, we will convert all to a yearly salary.
def get_salary(x):

    if str(x) == str(np.nan):
        low_salary = np.nan
        high_salary = np.nan

    else:
        # convert all values to yearly salary
        if str(x).__contains__('year'):
            conversion_fact = 1
        elif str(x).__contains__('month'):
            conversion_fact = 12
        elif str(x).__contains__('hour') or str(x).__contains__('hr'):
            conversion_fact = 2080 # working hours in 52 40hr work weeks

        if str(x).__contains__('-'):
            salary_list = str(x).split('-')
            low_salary = conversion_fact * float(salary_list[0].lstrip().rstrip()[1:].replace(',', ''))
            high_salary = conversion_fact * float(salary_list[1].lstrip().rstrip().split(' ')[0][1:].replace(',', ''))
        else:
            salary_list = str(x).split(' ')
#            print salary_list[0].lstrip().rstrip()[1:].replace(',', '')
            low_salary = conversion_fact * float(salary_list[0].lstrip().rstrip()[1:].replace(',', ''))
            high_salary = low_salary

    return (low_salary, high_salary)

df['NumSalaries'] = df.salary.apply(get_salary)

def get_low_salary(x):
    return x[0]

def get_high_salary(x):
    return x[1]

df['LowSalary'] = df.NumSalaries.apply(get_low_salary)
df['HighSalary'] = df.NumSalaries.apply(get_high_salary)
df['MeanSalary'] = 0.5 * (df.LowSalary + df.HighSalary)

df = df.drop(['salary', 'NumSalaries'], axis=1)

print df[['LowSalary', 'HighSalary', 'MeanSalary']].describe()
print df.corr()

#
# Make Salary dist plot
#

sns.set_style(style='darkgrid')
sns.distplot(df.MeanSalary.dropna(), bins=50, norm_hist=False, color='blue', label='Salary')
plt.xlabel('Salary ($/yr)', size=18)
plt.legend(loc='upper right')
plt.savefig('salary_dist_plot.png')
plt.clf()
os.system('open salary_dist_plot.png')


fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121)
ax1.scatter(df.companysize, df.MeanSalary, color='blue', alpha=0.4)
ax1.set_xlabel('N-ratings', size=18)
ax1.set_ylabel('Salary', size=18)

ax2 = fig.add_subplot(122)
ax2.scatter(df.companyrating/df.companyrating.max(), df.MeanSalary, color='blue', alpha=0.4)
ax2.set_xlabel('Normalized Company Rating', size=18)


plt.savefig('salary_scatter_plots.png')
plt.clf()
os.system('open salary_scatter_plots.png')



sns.set_style(style='ticks')
sns.jointplot(df.companysize, df.MeanSalary, kind="hex", stat_func=None, joint_kws={'gridsize':10}, color="#4CB391")

plt.tight_layout()
plt.savefig('salary_companysize_hex.png')
plt.clf()
os.system('open salary_companysize_hex.png')

sns.set_style(style='ticks')
sns.jointplot(df.companyrating, df.MeanSalary, kind="hex", stat_func=None, joint_kws={'gridsize':10}, color="#4CB391")

plt.tight_layout()
plt.savefig('salary_companyrating_hex.png')
plt.clf()
os.system('open salary_companyrating_hex.png')


print df.describe()
print df.corr()

east_df = df[df.longitude >= -100]
west_df = df[df.longitude < -100]

print east_df.describe()
print west_df.describe()

sns.set_style(style='darkgrid')
sns.distplot(east_df.MeanSalary.dropna(), bins=50, norm_hist=False, color='blue', label='East')
sns.distplot(west_df.MeanSalary.dropna(), bins=50, norm_hist=False, color='red', label='West')
plt.xlabel('Salary ($/yr)', size=18)
plt.legend(loc='upper right')
plt.savefig('eastwest_salary_dist_plot.png')
plt.clf()
os.system('open eastwest_salary_dist_plot.png')


