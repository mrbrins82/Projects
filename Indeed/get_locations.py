#! usr/bin/env python
import numpy as np
import pandas as pd
import os

from geopy.geocoders import Nominatim

# load job files
ai_df = pd.read_csv('new_artificial_intelligence_jobs.csv')
ds_df = pd.read_csv('new_data_scientist_jobs.csv')
ml_df = pd.read_csv('new_machine_learning_jobs.csv')

# make one data frame out of all job files and drop duplicates
temp_df = pd.concat((ai_df, ds_df))
all_df = pd.concat((temp_df, ml_df))

df = all_df.drop_duplicates()
df.location = df.location.replace(to_replace='Santa Clara Valley, CA', value='Santa Clara, CA')

# we don't need these anymore
del(ai_df, ds_df, ml_df, temp_df, all_df)

# load existing cities and coordinates
locations_df = pd.read_csv('locations.csv')

# see if we already have coordinates for each city and if not
# add it to the locations file
unique_locations = df.location.unique()


geolocator = Nominatim()
for city in unique_locations:
#    print city
    city_stem = city.partition('(')[0].rstrip()

    if locations_df.city.unique().__contains__(city_stem):
        print 'Already have coordinates for %s'%city_stem

    else:
        try:
            loc = geolocator.geocode(city_stem)
            print city_stem
            print loc.longitude, loc.latitude

            # write a new line to the locations file
            f = open('locations.csv', 'a')
            f.write('"' + city_stem + '",' + str(loc.longitude) + ',' + str(loc.latitude) + '\n')
            f.close()

        except:
            print 'Could not geolocate for --> %s'%city_stem


