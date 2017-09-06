import numpy as np
import requests
import time
import os

from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim

coords_dict = dict()

#def get_location_coords(city):
#    """
#    """
#    # for some reason, geolocator can't find coords
#    # for Santa Clara Valley, CA, but it can for
#    # Santa Clara, CA.
#    if city == 'Santa Clara Valley, CA':
#        city = 'Santa Clara, CA'
#
#    if coords_dict.has_key(city):
#        longitude = coords_dict[city][0]
#        latitude = coords_dict[city][1]
#    else:
#        geolocator = Nominatim()
#        loc = geolocator.geocode(city)
#
#        coords_dict[city] = [loc.longitude, loc.latitude]
#
#        longitude = loc.longitude
#        latitude = loc.latitude
#
#    return longitude, latitude

def get_job_info(job):
    """
        We'll want: job title, company, location, 
                    salary (if given), company size, 
                    company rating... Maybe more.
    """

    try:
        job_title = str(job.find('h2', 'jobtitle').a.text.strip())
    except:
        job_title = 'NA'

    try:
        company = str(job.find('span', 'company').text.strip())
    except:
        company = 'NA'

    try:
        location = str(job.find('span', 'location').text.strip())
    except:
        location = 'NA'

    try:
        salary = str(job.find('span', 'no-wrap').text)
    except:
        salary = 'NA'

    try:
        company_size =  int(job.find('span', 'slNoUnderline').text.split()[0]) # assume the number of company ratings scales linearly with size
    except:
        company_size = 'nan'

    try:
        company_rating = float(job.find('span', 'rating').get('style')[6:10])
    except:
        company_rating = 'nan'
    
    try:
        summary = str(job.find('span', 'summary').text.strip()).replace('"', "'")
    except:
        print 'Could not get job summary for %s: %s.'%(company, job_title)
        summary = 'NA'
    

    return  job_title, company, location, salary, company_size, company_rating, summary

def get_next_page(soup):
    """
    """
    next_page_info = soup.find('div', {'class':'pagination'})
    pages_list = next_page_info.find_all('a')
    next_page = pages_list[-1]
    # the new url has separate parts that we need to put together
    next_middle = next_page.get('href')
    next_end = next_page.get('data-pp')

    new_url = home + next_middle + '&pp=' + next_end

    print 'Next url --> ',new_url
    return new_url


####################################################################################
sleep_time = 1
number_of_pages = 60
job_type = ['artificial', 'intelligence']

home = 'https://www.indeed.com'
url = home + '/jobs?q=%s+%s'%(job_type[0], job_type[1])
print 'First page --> ', url
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

jobs = soup.find_all('div', {'class':' row result'})

next_page_number = 2
time.sleep(sleep_time) # wait a little while in case there is a limit on how quickly we can
while next_page_number <= number_of_pages:


    try:
        new_url = get_next_page(soup)
#        print'After %d pages, we have %d jobs'%(next_page_number - 1, len(jobs))
        new_html = requests.get(new_url).text
        new_soup = BeautifulSoup(new_html, 'html5lib')

        new_jobs = new_soup.find_all('div', {'class':' row result'})
        jobs = jobs + new_jobs
    
        soup = new_soup
        del(new_html, new_soup, new_jobs)
    except:
        print 'Could not load page %d'%next_page_number

    next_page_number += 1

    time.sleep(sleep_time)



# write all of the lines to a .csv file
filename = 'new_' + job_type[0] + '_' + job_type[1] + '_jobs.csv'
if os.path.exists(filename):
    os.system('rm ' + filename)

f = open(filename, 'a')
header =  '"jobtitle","company","location","salary","companysize","companyrating","summary"\n'
f.write(header)
for ii in xrange(len(jobs)):
    job_info = get_job_info(jobs[ii])
#    print job_info
    line = '"' + str(job_info[0]) + '","' + str(job_info[1]) + '","' + str(job_info[2]) + '","'\
               + str(job_info[3]) + '",' + str(job_info[4]) + ',' + str(job_info[5]) + ',"' + str(job_info[6]) + '"\n'
    f.write(line)

f.close()

