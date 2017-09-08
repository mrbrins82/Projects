import numpy as np
import requests
import time
import os

from bs4 import BeautifulSoup

def get_job_info(job):
    """
        We'll want: job title, company, location, 
                    salary (if given), company size, 
                    company rating, and a job summary.
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
        company_size =  int(job.find('span', 'slNoUnderline').text.split()[0]) # use number of company ratings as metric for company size
    except:
        company_size = 'nan'

    try:
        company_rating = float(job.find('span', 'rating').get('style')[6:10]) # Each company's star rating is given by pixel width
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
sleep_time = 5 # wait 5 seconds before going to the next page
number_of_pages = 60 # max number of pages to scrape from
job_type = ['artificial', 'intelligence']

home = 'https://www.indeed.com'
url = home + '/jobs?q=%s+%s'%(job_type[0], job_type[1])
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

jobs = soup.find_all('div', {'class':' row result'})

next_page_number = 2
time.sleep(sleep_time)
while next_page_number <= number_of_pages:

    try:
        new_url = get_next_page(soup)
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
filename = 'jobs_' + job_type[0] + '_' + job_type[1] + '.csv'
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

