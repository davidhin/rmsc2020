#%%#######################################################################
#                                SETUP                                   #
##########################################################################
import os
import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import requests
import pandas as pd
import numpy as np
from progressbar import progressbar as pb
from glob import glob

#%%#######################################################################
#                                  SETUP                                 #
##########################################################################

# Get vuln ids
df = pd.read_parquet('../data/post_topics_final.parquet')
df = df[df.key.str.contains('so')]
df[['postid','source']] = df.key.str.split('_',expand=True)
so_ids = np.array_split(df.postid.astype('str').tolist(), np.ceil(len(df) / 30000))

## Login Details
gusername = 'hello@davidhin.com'
gpassword = 'Lighterush741'

#%%#######################################################################
#                                  Login                                 #
##########################################################################

## Set up driver
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=chrome_options)

##%% Global Wait
wait = WebDriverWait(driver, 40)

## OAuth Login
driver.get("https://data.stackexchange.com/account/login")
driver.find_element_by_css_selector("div[class='preferred-login']").find_element_by_css_selector('span').click()
email = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='email")))
email.send_keys(gusername)
wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Next')]"))).click()
password = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password")))
password.send_keys(gpassword)
wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Next')]"))).click()
driver.get("https://data.stackexchange.com/account/login")
driver.find_element_by_css_selector("div[class='preferred-login']").find_element_by_css_selector('span').click()

#%%#######################################################################
#                               Basic Data                               #
##########################################################################

## Read queries
q = 'select Id as id, CreationDate as creationdate from Posts where Id IN ({})'

## Get SO (change sql when necessary)
for count, batch in pb(enumerate(so_ids)):
    driver.get("https://data.stackexchange.com/stackoverflow/query/new")
    query = q.format(','.join(batch))
    jscript = "document.getElementsByClassName('CodeMirror')[0].CodeMirror.setValue('{}')".format(query)
    driver.execute_script(jscript)

    ## Submit Query and download results
    button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[id='submit-query")))
    button.click()
    results = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[id='resultSetsButton")))
    results_link = results.get_attribute('href')
    result_csv = requests.get(results_link)

    ## Save results
    print(len(result_csv.content))
    open('../data/sodates-{}.csv'.format(count), 'wb').write(result_csv.content)

# %%
