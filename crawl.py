from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import time
import os

def crawling_ulasan_pos(split_load):
    option = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=option)
    # Buka Browser Berdasarkan URL
    url_en = 'https://www.google.com/maps/place/The+Palace+of+Yogyakarta/@-7.8052845,110.3642031,17z/data=!4m8!3m7!1s0x2e7a5796db06c7ef:0x395271cf052b276c!8m2!3d-7.8052845!4d110.3642031!9m1!1b1!16s%2Fm%2F0vb3k_5?hl=en'
    driver.get(url_en)
    time.sleep(1)
    # Mengurutkan dari yang terbaru
    driver.find_element('xpath', '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div[7]/div[2]/button').click()
    time.sleep(1)
    driver.find_element('xpath', '//*[@id="action-menu"]/div[3]').click()
    time.sleep(5)
    # Scroll Halaman
    last_review =driver.find_element('css selector','div.m6QErb.DxyBCb.kA9KIf.dS8AEf')
    for i in range(0, split_load):
        driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', last_review)
        time.sleep(1)
    # Mengambil Review
    item = driver.find_elements('xpath', '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div[9]')
    time.sleep(5)
    
    name_list = []
    stars_list = []
    review_list = []
    duration_list = []
    for i in item:
        button = i.find_elements('tag name','button')
        for m in button:
            if m.text == "More":
                m.click()
        time.sleep(5)    
        name = i.find_elements("class name","d4r55")
        stars = i.find_elements("class name", "kvMYJc")
        review = i.find_elements("class name","wiI7pd")
        duration = i.find_elements("class name","rsqaWe")
        for j,k,l,p in zip(name,stars,review,duration):
            name_list.append(j.text)
            duration_list.append(p.text)
            stars_list.append(k.get_attribute("aria-label"))
            review_list.append(l.text)
        driver.quit()
        df = pd.DataFrame({'name': name_list,
                           'duration': duration_list,
                           'rating': stars_list,
                           'review': review_list
                           })
        return df
        

def crawling_ulasan_neg(split_load):
    option = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=option)

    # Buka Browser Berdasarkan URL
    url_en = 'https://www.google.com/maps/place/The+Palace+of+Yogyakarta/@-7.8052845,110.3642031,17z/data=!4m8!3m7!1s0x2e7a5796db06c7ef:0x395271cf052b276c!8m2!3d-7.8052845!4d110.3642031!9m1!1b1!16s%2Fm%2F0vb3k_5?hl=en'
    driver.get(url_en)
    time.sleep(1)

    # Mengurutkan dari yang terbaru
    driver.find_element('xpath', '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div[7]/div[2]/button').click()
    time.sleep(1)
    driver.find_element('xpath', '//*[@id="action-menu"]/div[4]').click()
    time.sleep(5)

    # Scroll Halaman
    last_review =driver.find_element('css selector','div.m6QErb.DxyBCb.kA9KIf.dS8AEf')
    for i in range(0, split_load):
        driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', last_review)
        time.sleep(1)

    # Mengambil Review
    item = driver.find_elements('xpath', '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div[9]')
    time.sleep(5)

    name_list = []
    stars_list = []
    review_list = []
    duration_list = []

    for i in item:
        button = i.find_elements('tag name','button')
        for m in button:
            if m.text == "More":
                m.click()
        time.sleep(5)
        
        name = i.find_elements("class name","d4r55")
        stars = i.find_elements("class name", "kvMYJc")
        review = i.find_elements("class name","wiI7pd")
        duration = i.find_elements("class name","rsqaWe")

        for j,k,l,p in zip(name,stars,review,duration):
            name_list.append(j.text)
            duration_list.append(p.text)
            stars_list.append(k.get_attribute("aria-label"))
            review_list.append(l.text)

        driver.quit()

        df = pd.DataFrame({'name': name_list,
                           'duration': duration_list,
                           'rating': stars_list,
                           'review': review_list
                           })
        return df
        

def crawling_ulasan(jumlah_load):
    split_load = int(jumlah_load/2)
    df_pos = crawling_ulasan_pos(split_load)
    df_neg = crawling_ulasan_neg(split_load)
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    return df