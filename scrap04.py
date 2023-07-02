from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

import requests
from bs4 import BeautifulSoup

import os
import time

import re
import nltk
from nltk.corpus import stopwords

import random
import time

#   PARTE 3
# limpieza 
def limpieza(text):
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    return text

#   PARTE 2
# usa requests para scrapear cada full paper y guardarlo en un respectivo txt en la carpeta papers
# recibe de input, el id del articulo
# verifica varias cosas

if not os.path.exists("papers"):
    os.makedirs("papers")

def get_article_url(article_id):
    article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article_id}/"
    return article_url


user_agents_list = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 
                    'Mozilla/5.0 (Windows NT 6.1; Win64; x64)',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6)']  # reemplace esto con su lista de user-agents

def scrape_main_content_text(article_id):
    global counter
    main_content_verif = 0

    headers = { "User-Agent": random.choice(user_agents_list) }

    response = requests.get(get_article_url(article_id), headers=headers)
    
    if random.randint(1, 10) == 5:
        time.sleep(30)

    time.sleep(random.randint(1, 9))

    if response.status_code == 200:
        response_verif = 0
    else:
        response_verif = 1
    soup = BeautifulSoup(response.text, 'html.parser')

    main_content = soup.find('main')
    if not main_content:
        main_content_verif = 1
    
    # Elimina los a que lo arruinan todo
    for a in main_content.find_all('a'):
        a.decompose()

    text_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    # Une el texto de cada elemento de texto, quitando saltos de línea dentro de cada uno
    content_text = '\n'.join([element.get_text(separator='\n').replace('\n', ' ') for element in text_elements])
    
    #
    #escribir archivos
    content = content_text.strip()
    try:
        with open(f"papers/{article_id}.txt", 'w', encoding='utf-8') as file:
            file.write(limpieza(content))
            #file.write(content)
            save_verif = 0
            counter += 1
    except Exception as e:
        save_verif = 1

    print(f'on {counter},   res: {response_verif}, cont: {main_content_verif}, save: {save_verif}    id {article_id}')


#   PARTE 1
# usa selenium para abrir el buscador de papers y tomar los links de cada paper 
# pagina por pagina usando el input de pagina: ingresa numero y apreta enter

def scrape_article_links(driver):
    page_html = driver.page_source
    soup = BeautifulSoup(page_html, 'html.parser')
    # Buscar todos los elementos <a> con atributo href que comienza con "/pmc/articles/" y termina en "/"
    article_links = [a['href'].split("/")[-2] for a in soup.find_all('a', href=lambda href: href and href.startswith("/pmc/articles/") and href.endswith("/"))]
    for article_id in article_links:
        scrape_main_content_text(article_id)

def set_page_value_by_xpath(driver, xpath, page):
    try:
        page_input = driver.find_element(By.XPATH, xpath)
        page_input.clear()
        page_input.send_keys(str(page))
        page_input.send_keys(Keys.RETURN)

    except Exception as e:
        print("Error al input de pagina:", str(e))


#   MAIN
# abre selenium y corre las funciones pagina por pagina

# puede añadir filtros de busqueda en el base url o apretar mas botones
# base_url = 'https://www.ncbi.nlm.nih.gov/pmc/?term=(%5BAll+Fields%5D)'
# page_input_xpath = '//*[@id="pageno2"]'
# counter = 0
{}

def scrapper(term,page_index):
    base_url=f"https://www.ncbi.nlm.nih.gov/pmc/?term=({term})"
    page_input_xpath= f'//*[@id="pageno{page_index}"]'
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(base_url)

    for page in range(1, 11):
        print(f"█████   Página {page}:")
        scrape_article_links(driver)
        set_page_value_by_xpath(driver, page_input_xpath, page+1)
        #time.sleep(5)

    driver.quit()
