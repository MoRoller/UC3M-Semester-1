import os
import requests
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd

try:
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
except ImportError:
    os.system('python3 -m pip install python-dotenv==1.0.0')
    os.system('python3 -m pip install beautifulsoup4==4.12.2')
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
    
def barplotting(x_values, y_values, title):
    if title == 'Women':
        col = '#FFDA00'
    else: 
        col = '#E01326'
        
    plt.bar(x_values, y_values, width = 0.5, alpha = 1, color = col)
    plt.xlabel('Year')
    plt.ylabel('No of births')
    plt.title(f'{title} born in Catalonia per year')
    for x, y in zip(x_values, y_values):
        plt.text(x, y, str(y), ha='center', va='bottom')
    plt.show()
    

if __name__ == '__main__':
    load_dotenv()
    web_browser = os.getenv('WEB_BROWSER') 
    
    #set urls
    url_maria = 'https://api.idescat.cat/onomastica/v1/nadons/dades.json?id=40683&class=t&lang=en'
    births_data = 'https://www.idescat.cat/pub/?id=naix&n=364&lang=esWEB_BROWSER=Mozilla/5.0 Chrome/97.0.4692.99 Safari/537.36 OPR/83.0.4254.62'
    per_region_base_url = 'https://api.idescat.cat/onomastica/v1/nadons/dades.json?id=40683&class=com&lang=en&t='
    
    
    # Get data for Maria
    try:
        marias_born = requests.get(url_maria).json()['onomastica_nadons']['ff']['f']
    except requests.exceptions.RequestException:
        print(f'Error: Failed to establish a connection to {url_maria}.')
        print(str(requests.exceptions.RequestException))
        marias_born = None
    
    years = [int(dic['c']) for dic in marias_born]
    values = [int(k['pos1']['v']) for k in marias_born]
    df_maria = pd.DataFrame({'Year': years, 'Marias_born': values})
    
    
    # men and women born
    try: 
      response = requests.get(births_data, headers=web_browser)
      soup = BeautifulSoup(response.content, features='html.parser')
      table = soup.find(name='table', class_='ApartNum xs Cols4')
    except:
        print(f'Error: Failed to establish a connection to {births_data}.')
        print(str(requests.exceptions.RequestException))
        response = None

    
    # find all <th> tags
    tr_tags = soup.find_all('tr')
    
    women_values = []
    # loop through <tr>, extract values
    for tr in tr_tags:
        td_tags = tr.find_all('td')
        if td_tags:
            women = td_tags[1].get_text().replace('.','')
            women_values.append(int(women))
    
    # find years
    th_tags = soup.find_all('th')
    years = []
    for th in th_tags:
        text = th.get_text()
        if text == '2022 (p)':
            value = 2022
            years.append(value)
        else:
            try: # try to convert the text to an integer
                value = int(text)
                years.append(value)
            except ValueError:
                pass
    
    df_births = pd.DataFrame({'Year': years, 'Women': women_values})
    df_combined = pd.merge(df_maria, df_births, on= 'Year', how = 'inner')
    
    
    # Plotting
    barplotting(df_maria['Year'][df_maria['Year']>= 2018], df_maria['Marias_born'][df_maria['Year']>= 2018], 'Marias')    
    barplotting(df_births['Year'][df_births['Year']>= 2018], df_births['Women'][df_births['Year']>= 2018], 'Women')
    
    # combined plot (all women + Maria)   
    plt.bar(df_combined['Year'][df_combined['Year']>= 2018], df_combined['Women'][df_combined['Year']>= 2018], color = '#FFDA00', label = 'All women')
    plt.bar(df_combined['Year'][df_combined['Year']>= 2018], df_combined['Marias_born'][df_combined['Year']>= 2018], color = '#E01326', label = 'Maria')
    plt.legend(loc='upper right') 
    plt.xlabel('Year')
    plt.ylabel('No of births')
    plt.title('Total no of women born and women named Maria')
    plt.show()
    
    
    # per Region (separate years)
    per_region_base_url = 'https://api.idescat.cat/onomastica/v1/nadons/dades.json?id=40683&class=com&lang=en&t='
    years = [i for i in range(2018,2023)]
    for year in years:
        url_year = per_region_base_url + str(year)
        
        try:
            data = requests.get(url_year).json()['onomastica_nadons']['ff']['f']
        except requests.exceptions.RequestException as e:
            print(f'Error: Failed to establish a connection to {url_year}')
            print(str(e))
            continue
            
        region = []
        values = []
        for row in data:
            if row['c'] != 'Total':
                value = row['c']['content']
                value2 = row['pos1']['v']
                
                try:
                    value2 = int(value2)
                except:
                    value2 = 0
                
                region.append(value)
                values.append(value2)
               
        df = pd.DataFrame({'Region': region, 'Value': values})
        df.head()
        
        max_index = df['Value'].idxmax()
        city = df.loc[max_index]['Region']
        
        print(f'Region with most Marias born in the year {year}: {city} ({df.loc[max_index]["Value"]})')
    