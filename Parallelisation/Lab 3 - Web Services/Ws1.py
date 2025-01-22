import os
import requests
import matplotlib.pyplot as plt
plt.style.use('ggplot')

try:
    from pyjstat import pyjstat
except ImportError:
    os.system('python3 -m pip install pyjstat==2.4.0')
    from pyjstat import pyjstat

def barplot_per_month(df, title):
  x_values = df['Month']
  y_values = df['value']
  
  months_dict = {'Jan': '01','Feb': '02','Mar': '03','Apr': '04','May': '05',
                 'Jun': '06','Jul': '07','Aug': '08', 'Sep': '09','Oct': '10',
                 'Nov': '11','Dec': '12'}
  
  months = [key for x in x_values for key, value in months_dict.items() if x.split('-')[1] == value]
  plt.figure(figsize=(10, 5))
  colors = ['#d61e11' if val < 0 else '#71a811' for val in df['value']]
  y_max = max(max(y_values)+5, 5)
  y_min = min(min(y_values)-5, -5)

  plt.bar(months, y_values, color = colors)
  plt.xlabel('Month')
  plt.ylabel('Consumer Confidence Indicator')
  plt.title(f'Consumer Confidence Indicator - {title}')
  plt.ylim(y_min, y_max)
  # vertical line y=0
  plt.axhline(y=0, color='grey', linestyle='--', label='y=0', linewidth = 1)
  # add data value labels
  for x, y in zip(months, y_values):
      plt.text(x, y, f'{y:.2f}', ha='center', va='top')
  plt.show()


def barplot_per_year(df, year):
  x_values = df['Country']
  y_values = df['value'].fillna(0)

  plt.figure(figsize=(15, 7.5))
  colors = ['#d61e11' if val < 0 else '#71a811' for val in y_values]
  y_max = max(max(y_values)+5, 10)
  y_min = min(min(y_values)-5, -10)
  plt.ylim(y_min, y_max)
  plt.bar(df['Country'], df['value'], color = colors)
  plt.xticks(rotation=90)
  plt.xlabel('Country')
  plt.ylabel('Consumer Confidence Indicator')
  plt.title(f'Consumer Confidence Indicator per Country - {year}')
  # vertical line y=0
  plt.axhline(y=0, color='grey', linestyle='--', label='y=0', linewidth = 1)
  
  # add data value labels
  for x, y in zip(x_values, y_values):
      plt.text(x, y, f'{y:.2f}', ha='center', va='top')
  plt.show()
     

if __name__ == '__main__':
    eurostat_url = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m?format=JSON&unit=BAL&indic=BS-CSMCI&s_adj=SA&lang=en&sinceTimePeriod=2018&untilTimePeriod=2022'

    response = requests.get(eurostat_url)
    
    # get country names
    data = response.json()      # convert to dict
    countries_dict = data['dimension']['geo']['category']['label']
    countries = list(countries_dict.values())
    # filter non-EU countries
    non_relevant = ['Euro area – 20 countries (from 2023)','European Union - 27 countries (from 2020)','United Kingdom','Montenegro','North Macedonia','Albania','Serbia','Türkiye']
    countries_EU = [element for element in countries if element not in non_relevant]


    # Pyjstat: get data
    df_orig = pyjstat.Dataset.read(eurostat_url).write('dataframe')
    df = df_orig.iloc[:,-3:]
    df.rename(columns={'Geopolitical entity (reporting)': 'Country', 'Time': 'Month'}, inplace = True)
    
    # filter for EU countries
    mask = df.iloc[:,0].isin(countries_EU)
    df_countries_EU = df[mask]
    
    #EU overall
    df_EU = df[df['Country'] == 'European Union - 27 countries (from 2020)']
    df_EU['Country'].values[:] = 'European Union'
    df_EU_2022 = df_EU[df_EU['Month'].str.contains('2022')]
    
    # Plotting
    barplot_per_month(df_EU_2022, 'European Union - 2022')
    
    for year in range(2018,2023):
      # group by country
      df_year_grouped = df_countries_EU[df_countries_EU['Month'].str.contains(str(year), na = False)].groupby(['Country'])['value'].mean().reset_index()
      barplot_per_year(df_year_grouped, year)













