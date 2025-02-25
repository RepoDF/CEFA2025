import pandas as pd

def get_data(start_year, end_year):
    all_data = []
    start_year = str(start_year)
    end_year = str(end_year)
    for i in range(int(start_year[-2:]),int(end_year[-2:])):
        year = str(2000+i)
        url1 = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/'+year+'/all?type=daily_treasury_yield_curve&field_tdr_date_value='+year+'page&_format=csv'
        
        data = pd.read_csv(url1)
        all_data.append(data)
    pd_alldata = pd.concat(all_data,ignore_index=True)
    pd_alldata['Date'] = pd.to_datetime(pd_alldata['Date'])
    pd_alldata.index = pd_alldata['Date']
    
    req_cols = ['3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', 
                '7 Yr','10 Yr', '20 Yr', '30 Yr']
    final = pd_alldata[req_cols]
    return final