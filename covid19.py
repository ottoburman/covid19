
def covid_show(Arg1, Arg2):
    import numpy as np #Start: Otto Burman's COVID-19 report by Python 14.5.2020
    import pandas as pd
    datasource = 'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx'
    datat = pd.read_excel(datasource)
    df = pd.DataFrame(datat, columns= ['dateRep','day','month','year','cases','deaths','countriesAndTerritories','geoId','countryterritoryCode','popData2018','continentExp'])
    if Arg2=="": Arg2=Arg1
        
        
    #Arg1
    myCountry = Arg1 # 'Finland' # 'Finland' #  'Sweden' 'Italy' 'Japan' 'Estonia' 'New_Zealand' 'Russia' 
    df_my = df[df['countriesAndTerritories'] == myCountry] # You can change the criterias here
    last_date1 = df_my.iloc[0]['dateRep'] #first element
    first_date1 = df_my.iloc[-1]['dateRep'] #last elemnt
    df_rev = df_my.iloc[::-1] # reversing the order
    df_cum= df_rev['deaths'].cumsum() # counting cumulative deaths 
    deaths_my1 = np.array(df_cum) # making an array for plots
    
    #Arg2
    myCountry = Arg2 # 'Finland' # 'Finland' #  'Sweden' 'Italy' 'Japan' 'Estonia' 'New_Zealand' 'Russia' 
    df_my = df[df['countriesAndTerritories'] == myCountry] # You can change the criterias here
    last_date2 = df_my.iloc[0]['dateRep'] #first element
    first_date2 = df_my.iloc[-1]['dateRep'] #last elemnt
    df_rev = df_my.iloc[::-1] # reversing the order
    df_cum= df_rev['deaths'].cumsum() # counting cumulative deaths 
    deaths_my2 = np.array(df_cum) # making an array for plots
    
    
    import matplotlib.pyplot as plt
    from datetime import datetime
    current_date = datetime.date(datetime.now())
    
    plt.subplot(2, 1, 1) #211=vertically # 1,2,1=horisontally
    plt.plot(deaths_my1)
    plt.title('COVID-19 Deaths in {} \n First obs {} Latest obs {}'.format(Arg1,first_date1, last_date1))
    #plt.xlabel('First obs {} Latest obs {}.'.format(first_date, last_date))
    plt.ylabel('Cumulative data');
    
    plt.subplot(2, 1, 2) #212=vertically # 1,2,2 = horisontally
    plt.plot(deaths_my2)
    plt.title('\n  ')
    #plt.xlabel('First obs {} Latest obs {}.'.format(first_date, last_date))
    plt.xlabel('COVID-19 Deaths in {} \n First obs {} Latest obs {}'.format(Arg2,first_date2, last_date2))
    plt.ylabel('Cumulative data');
    
    plt.show()
    
    

import tkinter as tk

def get_texts():
    global c1, c2
    c1 = e1.get()
    c2 = e2.get()
    
    

def show_covid():
    global c1, c2
    c1 = e1.get()
    c2 = e2.get() 
    covid_show(c1,c2)
    
    


def open_web():
    import webbrowser
    webbrowser.open_new_tab("www.ottoburman.fi\documents\countries.txt")
    
    

root = tk.Tk()
root.geometry("600x300+300+200")
root.title("Compare COVID-19 Deaths between Countries ")


e1 =  tk.Entry(root)
e1.insert(0, 'Finland')
e2 =  tk.Entry(root)
e2.insert(0, 'Sweden')

b1 =  tk.Button(root, text="S_H_O_W ", command= show_covid)
b2 =  tk.Button(root, text="Quit Now" , command= root.destroy)
b3 =  tk.Button(root, text="List for Country Names" , command= open_web)

tk.Label(root,text="Country1: ").pack()
e1.pack()
tk.Label(root,text="Country2: ").pack()
e2.pack()

b1.pack()
b2.pack()
b3.pack()

tk.Label(root,text="\n Data Source: COVID-19 Coronavirus data \n European Union open data").pack()

root.mainloop()

