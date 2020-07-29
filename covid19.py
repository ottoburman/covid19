#**************************************************************#
# UI is defined by tkinter - the code is in the end of this program
#**************************************************************#
# Otto Burman's COVID-19 report by Python 27.5.2020
# updated 2020_0723
#-----------------------------------------------------------------------#
# to make a exe use appropriate spec file!
# python covid19_forec.py
#-----------------------------------------------------------------------#
import os
os.system('cls')
#-----------------------------------------------------------------------#
print('\n Machine Learning \n- Polynomial Regression Modeling by Dr Otto Burman ')
#-----------------------------------------------------------------------#

def covid_show(Arg1, Arg2, days, model_d,forecast_len):
    global pol_model, dgr, r2adj, r2
    import numpy as np 
    import pandas as pd
    datasource = 'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx' 
    print('\n Open Data Source: \n ', datasource)
    #--------------------------------------------------------------------#
    datat = pd.read_excel(datasource)
    df = pd.DataFrame(datat, columns= ['dateRep','day','month','year','cases','deaths','countriesAndTerritories','geoId','countryterritoryCode','popData2019','continentExp'])
    if Arg2=="": Arg2=Arg1
    nr_lastdays=days # how many latest datapoints
    # Something to read :) https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
    
    #Arg1
    #1st Country --------------------------------------------------------#
    myCountry = Arg1 # 'Finland' # 'Finland' #  'Sweden' 'Italy' 'Japan' 'Estonia' 'New_Zealand' 'Russia' 
    df_my = df[df['countriesAndTerritories'] == myCountry] # You can change the criterias here
    last_date1 = df_my.iloc[0]['dateRep'] #first element
    first_date1 = df_my.iloc[-1]['dateRep'] #last element
    popData1 = df_my.iloc[-1]['popData2019'] #last element - dis/does not work 24.6.:name has been changed?
    
    df_rev = df_my.iloc[::-1] # reversing the order
    df_cum= df_rev['deaths'].cumsum() # counting cumulative deaths 
    deaths_my  = np.array(df_cum) # making an array for plots
    # how many days for making the forecast
    if  nr_lastdays < 40 : 
        nr_lastdays= 40 
    if  nr_lastdays > len(deaths_my) : 
        nr_lastdays = len(deaths_my) 
        
    from_these = int(len(deaths_my)-nr_lastdays) 
    if from_these < 0: from_these = 0 
    # the data for the 1st country
    deaths_my1 = deaths_my[from_these:] # elements after from_these 
    sum_deaths1 = deaths_my1[-1]
    deaths_rate1=(sum_deaths1/popData1)*100
    deaths_rate1=round(deaths_rate1,3)
    
    
    #forecasts for the first country-------------#
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn.metrics import r2_score
    # https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
    # https://www.w3schools.com/python/python_ml_polynomial_regression.asp
    y = deaths_my1 # [8,10,12,16,18,24,27,32,37,36,39,44,47]
    x = np.linspace(0, len(y), len(y)) # time is explaining y 
    
    # calculate the best model for the country
    dgr = 0
    pol_model=""
    r2adj=0.0
    r2=0.0
    # step is to call sub-program, that selects the model and is degree
    formulate(y) # the dataserie (y) to the sub-program
    print('\n Machine Learning process is giving the most suitable model for \n', 
            Arg1, ' /  ', 'Deaths:', sum_deaths1, 'Population:', popData1, "deaths/population*100=", deaths_rate1, '%\n',
            pol_model, '\n dgr: ', dgr,' R2(adj):', r2adj, ' R2: ', round(r2,2), '\n')
    # exit() # the exit was for testing purpose only
    
    #dgr-value is used for my_pol_model fkn y= x0+x1**2+x2**3+...
    my_pol_model = np.poly1d(np.polyfit(x, y, dgr))
    # r2 was already calculated #
    r2=((r2_score(y, my_pol_model(x)))*100) # how many % of the variation is explaided
    r2=int(round(r2,0))
    
    #We can make a forecast based one single value
    #next_x = x[-1]+1  # One single value  for x = last x + 1 # next_x = len(x)+1 
    #next_y = int(round(my_pol_model(next_x),0)) # using the model to make prediction y
    #or by using several values
    lenght_fo = forecast_len # lenght of the forecast was asked in my_UI() 
    next_x = np.linspace(len(x), len(x)+lenght_fo, lenght_fo) # forecast for several days
    # the following prints are for learning and testing purposes 
    #print('next_x', next_x) # testing purposes 
    next_y = (my_pol_model(next_x),0) # using the above defined model to make prediction for y (deaths) # print(next_y)
    next_y = np.array(next_y) # #print(next_y)
    next_y = next_y.tolist() # the np.array should be converted to python list #print(next_y)
    next_y = (next_y[0]) #print(next_y)
    # print('next_y start end', next_y[0],next_y[-1]) 
    if (next_y[0]-next_y[-1]) <= 0: 
        trend1='(rising trend!)'
    else: 
        trend1='(no new cases!)'
        
    my_timeline = np.linspace(0, len(x)+lenght_fo, len(x)+lenght_fo) #for plotting
    #my_forecast------------------------------------------#
    from datetime import datetime
    current_date = datetime.date(datetime.now())
    # PLOT1 -------------------------------------------------------------
    plt.subplot(2, 1, 1) #211=vertically # 1,2,1=horisontally
    #plt.subplot(1, 2, 1) #211=vertically # 1,2,1=horisontally
    #plt.plot(deaths_my1)
    #-------------------------------------------------------------------
    
    
    
    plt.scatter(x, y, color='black', marker='.')
    plt.plot(x, my_pol_model(x),label='Polynomial Model (d={}; $R^2$(adj)={})'.format(dgr,r2adj))
    plt.plot(next_x, next_y, 'rX', label='Forecast for {} days: {}'.format(lenght_fo,trend1))
    plt.legend(loc='lower right')
    #--------------------------------------------------------------------
    plt.title('COVID-19 Deaths({}) \n Model Estimation based on the last {} days'
              .format(last_date1, nr_lastdays), fontsize=10) 
    #plt.text(nr_lastdays*0.50,sum_deaths1*0.90, 'Source: European Open Data', fontsize=8)
    plt.axis([1, (len(x)+lenght_fo)*1.05, 1, next_y[0]*1.05])# xmin - max -- ymin - max 
    plt.text(5,next_y[0]*0.8 , '{} ({} population) \n{} deaths \n({} %)'
            .format(Arg1, int(popData1), sum_deaths1, deaths_rate1,  ), fontsize=8    )
    plt.ylabel('Cumulative Deaths')
    # PLOT1 ending-------------------------------------------------------------------
    
    #Arg2 --------------------------------------------------------#
    #2nd Country -------------------------------------------------#
    myCountry = Arg2 # 'Finland' # 'Finland' #  'Sweden' 'Italy' 'Japan' 'Estonia' 'New_Zealand' 'Russia' 
    df_my = df[df['countriesAndTerritories'] == myCountry] # You can change the criterias here
    last_date2 = df_my.iloc[0]['dateRep'] #first element
    first_date2 = df_my.iloc[-1]['dateRep'] #last element
    popData2 = df_my.iloc[-1]['popData2019'] #last element
    df_rev = df_my.iloc[::-1] # reversing the order
    df_cum= df_rev['deaths'].cumsum() # counting cumulative deaths 
    deaths_my = np.array(df_cum) # making an array for plots
    from_these=int(len(deaths_my)-nr_lastdays) 
    # the data for the 2nd country:
    deaths_my2=deaths_my[from_these:]
    sum_deaths2=deaths_my2[-1]
    deaths_rate2=(sum_deaths2/popData2)*100
    # print (deaths_rate2)
    deaths_rate2= round(deaths_rate2,3)
    # print (deaths_rate2)
    
    #forecasts for the second country 
    # print (len(deaths_my1))
    #print(nr_lastdays)
    # https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
    # https://www.w3schools.com/python/python_ml_polynomial_regression.asp
    #https://medium.com/@aadhil.imam/plotting-polynomial-function-in-python-361230a1e400
    #print(deaths_my1)
    y = deaths_my2 # [8,10,12,16,18,24,27,32,37,36,39,44,47]
    x = np.linspace(0, len(y), len(y)) # time is explaining y 
    # print(y,x)
    
    # calculate the best model for the country
    dgr = 0
    pol_model=""
    r2adj=0.0
    r2=0.0
    # Calling the sub-program
    formulate(y)
    # Printing the results
    print('\n Machine Learning process is giving the most suitable model for \n', 
            Arg2, ' /  ', 'Deaths:', sum_deaths2, 'Population:', popData2, "deaths/population*100=", deaths_rate2, '%\n',
            pol_model, '\n dgr: ', dgr,' R2(adj): ', r2adj, ' R2: ', round(r2,2), '\n')
    # exit() # the exit was for testing purpose only
    
    #dgr=model_d # degree for the polynom fkn y= x0+x1**2+x2**3+...
    my_pol_model = np.poly1d(np.polyfit(x, y, dgr))
    r2=((r2_score(y, my_pol_model(x)))*100) # how many % of the variation is explaided
    r2=int(round(r2,0))
    #print('Explained ', r2,'%')
    #One single value
    #next_x = x[-1]+1  # One single value  for x = last x + 1 # next_x = len(x)+1 
    #next_y = int(round(my_pol_model(next_x),0)) # using the model to make prediction y
    #Several values
    lenght_fo = forecast_len # lenght of the forecast
    next_x = np.linspace(len(x), len(x)+lenght_fo, lenght_fo) # forecast for seven days
    #print('next_x', next_x)
    next_y = (my_pol_model(next_x),0) # using the model to make prediction y
    next_y = np.array(next_y)
    next_y = next_y.tolist()
    next_y = (next_y[0])
    #print(next_y) 
    if (next_y[0]-next_y[-1]) <= 0: 
        trend2='(rising trend!)'
    else: 
        trend2='(no new cases!)'
        
        
    #plots 
    my_timeline = np.linspace(0, len(x)+lenght_fo, len(x)+lenght_fo) #for plotting
    #my_forecast for the second country ends ---------------------#
    # PLOTING for the 2nd country -------------------------
    plt.subplot(2, 1, 2) #212=vertically # 1,2,2 = horisontally
    #plt.subplot(1, 2, 2) #212=vertically # 1,2,2 = horisontally
    #plt.plot(deaths_my2)
    #-------------------------------------------------------------------
    plt.scatter(x, y, color='black', marker='.')
    plt.plot(x, my_pol_model(x),label='Polynomial Model(d={}; $R^2$(adj)={})'.format(dgr,r2adj))
    plt.plot(next_x, next_y, 'rX', label='Forecast for {} days: {}'.format(lenght_fo,trend2))
    plt.legend(loc='lower right')
    #--------------------------------------------------------------------
    plt.title('\n ')
    plt.axis([1, (len(x)+lenght_fo)*1.05, 1, next_y[0]*1.05])# xmin - max -- ymin - max 
    
    plt.text(5,next_y[0]*0.8, '{} ({} population) \n{} deaths \n({} %)'
            .format(Arg2, int(popData2), sum_deaths2, deaths_rate2  ), fontsize=8    )
    
    plt.xlabel('Time (days) \nSource: European Open Data \n', fontsize=8 )
    plt.ylabel('Cumulative Deaths')
    
    #----------------------------------------------------------------------
    if Create_Picture==Arg1+'_'+Arg2+'.pdf': 
        plt.savefig(Arg1+'_'+Arg2+'.pdf')
    else:
        plt.show()
        
    #----------------------------------------------------------------------
    #The picture memeory should be emptied
    # https://www.kite.com/python/examples/1886/matplotlib-clear-a-figure
    plt.clf() # Otherwise picures are accumulated
    #----------------------------------------------------------------------
    #**********************************************************************
    
    
    # End ot the 2nd PLOT for the 2nd Country ----------------------------
    
    # Selecting the best polynomial model for presenting the data
def formulate(y):
    best_model=10
    my_digits=3
    global pol_model, dgr, r2adj, r2
    import numpy as np
    #import matplotlib.pyplot as plt
    import sklearn
    from sklearn.metrics import r2_score
    # https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
    # https://www.w3schools.com/python/python_ml_polynomial_regression.asp
    # Testdata
    # y = [8,10,12,16,18,24,35,32,37,36,39,44,47]
    x = np.linspace(0, len(y), len(y)) # time is explaining y 
    
    dgrmax=best_model
    my_round=my_digits
    dgr=0
    lists=np.array([0.0,0.0,0.0,0.0,0.0])
    for dgr in range(1, dgrmax):
        #print(dgr) 
        #degee  loop starts 
        fkn = np.polyfit(x, y, dgr)
        predict = np.poly1d(fkn)
        from sklearn.metrics import mean_squared_error
        MSE = round(mean_squared_error(y, predict(x)),4)
        # https://www.listendata.com/2014/08/adjusted-r-squared.html
        yhat = predict(x)
        SS_Residual = sum((y-yhat)**2)       
        SS_Total    = sum((y-np.mean(y))**2)     
        r_2     = round((1 - (SS_Residual/SS_Total)), 5)
        p       = (dgr + 1)
        n       = len(y)
        r_2adj  = round(1 - (1-r_2)*((n - 1)/(n - p - 1)),my_round) # accuracte enoug?
        RMSE = round(np.sqrt(mean_squared_error(y,yhat)),4)
        list1= np.array([MSE, RMSE, r_2adj , dgr, r_2])
        # next is order to avoid the overestimation 
        # if r_2adj<1.0: 
        lists = np.vstack([lists, list1]) # must be the same size as list1
        
        
    # degree loop ends
    
    #print('[MSE,  RMSE,  r_2adj,  dgr,  r_2] ==> lists:')
    #print(lists)
    r2adj_col=(lists[:,2])
    #print('\n r2adj column [:,2] ', r2adj_col)
    
    # https://thispointer.com/find-max-value-its-index-in-numpy-array-numpy-amax/
    # Find index of maximum value from numpy array
    result = np.where(r2adj_col == np.amax(r2adj_col))
    # print('Observe: Returning tuple of arrays :', result)
    # https://stackoverflow.com/questions/20637970/python-convert-tuple-to-array
    # print('that tuple has to be convered to np.array \n in two steps as follows')
    result = np.array(result)
    result = result.flatten()
    # Select the first of the indices
    # https://stackoverflow.com/questions/15887885/converting-a-one-item-list-to-an-integer
    first_indice = (result[0]) #
    row_arr=(lists[first_indice,:])
    # print('\n The row where is the first occurence,: ', row_arr)
    r2adj= (row_arr[2])
    r2   = (row_arr[4])
    # print('\n r2adj (best): ', r2adj, '  r2: ', r2, '\n')
    dgr=int((result[0]))
    # print('and the most suitable degree was: ', dgr, ' beacuse of the fewest amount of the parameters! \n')
    
    pol_model=''
    if dgr==0:  pol_model=('f(x)= {:0.2f} '.format(fkn[0]))
    if dgr==1:  pol_model=('f(x)= {:0.2f} + {:0.2f}*x '.format(fkn[0],fkn[1]))
    if dgr==2:  pol_model=('f(x)= {:0.2f} + {:0.2f}*x + {:0.2f}*(x**2) '.format(fkn[0],fkn[1],fkn[2]))
    if dgr==3:  pol_model=('f(x)= {:0.2f} + {:0.2f}*x + {:0.2f}*(x**2) + {:0.2f}*(x**3)'.format(fkn[0],fkn[1],fkn[2],fkn[3]))
    if dgr==4:  pol_model=('f(x)= {:0.2f} + {:0.2f}*x + {:0.2f}*(x**2) + {:0.2f}*(x**3) + {:0.2f}*(x**4)'.format(fkn[0],fkn[1],fkn[2],fkn[3],fkn[4]))
    if dgr==5:  pol_model=('f(x)= {:0.2f} + {:0.2f}*x + {:0.2f}*(x**2) + {:0.2f}*(x**3)+ {:0.2f}*(x**4) + {:0.2f}*(x**5)'.format(fkn[0],fkn[1],fkn[2],fkn[3],fkn[4],fkn[5]))
    if dgr>5 :  pol_model=('f(x)= x0 + x1 + x2*(x*x) + x3*(x*x*x)+ ... x{}*(x*x*x*x...x{})'.format(dgr,dgr))
    # print('The best suitable model was ===> ', pol_model, 'degree: ', dgr)
    dgr_new=dgr # global varable which value defines the plots 
    
    return()
    # best model selection process ends here
    
#**************************************************************#
# UI by tkinter starts here
#**************************************************************#

import tkinter as tk
from tkinter.ttk import Combobox
 # https://www.tutorialsteacher.com/python/create-ui-using-tkinter-in-python
window = tk.Tk()


def get_texts():
    global c1, c2
    c1 = cb1.get()
    c2 = cb2.get()
    
    
def show_covid():
    global Create_Picture
    Create_Picture=''
    
    global c1, c2
    c1 = cb1.get()
    c2 = cb2.get() 
    degree=3 #int(d1.get())
    days=int(d2.get())
    forecast_len=int(d3.get())
    ## Causes an error return self.func(*args)
    covid_show(c1,c2,days,degree,forecast_len)
    
    
def open_help():
    import webbrowser
    webbrowser.open_new_tab("www.ottoburman.fi\documents\countries.txt")
    
    
def show_picture_pdf():
    global Create_Picture
    Create_Picture=cb1.get()+'_'+cb2.get()+'.pdf'
    
    global c1, c2
    c1 = cb1.get()
    c2 = cb2.get() 
    degree=3 #int(d1.get())
    days=int(d2.get())
    forecast_len=int(d3.get())
    ## Causes an error return self.func(*args)
    covid_show(c1,c2,days,degree,forecast_len)
    #----------------------------------------------#
    import webbrowser
    webbrowser.open_new_tab(cb1.get()+'_'+cb2.get()+'.pdf')
    

# data=("one", "two", "three", "four")
# import numpy as np 
import pandas as pd
# for testing download the data  
# C:\Users\xxx\Downloads\COVID-19-geographic-disbtribution-worldwide.xlsx
# datasource = 'C:/Users/xxx/Downloads/COVID-19-geographic-disbtribution-worldwide.xlsx' #6.7.2020
datasource = 'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx' #6.7.2020
# print('\n Data Source: \n ', datasource)
datat = pd.read_excel(datasource)
df = pd.DataFrame(datat, columns= ['dateRep','day','month','year','cases','deaths','countriesAndTerritories','geoId','countryterritoryCode','popData2019','continentExp'])
# applying groupby() function to group the data by the Code  
gk = df.groupby('countryterritoryCode') 
CCodes = gk.first()
# type(CCodes) # pandas Dataframe converts to lists...
# https://datatofish.com/convert-pandas-dataframe-to-list/
CountryNames = CCodes['countriesAndTerritories'].values.tolist()
# print ('List of all Contries: ', CountryNames)

data = CountryNames
cb1 = Combobox(window, values=data)
# cb1.place(x=30, y=150)
cb2 = Combobox(window, values=data)
# cb2.place(x=30, y=350)
cb1.insert(0, 'Finland')
cb2.insert(0, 'Sweden')

# d1 =  tk.Entry(window)
# d1.insert(0, '3')
d2 =  tk.Entry(window)
d2.insert(0, '100')
d3 =  tk.Entry(window)
d3.insert(0, '2')

tk.Label(window,text="Country1: ").pack()
cb1.pack()
tk.Label(window,text="Country2: ").pack()
cb2.pack()

tk.Label(window,text="\n ").pack()
b1 =  tk.Button(window, text="_S_H_O_W__PLOTS ", command = show_covid)
b1.pack()

tk.Label(window,text="\n ").pack()
b2 =  tk.Button(window, text="_S_H_O_W__AND_SAVE_picture as pdf", command = show_picture_pdf)
b2.pack()

tk.Label(window,text="\n EXTRA OPTIONS: \n Automatic Polynom Degree (d) Modelling \n for the forecasts \n f(y)= x0 + x1 + x2*(x*x) + x3*(x*x*x)+ ... xd*(x*x*x*x...xd)}").pack()
# d1.pack() 
tk.Label(window,text="How many (last) observations for the model:").pack()
d2.pack()
tk.Label(window,text="Lengt for the forecast (days):").pack()
d3.pack()

b3 =  tk.Button(window, text="HELP: Gives a list and shows the source code!" , command= open_help)
b3.pack()

b2 =  tk.Button(window, text="Quit Now" , command= window.destroy)
b2.pack()

tk.Label(window,text="\n Data Source: \n COVID-19 Coronavirus data \n European Union open data \n").pack()

window.geometry("600x700+300+100")
window.title("Compare COVID-19 Deaths between Countries ")
window.mainloop()
#********************************************************************
#UI code ends here
#********************************************************************
