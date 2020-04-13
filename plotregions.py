#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px 
import numpy as np
from plotly.offline import init_notebook_mode, iplot
from plotly.tools import FigureFactory as FF
init_notebook_mode()
import pandas as pd 
import numpy as np
import adjustText as aT

# In[2]:


df_cases=pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv')
df_deaths=pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv')


# In[3]:


df_cases=df_cases[df_cases['Country/Region']=='India']
df_deaths=df_deaths[df_deaths['Country/Region']=='India']
df_cases=df_cases.drop(columns=['Province/State', 'Lat','Long'])
df_deaths=df_deaths.drop(columns=['Province/State', 'Lat','Long'])
df_cases.iloc[0]


# In[4]:


cases_list=df_cases.iloc[0, :].tolist()
del cases_list[0]
deaths_list=df_deaths.iloc[0, :].tolist()
del deaths_list[0]
Dates=list(df_cases.columns.values)
del Dates[0]


# In[5]:


def generate_graph(y):
    X=[]
    for n in range(1,len(y)+1):
        X.append([n])

    # Fitting Polynomial Regression to the dataset

    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(X)
    poly_reg.fit(X_poly, y)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)
    prediction=lin_reg_2.predict(poly_reg.fit_transform([[len(y)+1]]))
    # Visualising the Polynomial Regression results
#         plt.scatter(X, y, color = 'red')
#         predline=[]
#         for n in range(1,100):
#             predline.append([n])
#         #predline = predline[40:]    
#         plt.plot(predline, lin_reg_2.predict(poly_reg.fit_transform(predline)), color = 'blue')

    #plt.title('Cases')
#         plt.xlabel('Days')
#         plt.ylabel('Cases')
#         plt.show()
    return prediction

def listtostr(s):  
    res = str(s)[1:-1]   
    return (res)


# In[6]:


casepred=listtostr(generate_graph(cases_list))
deathpred=listtostr(generate_graph(deaths_list))


# In[7]:

def deathindia():
    df = pd.DataFrame()
    df["Date"] = pd.Series(Dates)
    df['Date'] = pd.to_datetime(df.Date)
    #df["Cases"] = pd.Series(cases_list)
    df["Deaths"] = pd.Series(deaths_list)


    # data
    df_wide = df
    df_long=pd.melt(df_wide, id_vars=['Date'], value_vars=["Deaths"])
    # plotly 

    print(df_long)
    fig = plt.figure(figsize=(18, 18))
    (df_long["Date"])


    fig.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=70)
    #ax.xaxis.set_major_formatter(df_long["Date"])
    plt.plot(df_long["Date"],df_long['value'],linewidth=10)
    plt.grid(linestyle='dotted')
    plt.savefig("static/plots/deathindia.png")

    #plt.show()
    
def casesindia():
    df = pd.DataFrame()
    
    df["Date"] = pd.Series(Dates)
    df['Date'] = pd.to_datetime(df.Date)
    df["Cases"] = pd.Series(cases_list)
    #df["Deaths"] = pd.Series(deaths_list)


    # data
    df_wide = df
    df_long=pd.melt(df_wide, id_vars=['Date'], value_vars=["Cases"])
    # plotly 

    print(df_long)
    fig = plt.figure(figsize=(18, 18))
    (df_long["Date"])


    fig.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=70)
    #ax.xaxis.set_major_formatter(df_long["Date"])
    plt.plot(df_long["Date"],df_long['value'],linewidth=10)
    plt.yticks(np.arange(min(df_long['value']), max(df_long['value'])+1, 1000))
    plt.grid(linestyle='dotted')
    plt.savefig("static/plots/casesindia.png")
casesindia()

deathindia()
# In[2]:




#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap

dataset = pd.read_csv('test.csv')
def listtostr(s):  
    res = str(s)[1:-1]   
    return (res)
def generate_pred(i):
    
        X=[]
        y=[]
        y = dataset.iloc[:, i].values
        for n in range(1,len(y)+1):
            X.append([n])
        #print(X)
        #print(y)
        poly_reg = PolynomialFeatures(degree = 3)
        X_poly = poly_reg.fit_transform(X)
        poly_reg.fit(X_poly, y)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, y)




        # Visualising the Polynomial Regression results
    #     plt.scatter(X, y, color = 'red')
    #     predline=[]
    #     for n in range(1,21):
    #         predline.append([n])  
    #     plt.plot(predline, lin_reg_2.predict(poly_reg.fit_transform(predline)), color = 'blue')

    #     plt.title(dataset.columns[i])
    #     plt.xlabel('Days')
    #     plt.ylabel('Cases')
    #     plt.show()
    #     for n in range(15,20):
    #         result.append(listtostr(lin_reg_2.predict(poly_reg.fit_transform([[n]]))))
    
        prediction=lin_reg_2.predict(poly_reg.fit_transform([[len(y)+1]]))
        return(prediction) 
def generate_dataframe(pred):
    X_list=['Bombay','n.a. ( 1556)','n.a. ( 1557)','n.a. ( 1558)','n.a. ( 1565)','n.a. ( 1569)','n.a. ( 1571)','Thane','Kalyan']
    Y_list=[float(0.43)*float(pred[0]),float(0.01)*float(pred[0]),float(0.15)*float(pred[0]),0,float(0.14)*float(pred[0]),float(0.27)*float(pred[0]),0,float(pred[2])+float(pred[3]),float(pred[1])]
    df = pd.DataFrame()
    df["state_name"] = pd.Series(X_list)
    df["Cases"] = pd.Series(Y_list)
    return df


# In[10]:


prediction=[]
current=[]
for i in range(1,5):
    prediction.append(listtostr(generate_pred(i)))
    
current=list(dataset.iloc[14, :].values)
del current[0]   
#df1=generate_dataframe(prediction)
#df2=generate_dataframe(current)


# In[11]:


# In[12]:





def generate_linegraph(y,name,City):
            
        predline=[]
        fig, ax = plt.subplots(1, figsize=(10, 6))
        for n in range(1,len(y)+1):
            predline.append([n])  
        plt.plot(predline, y, color = 'blue',linewidth=5)
        plt.title(City)
        plt.xlabel('Days')
        plt.ylabel('Cases')
        plt.grid(linestyle='dotted')
        fig.savefig('static/plots/'+name+'.png')
        


# In[14]:


import numpy as np

def regionsplots():
    
    for i in range (1,5):
        y=dataset.iloc[:, i].values
        curr=sum(y)
        avg=y[-2]+y[-3]+y[-4]
        avg=int(avg/3)
        growth_rate = np.exp(np.diff(np.log(y))) - 1
        str(list(growth_rate).pop()*100)+'%'
        generate_linegraph(y,dataset.columns[i], dataset.columns[i]+' Expected Cases '+str(avg)+' Current Cases '+str(y[-1])+ ' (Cases Growth Rate:' + str("{:.1f}".format(list(growth_rate).pop()*100))+'%)')
    # In[15]:

def wholeregion():
    y=list(dataset.iloc[:, 1].values)
    for i in range (2,5):
        y1=dataset.iloc[:, i].values
        for n in range(0,len(y)):
            y[n]+=y1[n]   
    current=y.pop()
    growth_rate = np.exp(np.diff(np.log(y))) - 1
    str(list(growth_rate).pop()*100)+'%'
    generate_linegraph(y,"wholeregion","Whole region "+'Case Growth Rate:' + str("{:.1f}".format(list(growth_rate).pop()*100))+'%'+'   Mean:'+str("{:.1f}".format((y[-3]+y[-2]+y[-1])/3))+'   Current Cases:'+str(current))

# In[ ]:







# In[ ]:




texts=[]

def generate_predheatmap():
    
    df1=generate_dataframe(prediction)
    data_for_map=df1
    fp = "shapefiles/gadm36_IND_3.shp"
    map_df = gpd.read_file(fp)
    #print(map_df)
    cmap = LinearSegmentedColormap.from_list("", ["#00A1FF","#15FF00","#FF0000"])
    map_df1 = map_df[map_df['NAME_3']=='Thane']
    map_df3 = map_df[map_df['NAME_3']=='Kalyan']
    map_df4 = map_df[map_df['NAME_2']=='Mumbai City']
    map_df5 = map_df[map_df['NAME_2']=='Mumbai Suburban']
    map_df = pd.concat([map_df1,map_df3,map_df4,map_df5])
    #print(map_df)
    map_df["center"] = map_df["geometry"].representative_point()
    za_points = map_df.copy()
    za_points.set_geometry("center", inplace = True)
    merged = map_df.set_index('NAME_3').join(data_for_map.set_index('state_name'))
    #merged.head()
    merged.fillna(0, inplace=True)
    #pint(merged)
    #fig, ax = plt.subplots(1, figsize=(10, 6))
    distlist=['Thane','Kalyan','Mumbai City','Borivali','Kandivali   ','','Kurla','Andheri   ','']
    ax = map_df.plot(figsize = (15, 12), color = "whitesmoke", edgecolor = "0.9", linewidth = 0.5)
    ax.axis('off')
    ax.set_title('Predicted on Area Entering in Hotspot Zone', fontdict={'fontsize': '25', 'fontweight' : '3'})
    aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1,1), expand_text=(1,1))
    merged.plot(column='Cases', cmap=cmap, linewidth=0.5, ax=ax, edgecolor='0.9', legend=True)
    for x, y, label in zip(za_points.geometry.x-0.01, za_points.geometry.y, distlist):
        texts.append(plt.text(x, y, label, fontsize = 12))
    #plt.show()
    plt.savefig('static/plots/generate_predheatmap.png')

def generate_origheatmap():
    df2=generate_dataframe(current)
    df1=generate_dataframe(prediction)
    data_for_map=df1
    fp = "shapefiles/gadm36_IND_3.shp"
    map_df = gpd.read_file(fp)
    #print(map_df)
    cmap = LinearSegmentedColormap.from_list("", ["#00A1FF","#15FF00","#FF0000"])
    map_df1 = map_df[map_df['NAME_3']=='Thane']
    map_df3 = map_df[map_df['NAME_3']=='Kalyan']
    map_df4 = map_df[map_df['NAME_2']=='Mumbai City']
    map_df5 = map_df[map_df['NAME_2']=='Mumbai Suburban']
    map_df = pd.concat([map_df1,map_df3,map_df4,map_df5])
    #print(map_df)
    map_df["center"] = map_df["geometry"].representative_point()
    za_points = map_df.copy()
    za_points.set_geometry("center", inplace = True)
    merged = map_df.set_index('NAME_3').join(data_for_map.set_index('state_name'))
    #merged.head()
    merged.fillna(0, inplace=True)
    #pint(merged)
    #fig, ax = plt.subplots(figsize=(10, 6))
    
    distlist=['Thane','Kalyan','Mumbai City','Borivali','Kandivali   ','','Kurla','Andheri   ','']
    ax = map_df.plot(figsize = (15, 12), color = "whitesmoke", edgecolor = "0.9", linewidth = 0.5)
    ax.axis('off')
    ax.set_title('Area Declared as Hotspot', fontdict={'fontsize': '25', 'fontweight' : '3'})
    aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1,1), expand_text=(1,1))
    merged.plot(column='Cases', cmap=cmap, linewidth=0.5, ax=ax, edgecolor='0.9', legend=True)
    for x, y, label in zip(za_points.geometry.x-0.01, za_points.geometry.y, distlist):
        texts.append(plt.text(x, y, label, fontsize = 12))
    
    plt.savefig('static/plots/generate_origheatmap.png')
    #plt.show()

    
texts=[]

def generate_predheatmapindex():
    
    df1=generate_dataframe(prediction)
    data_for_map=df1
    fp = "shapefiles/gadm36_IND_3.shp"
    map_df = gpd.read_file(fp)
    #print(map_df)
    cmap = LinearSegmentedColormap.from_list("", ["#00A1FF","#15FF00","#FF0000"])
    map_df1 = map_df[map_df['NAME_3']=='Thane']
    map_df3 = map_df[map_df['NAME_3']=='Kalyan']
    map_df4 = map_df[map_df['NAME_2']=='Mumbai City']
    map_df5 = map_df[map_df['NAME_2']=='Mumbai Suburban']
    map_df = pd.concat([map_df1,map_df3,map_df4,map_df5])
    #print(map_df)
    map_df["center"] = map_df["geometry"].representative_point()
    za_points = map_df.copy()
    za_points.set_geometry("center", inplace = True)
    merged = map_df.set_index('NAME_3').join(data_for_map.set_index('state_name'))
    #merged.head()
    merged.fillna(0, inplace=True)
    #pint(merged)
    #fig, ax = plt.subplots(1, figsize=(10, 6))
    distlist=['Thane','Kalyan','Mumbai City','Borivali','Kandivali   ','','Kurla','Andheri   ','']
    ax = map_df.plot(figsize = (15, 12), color = "whitesmoke", edgecolor = "0.9", linewidth = 0.5)
    ax.axis('off')
    ax.set_title('Predicted on Area Entering in Hotspot Zone', fontdict={'fontsize': '25', 'fontweight' : '3'})
    aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1,1), expand_text=(1,1))
    merged.plot(column='Cases', cmap="Reds", linewidth=0.5, ax=ax, edgecolor='0.9', legend=True)
    for x, y, label in zip(za_points.geometry.x-0.01, za_points.geometry.y, distlist):
        texts.append(plt.text(x, y, label, fontsize = 12))
    plt.savefig('static/plots/generate_predheatmapindex.png')
    #plt.show()
    

def generate_origheatmapindex():
    df2=generate_dataframe(current)
    df1=generate_dataframe(prediction)
    data_for_map=df1
    fp = "shapefiles/gadm36_IND_3.shp"
    map_df = gpd.read_file(fp)
    #print(map_df)
    cmap = LinearSegmentedColormap.from_list("", ["#00A1FF","#15FF00","#FF0000"])
    map_df1 = map_df[map_df['NAME_3']=='Thane']
    map_df3 = map_df[map_df['NAME_3']=='Kalyan']
    map_df4 = map_df[map_df['NAME_2']=='Mumbai City']
    map_df5 = map_df[map_df['NAME_2']=='Mumbai Suburban']
    map_df = pd.concat([map_df1,map_df3,map_df4,map_df5])
    #print(map_df)
    map_df["center"] = map_df["geometry"].representative_point()
    za_points = map_df.copy()
    za_points.set_geometry("center", inplace = True)
    merged = map_df.set_index('NAME_3').join(data_for_map.set_index('state_name'))
    #merged.head()
    merged.fillna(0, inplace=True)
    #pint(merged)
    #fig, ax = plt.subplots(figsize=(10, 6))
    
    distlist=['Thane','Kalyan','Mumbai City','Borivali','Kandivali   ','','Kurla','Andheri   ','']
    ax = map_df.plot(figsize = (15, 12), color = "whitesmoke", edgecolor = "0.9", linewidth = 0.5)
    ax.axis('off')
    ax.set_title('Area Declared as Hotspot', fontdict={'fontsize': '25', 'fontweight' : '3'})
    aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1,1), expand_text=(1,1))
    merged.plot(column='Cases', cmap="Reds", linewidth=0.5, ax=ax, edgecolor='0.9', legend=True)
    for x, y, label in zip(za_points.geometry.x-0.01, za_points.geometry.y, distlist):
        texts.append(plt.text(x, y, label, fontsize = 12))
    
    plt.savefig('static/plots/generate_origheatmapindex.png')
    

    
    




# In[ ]:




