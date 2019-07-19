#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:18:53 2019

@author: priyanka
"""

from flask import Flask, render_template, request, redirect
#import bokeh
import pandas as pd
#import json
# import os
from bokeh.plotting import figure
from bokeh.models import HoverTool, GeoJSONDataSource, LinearColorMapper, ColorBar,CategoricalColorMapper
from bokeh.transform import factor_cmap
from bokeh.palettes import inferno,Spectral6,Plasma6
from bokeh.io import show, output_notebook
from bokeh.embed import components
import geopandas as gpd
from geopandas import GeoDataFrame
import urllib
import shutil
import zipfile
from shapely.geometry import Point
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pickle

app = Flask(__name__)

cities_to_states = {'Atlanta':'GA',
                    'Austin':'TX',
                    'Baltimore':'MD',
                    'Boston':'MA',
                    'Boulder':'CO',
                    'Charlotte':'NC',
                    'Charlottesville':'VA',
                    'Chicago':'IL',
                    'Cleveland':'OH',
                    'Columbus':'OH',
                    'Dallas':'TX',
                    'Denver':'CO',
                    'Detroit':'MI',
                    'Honolulu':'HI',
                    'Houston':'TX',
                    'Las Vegas':'NV',
                    'Los Angeles':'CA',
                    'Miami':'FL',
                    'Minneapolis':'MN',
                    'New Orleans':'LA',
                    'New York':'NY',
                    'Oakland':'CA',
                    'Philadelphia':'PA',
                    'Pittsburgh':'PA',
                    'Portland':'OR',
                    'Salt Lake City':'UT',
                    'San Diego':'CA',
                    'San Francisco':'CA',
                    'Santa Fe':'NM',
                    'Seattle':'WA',
                    'Washington':'DC',}

# Definition pulls information from Zillow neighborhood boundaries to return geopandas dataframe for provided city
def create_zillow_map(city):
    state = cities_to_states[city]

    zillow_call = 'https://www.zillowstatic.com/static-neighborhood-boundaries' + \
    '/LATEST/static-neighborhood-boundaries/shp/' + \
    'ZillowNeighborhoods-{}.zip'.format(state)

    city_shp = 'ZillowNeighborhoods-{}.shp'.format(state)
    file_name = '{}_shape.zip'.format(state)
    
    # download zip from Zillow website
    with urllib.request.urlopen(zillow_call) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    # extract shapefile into current folder
    with zipfile.ZipFile(file_name,"r") as zip_ref:
        zip_ref.extractall()

    data = gpd.read_file(city_shp)
    if city != 'Boston':
        data = data[data['City']==city]
    elif city == 'Boston':
        data = data[(data['City']==city) | (data['City']=='Cambridge') \
                    | (data['City']=='Newton')]
        
    data.crs = {'init' :'epsg:4326'}
    
    return data

# For ML Predictions
url='https://raw.githubusercontent.com/PriyankaChakraborti/CoFFee-Shop-Blues/master/Reference_Data/full_feature_matrix.csv'
df_feature=pd.read_csv(url)
df_feature.drop(columns=['Unnamed: 0'],axis=1,inplace=True)
df_feature.set_index('loc_neighborhood',inplace=True)

Y=np.asarray(df_feature['Avg_Utility_Score'])
X=np.asarray(df_feature.loc[:,~df_feature.columns.isin(['loc_City','Avg_Utility_Score','Count','mean_rating','mean_review_count'])])

columns=[col for col in df_feature.columns if col not in ['loc_City','Avg_Utility_Score','Count','mean_rating','mean_review_count']]


# Scale data
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)
scaler = preprocessing.MinMaxScaler()
minmax_scaled_df = scaler.fit_transform(X)
minmax_scaled_df = pd.DataFrame(minmax_scaled_df, 
                                columns=columns,index=df_feature.index)

minmax_scaled_df=pd.concat([minmax_scaled_df,pd.DataFrame(df_feature['loc_City'],index=df_feature.index)],axis=1)
minmax_scaled_df['loc_id'] = minmax_scaled_df.index + ', ' + minmax_scaled_df.loc_City
minmax_scaled_df.drop(columns=['loc_City'],inplace=True)
minmax_scaled_df.set_index('loc_id',drop=True,inplace=True)

# Apply PCA and keep top 5 features
pca = PCA().fit(minmax_scaled_df.values)
pca = PCA(5)
projected = pca.fit_transform(minmax_scaled_df.values)

# Load pre-created ML pickle file
url='https://raw.githubusercontent.com/PriyankaChakraborti/CoFFee-Shop-Blues/master/Reference_Data/finalized_Lasso_model.sav'
from urllib.request import urlopen
lasso_regress_model = pickle.load(urlopen(url))

# Create predictions for all neighborhoods
predictions_neighbs=list(lasso_regress_model.predict(projected))
df_feature['Predicted_Utility_Score']=[abs(number) for number in predictions_neighbs]
df_feature.reset_index(level=0, inplace=True)

# Create utility score to display on map
def fun(row):
    if row['Count'] >= 10:
        val = row['Avg_Utility_Score']
    else:
        val = row['Predicted_Utility_Score']
    return val

df_feature['disp_score'] = df_feature.apply(fun, axis=1)
df_feature['Name'] = df_feature['loc_neighborhood']




# For displaying cofee shops
url='https://raw.githubusercontent.com/PriyankaChakraborti/CoFFee-Shop-Blues/master/Reference_Data/df_city.csv'
df_city=pd.read_csv(url)

list_of_cities=list(cities_to_states.keys())

# Shows utility scores for neighborhoods
def showViz1(gpd_city_neighborhoods, chosen_city):
    
    comb_gpd = gpd.GeoDataFrame(gpd_city_neighborhoods[['State','County','City','Name','geometry']].merge(df_feature[['Name','disp_score','Count']],how='inner',on='Name'))
    chosen_city_map_gpd = comb_gpd.loc[comb_gpd['City']==chosen_city]
    chosen_city_map_gpd = chosen_city_map_gpd.drop_duplicates(subset='Name', keep='first')# Drop any duplicate neighborhoods
    
    # Read the data
    main_map = gpd_city_neighborhoods
    
    # Reproject to the same coordinate system
    CRS = {'init' :'epsg:4326'}
    data=chosen_city_map_gpd
    data.crs = CRS
    main_map.crs = CRS
    
    # Convert GeoDataFrames into GeoJSONDataSource objects (similar to ColumnDataSource)
    point_source = GeoJSONDataSource(geojson=data.to_json())
    map_source = GeoJSONDataSource(geojson=main_map.to_json())
    
    # Initialize our plot figure
    p = figure(title="Neighborhood Utility Scores (Higher is better for business)")
    
    # Add the background neighborhood regions to the map from our map_source GeoJSONDataSource object
    # (it is important to specify the columns as 'xs' and 'ys')
    p.patches('xs', 'ys', source=map_source, color='gray', alpha=0.30, line_width=3,line_color = "white")
    
    # Create color map which will be used for displaying utility scores as unique colors
    color_mapper = LinearColorMapper(palette='Magma256', low=0, high=1000)
    # color_mapper = LogColorMapper(palette='Magma256', low=400, high=4000)
    
    # Add the neighborhood data to the map from the point_source ColumnDataSource object
    # (it is important to specify the columns as 'xs' and 'ys')
    p.patches('xs', 'ys', source=point_source, color={'field': 'disp_score','transform': color_mapper},
              line_width=3,line_color = "white",name="foo")
    
    # Add color bar
    color_bar = ColorBar(color_mapper=color_mapper, width=8,  location=(0,0))
    p.add_layout(color_bar, 'right')
    
    # Implement interactivity
    my_hover = HoverTool(names=["foo"])
    my_hover.tooltips = [('Neighborhood Name','@Name'),('Neighborhood Utility Score', '@disp_score'),
                         ('Number of Stores','@Count')]
    p.add_tools(my_hover)
    
    return p

# Shows positions of coffee shops
def showViz2(gpd_city_neighborhoods, chosen_city):
    
    city_map  = gpd_city_neighborhoods
    city_data = df_city[(df_city.City == chosen_city)]
    
    a = city_data.Location.apply(lambda x: x.split(', '))
    x = a.apply(lambda x: float(x[0][1:]))
    y = a.apply(lambda x: float(x[1][:-1]))
    city_data['x'] = x
    city_data['y'] = y
    geometry=[Point(xy) for xy in zip(city_data.x, city_data.y)]
    crs = city_map.crs
    city_data = GeoDataFrame(city_data, crs=crs, geometry=geometry)
    label=city_data.category.tolist()
    label_unique=list(set(label))
    print(label_unique)
    # Convert GeoDataFrames into GeoJSONDataSource objects (similar to ColumnDataSource)
    point_source = GeoJSONDataSource(geojson=city_data.to_json())
    map_source = GeoJSONDataSource(geojson=city_map.to_json())
    p = figure(title="Surveyed Coffee Shops in Chosen City (View For Density)")
    # Add the lines to the map from our GeoJSONDataSource -object (it is important to specify the columns as 'xs' and 'ys')
    p.patches('xs', 'ys', source=map_source, color='gray', alpha=0.30, line_width=3,line_color = "white")
    # Create color map which will be used for displaying utility scores as unique colors
    color_mapper = CategoricalColorMapper(factors=label_unique,palette=Plasma6)
    
    # Add the lines to the map from our 'msource' ColumnDataSource -object
    p.circle('y', 'x', source=point_source, size=4, color={'field': 'category','transform': color_mapper},
             name="foo")

    for factor, color in zip(color_mapper.factors, color_mapper.palette):
        p.circle(x=[], y=[], fill_color=color, legend=factor)
    
    # # Implement interactivity
    my_hover = HoverTool(names=["foo"])
    my_hover.tooltips = [('Neighborhood Name','@Name'),('Neighborhood Category', '@category')]
    p.add_tools(my_hover)
    
    return p

@app.route('/', methods=['POST', 'GET'])
def visualize():
    
    chosen_city = request.args.get("chosen_city")
    if chosen_city == None:
        chosen_city = 'Austin'
    
    
    gpd_city_neighborhoods = create_zillow_map(chosen_city)
    gpd_city_neighborhoods2 = create_zillow_map(chosen_city)
    
       #create the plot
    plot1=showViz1(gpd_city_neighborhoods,chosen_city)
    plot2=showViz2(gpd_city_neighborhoods2,chosen_city)
    # Embed plot into HTML via Flask Render
    script, div = components(plot1)
    script2, div2 = components(plot2)
    return render_template("coffee_map.html", script=script, div=div,
                           script2=script2,div2=div2,list_of_cities=list_of_cities,
                           chosen_city=chosen_city,chosen_city2=chosen_city)

if __name__ == '__main__':
	app.run(port=33507)
