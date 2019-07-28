#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 24 11:18:53 2019

@author: priyanka
"""

from flask import Flask, render_template, request

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from math import pi
from bokeh.models import LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,HoverTool,ColorBar,BasicTicker
from bokeh.transform import transform
from bokeh.embed import components
from scipy.cluster.hierarchy import set_link_color_palette,linkage,dendrogram

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
    
app = Flask(__name__)

url='https://raw.githubusercontent.com/PriyankaChakraborti/CoFFee-Shop-Blues/master/Reference_Data/full_feature_matrix.csv'
df_feature=pd.read_csv(url)
df_feature.drop(columns=['Unnamed: 0'],axis=1,inplace=True)
df_feature.set_index('loc_neighborhood',inplace=True)
df_feature.drop(columns=['mean_rating','mean_review_count'],axis=1,inplace=True)

def clean_select_neighb(df_feature):
    # Keep only neighborhoods with 10 or more known coffee shops
    data_temp=df_feature[df_feature['Count']>=10]

    Y=np.asarray(data_temp['Avg_Utility_Score'])
    X=np.asarray(data_temp.loc[:,~data_temp.columns.isin(['loc_City','Avg_Utility_Score','Count'])])

    columns=[col for col in data_temp.columns if col not in ['Avg_Utility_Score', 'Count','loc_City']]

    imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    scaler = preprocessing.MinMaxScaler()
    minmax_scaled_df = scaler.fit_transform(X)
    minmax_scaled_df = pd.DataFrame(minmax_scaled_df, 
                                    columns=columns,index=data_temp.index) 
    minmax_scaled_df=pd.concat([minmax_scaled_df,pd.DataFrame(data_temp['loc_City'],index=data_temp.index)],axis=1)
    minmax_scaled_df['loc_id'] = minmax_scaled_df.index + ', ' + minmax_scaled_df.loc_City
    minmax_scaled_df.drop(columns=['loc_City'],inplace=True)
    minmax_scaled_df.set_index('loc_id',drop=True,inplace=True)
    minmax_scaled_df = minmax_scaled_df.loc[~minmax_scaled_df.index.duplicated(keep='first')]

    samples=minmax_scaled_df.values
    labs = minmax_scaled_df.index
    set_link_color_palette(['teal','sandybrown', 'steelblue', 'firebrick', 'forestgreen', 'darkviolet', 'crimson', 'darkcyan', 'peru', 'indigo', 'darkorange'])
    mergings = linkage(samples, method='complete')

    # Apply dendogram being applying PCA
    dendo = dendrogram(mergings,
                       labels=labs,
                       leaf_rotation=0,
                       leaf_font_size=14,
                       color_threshold=1.0, # 1.2 gives 11 clusters
                       orientation='right',
                       no_plot=True)
    a = pd.DataFrame(dendo['color_list'])
    val_dict = dict(a.iloc[:,0].value_counts())
    colors = a.iloc[:,0].unique()
    color_list = []
    for color in colors:
        if color != 'b':
            for i in range(val_dict[color] + 1):
                color_list.append(color)



    neighb_colors = pd.DataFrame([color_list, dendo['ivl'], dendo['leaves']]).T
    neighb_colors.rename({0:'color', 1:'loc_id', 2:'leaf'}, axis=1,inplace=True)
    neighb_colors = neighb_colors.set_index('loc_id', drop=True)

    # Apply PCA, keeping top 5 features
    pca = PCA(5)
    projected = pca.fit_transform(minmax_scaled_df.values)

    # Now, after PCA, apply K-Means and GMM clustering
    n_clusters = 7
    kmeans = KMeans(n_clusters, random_state=42)
    labels_kmeans = kmeans.fit(projected).predict(projected)
    gmm = GaussianMixture(n_components=n_clusters).fit(projected)
    labels_GMM = gmm.predict(projected)

    minmax_scaled_df['labels_GMM']=labels_GMM
    minmax_scaled_df['labels_KMeans']=labels_kmeans
    neighb_colors = neighb_colors[['color']]
    neighb_colors = neighb_colors.sort_index()
    # Merge with Dendogram output
    minmax_scaled_df=minmax_scaled_df.merge(neighb_colors, on='loc_id',how='outer')
    minmax_scaled_df.rename(columns={'color': 'labels_dendo'}, inplace=True)
    return minmax_scaled_df

minmax_scaled_df = clean_select_neighb(df_feature)


# Shows utility scores for neighborhoods
def showViz1(minmax_scaled_df, neighb):
    
    # Find correlation between all neighborhoods
    a = [[int(i == j) for i in minmax_scaled_df.labels_KMeans] for j in minmax_scaled_df.labels_KMeans]
    b = [[int(i == j) for i in minmax_scaled_df.labels_GMM] for j in minmax_scaled_df.labels_GMM]
    c = [list(map(lambda a,b: a+b, a[i], b[i])) for i in range(len(a))]
    d = [[int(i == j) for i in minmax_scaled_df.labels_dendo] for j in minmax_scaled_df.labels_dendo]
    f = [list(map(lambda a,b: a+b, c[i], d[i])) for i in range(len(d))]
    
    temp = pd.DataFrame(data=f, index=minmax_scaled_df.index, columns=minmax_scaled_df.index)
    
    # Keep only those correlations with the current neighborhood
    hi = temp.loc[:,[neighb]].sort_values(by=neighb,ascending=False).head(20).index
    test=minmax_scaled_df.loc[hi]
    
    # Find all correlations between neighborhoods with highest correlation with current neighborhood
    a = [[int(i == j) for i in test.labels_KMeans] for j in test.labels_KMeans]
    b = [[int(i == j) for i in test.labels_GMM] for j in test.labels_GMM]
    c = [list(map(lambda a,b: a+b, a[i], b[i])) for i in range(len(a))]
    d = [[int(i == j) for i in test.labels_dendo] for j in test.labels_dendo]
    f = [list(map(lambda a,b: a+b, c[i], d[i])) for i in range(len(d))]
    
    # Save these correlations in a dataframe
    temp = pd.DataFrame(data=f, index=test.index, columns=test.index)
    
    #Now we will create correlation matrix using pandas
    df = temp.copy(deep=True)

    df.index.name = 'AllColumns1'
    df.columns.name = 'AllColumns2'

    # Prepare data.frame in the right format
    df = df.stack().rename("value").reset_index()

    # You can use your own palette here
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']

    # I am using 'Viridis256' to map colors with value, change it with 'colors' if you need some specific colors
    mapper = LinearColorMapper(
        palette=Viridis256, low=0, high=3)

    # Define a figure and tools
    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help"
    
    p = figure(
        tools=TOOLS,
        plot_width=800,
        plot_height=800,
        title="Correlation plot for neighborhoods similar to " + neighb,
        x_range=list(df.AllColumns1.drop_duplicates()),
        y_range=list(df.AllColumns2.drop_duplicates()),
        toolbar_location="right",
        x_axis_location="below")

    # Create rectangle for heatmap
    p.rect(
        x="AllColumns1",
        y="AllColumns2",
        width=0.9,
        height=0.9,
        source=ColumnDataSource(df),
        line_color=None,
        fill_color=transform('value', mapper),name='foo')
    
    p.xaxis.major_label_orientation = pi/4
    p.yaxis.major_label_orientation = pi/4

    # Add legend
    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=10))
    
    p.add_layout(color_bar, 'right')
    hover = HoverTool(names=["foo"])
    hover.tooltips = [('Neighborhood 1','@AllColumns1'),('Neighborhood 2','@AllColumns2'),('Num Clustering Matches','@value')]
    p.add_tools(hover)
    
    return p

# Shows positions of coffee shops
def showViz2(minmax_scaled_df):
    
    n_clusters = 7
    importance_df = minmax_scaled_df.drop(columns=['labels_GMM','labels_KMeans','labels_dendo'])
    gmm = GaussianMixture(n_components=n_clusters).fit(importance_df.values)
    labels = gmm.predict(importance_df.values)
    
    #X_train,X_test,Y_train,Y_test=train_test_split(importance_df.values,labels,test_size=0.2,random_state=42)
    X = importance_df.values
    y = labels
    
    
    tree = DecisionTreeClassifier( random_state = 42 )
    tree.fit( X , y )
    
    imp = pd.DataFrame(
            tree.feature_importances_,
            columns = ['Importance'],
            index = [item for item in list(importance_df.columns)]
            )
        
    imp = imp.sort_values( [ 'Importance' ] , ascending = False )
    imp.reset_index(level=0, inplace=True)
        
    source = ColumnDataSource(imp)
    features = source.data['index'].tolist()
        
    p = figure(x_range=features)
    p.vbar(x='index', top='Importance', source=source, width=0.70, name='foo')
    
    p.title.text ='Feature importance as classified by a decision Tree'
    p.xaxis.axis_label = 'Feature Names'
    p.yaxis.axis_label = 'Feature importance'
    
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = pi/4
    
    my_hover = HoverTool(names=["foo"])
    my_hover.tooltips = [('Feature','@index'),('Importance', '@Importance{1.111}')]
    p.add_tools(my_hover)
    
    return p

@app.route('/', methods=['POST', 'GET'])
def visualize():
    
    neighb = request.args.get("chosen_neighborhood")
    if neighb == None:
        neighb = 'Midtown, New York'
    
       #create the plot
    plot1=showViz1(minmax_scaled_df,neighb)
    plot2=showViz2(minmax_scaled_df)
    # Embed plot into HTML via Flask Render
    script, div = components(plot1)
    script2, div2 = components(plot2)
    #return render_template("coffee_map.html", script=script, div=div,
    #                       script2=script2,div2=div2,list_of_cities=list_of_cities,
    #                       chosen_city=chosen_city,chosen_city2=chosen_city)
    return render_template("coffee_map.html", script=script, div=div,
                           script2=script2, div2=div2,
                           list_of_neighb=minmax_scaled_df.index.to_list(),
                           neighb=neighb)

if __name__ == '__main__':
	app.run(port=33507)
