from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC
from bokeh.plotting import figure
from bokeh.models import HoverTool, GeoJSONDataSource, LinearColorMapper, ColorBar,CategoricalColorMapper,ColumnDataSource,BasicTicker
from bokeh.palettes import Category10,inferno,Viridis256
from bokeh.embed import components
from bokeh.transform import factor_cmap,transform
import geopandas as gpd
from geopandas import GeoDataFrame
import urllib
from urllib.request import urlopen
import shutil
import zipfile
from shapely.geometry import Point
import pickle
from math import pi
from scipy.cluster.hierarchy import set_link_color_palette,linkage,dendrogram
from spacy.lang.en.stop_words import STOP_WORDS

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
    color_mapper = CategoricalColorMapper(factors=label_unique,palette=Category10[4])
    
    # Add the lines to the map from our 'msource' ColumnDataSource -object
    p.circle('y', 'x', source=point_source, size=4, color={'field': 'category','transform': color_mapper},
             name="foo")
    
    # Create legend separately
    for factor, color in zip(color_mapper.factors, color_mapper.palette):
        p.circle(x=[], y=[], fill_color=color, legend=factor)
    
    # # Implement interactivity
    my_hover = HoverTool(names=["foo"])
    my_hover.tooltips = [('Neighborhood Name','@Name'),('Shop Primary Category', '@category')]
    p.add_tools(my_hover)
    
    return p


############ Below is for neighborhood clustering ##########################
    

    
app = Flask(__name__)

url='https://raw.githubusercontent.com/PriyankaChakraborti/CoFFee-Shop-Blues/master/Reference_Data/full_feature_matrix.csv'
df_feature2=pd.read_csv(url)
df_feature2.drop(columns=['Unnamed: 0'],axis=1,inplace=True)
df_feature2.set_index('loc_neighborhood',inplace=True)
df_feature2.drop(columns=['mean_rating','mean_review_count'],axis=1,inplace=True)

def clean_select_neighb(df_feature2):
    # Keep only neighborhoods with 10 or more known coffee shops
    data_temp=df_feature2[df_feature2['Count']>=10]
    
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
                       color_threshold=1.0,
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

minmax_scaled_df = clean_select_neighb(df_feature2)


# Shows utility scores for neighborhoods
def showViz3(minmax_scaled_df, neighb):
    
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
def showViz4(minmax_scaled_df):
    
    n_clusters = 7
    importance_df = minmax_scaled_df.drop(columns=['labels_GMM','labels_KMeans','labels_dendo'])
    gmm = GaussianMixture(n_components=n_clusters).fit(importance_df.values)
    labels = gmm.predict(importance_df.values)
    
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


###### From NLP Section
def positive_negative(score):
    # This is the division, where above this is considered positive and below is considered negative
    split_score = 600
    # This is essentially a correction to above. It now needs to be this much above split_score to be considered
    # positive, or this much below to be considered negative
    buffer = 100
    
    # Positive is signified by 2, negative by 0, and neutral by 1
    if score>(split_score+buffer):
        rating = 2 # Positive ratings signified by 2
        
    elif score < (split_score-buffer):
        rating = 0 # Negative ratings signified by 0
        
    else:
        rating = 1 # Neutral rating
        
    return rating

def find_polarity_score(relevant_frequency,relevant_words_score,relevant_reviews):
    
    # Set words as index
    relevant_words_score.set_index('word', inplace=True)
    relevant_polarity_score = relevant_words_score
    relevant_polarity_score['frequency'] = relevant_frequency
    
    # Calculate polarity score 
    relevant_polarity_score['polarity'] = relevant_polarity_score.score * relevant_polarity_score.frequency / relevant_reviews.shape[0]
    
    # Drop unnecessary words not useful for coffee shop owners (Combination of stopwords and manually found)
    stop_words_list= list(STOP_WORDS)
    
    manually_rev_words = ['great','amazing','love','best','awesome','excellent','good','favorite','loved',
                          'perfect','gem','perfectly','wonderful','happy','enjoyed','nice','well','super',
                          'like','better','decent','fine','pretty','enough','excited','impressed','ready',
                          'fantastic','glad','right','fabulous','bad','disappointed','unfortunately','always',
                          'disappointing','horrible','lacking','terrible','sorry', 'disappoint','ni','ha',
                          'try','nwe','review','come','diego','friend','two','10','nservice','nfood','giving',
                          'went','coming','hour','austin','thing','visit','first','nthe','starbucks','since',
                          'mcdonald','today','dunkin','work','experience','star','cup','truck','make','easy',
                          'day','gas','peet','trip','night','place','restaurant','time','ordered','cool','keep',
                          'reservation','open','building','worst','want','store','located','lady','totally',
                          'different','chicago','wife','campus','afternoon','maybe','week','eleven','inside',
                          'something','month','weekend','away','say','ever','year','nfirst','however','la',
                          'nthis','couple','nmy','town','mind','na','ngreat','boyfriend','really','gift','car',
                          'wanted','morning','white','piece','office','run','low','area','cleveland','three',
                          'go','got','long','eat','seat','took','thought','yelp','vegas','san','bomb','choose',
                          'though','group','dc','sd','paying','brand','seemed','woman','item','life','stumbled',
                          'stuff','corner','java','ton','sit','visiting','mi','also','nif','top','past','heard',
                          'although','sunday','wish','average','neighborhood','across','told','customer',
                          'probably','wa','wall','list','literally','ended','prior','bit','saturday','found',
                          'job','may','concept','seattle','given','front','usually','trying','little','recent',
                          'one','look','hotel','take','business','drink','nthey','would','take','made','yet',
                          'tried','walking','street','feel','whole','made','give','way','next','know','never',
                          'think','even','definitely','cafe','absolutely','quite','sat','husband','ate','torchy',
                          'twice','kerbey','south','without','texas','stop','several','eating','stopped','point',
                          'must','visited','need','house','many','bowl','almost','let','decided','still','yes',
                          'end','kinda','saw','nit','felt','done','le','mary','sure','poke','said','care',
                          'easily','name','lamar','came','walked','everyone','recommend','foods','nothing',
                          'leave','used','looking','nso','plus','buy','ncame','taken','arrived','around','back',
                          'nyc','lovely','upon','ago','matter','least','doe','recently','ideal','question',
                          'pick','often','use','second','every','downtown','city','much','solid','nthere',
                          'serve','get','left','started','finally','high','add','actually','expect','far',
                          'available','within','shop','bike','boston','could','going','everything',
                          'detroit','ve','atlanta','baltimore','boulder','charlotte','charlottesville',
                          'columbus','minute','lot','waikiki','houston','minneapolis','nola','orleans',
                          'oakland','philly','pittsburgh','slc','santa','fe','ta','nm','spot','pearl',
                          'start','north','find','hawaiian','honolulu','center','brooklyn','strip',
                          'denver','space','dallas','worth','island','ll','bay']
    
    # Combine into full list of words to ignore
    all_unnecessary_words_list = stop_words_list + manually_rev_words

    keep_list = [(word not in all_unnecessary_words_list) for word in relevant_polarity_score.index]
    relevant_polarity_score = relevant_polarity_score.loc[keep_list].copy(deep=True)
    
    # Convert output to float
    relevant_polarity_score.polarity = relevant_polarity_score.polarity.astype(float)
    relevant_polarity_score.frequency = relevant_polarity_score.frequency.astype(float)

    return relevant_polarity_score

def plot_word_polarity(df,title):
    
    # construct features and labels
    text_list=list(df['Text_Lemmatized'])
    labels_list=list(df['rat_cat'])
    
    # Get bag of words for passed dataframe
    vectorizer = CountVectorizer()
    feature_counts=vectorizer.fit_transform(text_list)
    
    # Use linear support vector classifier to differentiate between positive and negative words
    svm = LinearSVC()
    svm.fit(feature_counts, labels_list)
    
    # Below must be repeated for positive and negative words separately
    
    # Create dataframe for score of each word in a review calculated by svm model
    coeff = svm.coef_[0]
    relevant_words_score = pd.DataFrame({'score': coeff, 'word': vectorizer.get_feature_names()})
    # Get frequency of each word in all reviews grouped as positive
    relevant_reviews = pd.DataFrame(feature_counts.toarray(), columns=vectorizer.get_feature_names())
    relevant_reviews['labels'] = labels_list
    relevant_frequency = relevant_reviews[relevant_reviews['labels'] ==2].sum()[:-1]
    # Calculate polarity score
    relevant_polarity_score = find_polarity_score(relevant_frequency,relevant_words_score,relevant_reviews)
    relevant_polarity_score = relevant_polarity_score[relevant_polarity_score.polarity>0].sort_values('polarity', ascending=False)[:10]
    top_words_pos = relevant_polarity_score.loc[list(relevant_polarity_score.index),'polarity']
    
    # Create dataframe for score of each word in a review calculated by svm model
    coeff = svm.coef_[0]
    relevant_words_score = pd.DataFrame({'score': coeff, 'word': vectorizer.get_feature_names()})
    # get frequency of each word in all reviews grouped as negative
    relevant_reviews = pd.DataFrame(feature_counts.toarray(), columns=vectorizer.get_feature_names())
    relevant_reviews['labels'] = labels_list
    relevant_frequency = relevant_reviews[relevant_reviews['labels'] ==0].sum()[:-1]
    # Calculate polarity score
    relevant_polarity_score = find_polarity_score(relevant_frequency,relevant_words_score,relevant_reviews)
    relevant_polarity_score = relevant_polarity_score[relevant_polarity_score.polarity<0].sort_values('polarity', ascending=True)[:10]
    top_words_neg = relevant_polarity_score.loc[list(relevant_polarity_score.index),'polarity']
    
    # Make a plot of the top words for positive reviews and negative reviews using polarity
    top_words = pd.Series(top_words_pos).append(pd.Series(top_words_neg))
    top_words = top_words.sort_values(ascending=False)
    
    words = top_words.index.tolist()
    polarity_scores = top_words.values.tolist()
    
    # Needed for using bokeh
    source = ColumnDataSource(data=dict(words=words, polarity_scores=polarity_scores))
    
    p = figure(x_range=words, plot_height=500, plot_width=1100, toolbar_location=None, title="Positive and Negative Words")
    
    p.vbar(x='words', top='polarity_scores', width=0.9, source=source,
           line_color='white', fill_color=factor_cmap('words', palette=inferno(len(top_words.index.tolist())), factors=words))
    
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = pi/4
    p.xaxis.major_label_text_font_size = "17pt"
    
    # Implement interactivity
    my_hover = HoverTool()
    my_hover.tooltips = [('Word','@words'),('SVM Coefficient','@polarity_scores{0.0000}')]
    p.add_tools(my_hover)
    
    return p

# Download Yelp Reviews
url='https://raw.githubusercontent.com/PriyankaChakraborti/CoFFee-Shop-Blues/master/Reference_Data/coffee_reviews_cleaned.csv'
df_coffee_reviews=pd.read_csv(url)
# Now apply binning to all shops
df_coffee_reviews['rat_cat'] = df_coffee_reviews['Utility_Score'].apply(positive_negative)



@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/about')
def aboutpage():
    return render_template('about.html')

@app.route('/contact')
def contactpage():
    return render_template('contact.html')

@app.route('/neighb_map', methods=['POST', 'GET'])
def neighb_map():
    
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
    return render_template("neighb_map.html", script=script, div=div,
                           script2=script2,div2=div2,list_of_cities=list_of_cities,
                           chosen_city=chosen_city,chosen_city2=chosen_city)


@app.route('/neighb_clusters', methods=['POST', 'GET'])
def neighb_clusters():
    neighb = request.args.get("chosen_neighborhood")
    if neighb == None:
        neighb = 'Midtown, New York'
    
    #create the plot
    plot1=showViz3(minmax_scaled_df,neighb)
    
    # Embed plot into HTML via Flask Render
    script, div = components(plot1)
    
    return render_template("neighb_clusters.html", script=script, div=div,
                           list_of_neighb=minmax_scaled_df.index.to_list(),
                           neighb=neighb)

@app.route('/pos_neg_words', methods=['POST', 'GET'])
def pos_neg_words():
    # Only send in subset of cities as some fail with SMV calculations
    truncated_city_list=['Atlanta', 'Baltimore', 'Boston', 'Boulder',
                         'Charlotte', 'Charlottesville', 'Cleveland',
                         'Columbus', 'Dallas', 'Denver', 'Detroit', 'Honolulu',
                         'Houston', 'Las Vegas', 'Miami', 'Minneapolis',
                         'New Orleans', 'New York', 'Oakland', 'Philadelphia',
                         'Pittsburgh', 'Salt Lake City', 'San Diego',
                         'San Francisco', 'Washington']

    chosen_city = request.args.get("chosen_city")
    
    if chosen_city == None:
       chosen_city='Boston'
       
    df_chosen_city = df_coffee_reviews.loc[df_coffee_reviews['City'] == chosen_city]
    title = 'TOP 10 Positive and Negative Words For The City Of ' + chosen_city
    plot = plot_word_polarity(df_chosen_city,title)
    script, div = components(plot)
    
    plot2=showViz4(minmax_scaled_df)
    script2, div2 = components(plot2)
    
    return render_template("pos_neg_words.html", script=script, div=div,
                           script2=script2, div2=div2,
                           list_of_cities=truncated_city_list,chosen_city=chosen_city)

if __name__ == '__main__':
    app.run(port=33507)
