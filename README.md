# CoFFee-Shop-Blues
'Serving coffee shop success with a side of Regression.'<br>
![Alt Text](https://miro.medium.com/max/1000/1*K0wyFSD3tHy8rDy5kd9caw.gif)



The goal of this project is to develop an end-to-end solution for a potential coffee business, big or small, to unlock their true potential as a coffee shop. In lieu of publicly available coffee shop repositeries I have attempted to create my own by extensive feature engineering from the sources listed below. Additionally, borrowing some ideas from this page (https://medium.com/topos-ai/the-next-wave-predicting-the-future-of-coffee-in-new-york-city-23a0c5d62000) I engineered a target that I label the 'Utility Score'. Basically, it represents the popularity metric for a neighborhood coffee shop. I then weighted this 'score' by classification results from Bayesian NLP.

I then applied Lasso Regression with optimal parameters from 5-fold Cross Validation and was able to generate Utility Scores for neighborhoods having very few coffee shops with an r^2 score of 0.82. The visualization of these results can be seen in the Heroku app below. If the user wishes to start a coffee shop in a neighborhood of their choice they can visualize the Utility Scores generated from my app and decide which might be the best neighborhood for them.

In the future,I would like to update my web based app to recommend additional items they might want to sell in their coffee shop to optimize their profits. The idea behind this is to model additional items sold at their competitor coffee shops as strategies in a Nash game with our user being one of the players. A regret-based  neural network will play this game until we hit Nash equilibrium, and then display the optimal strategy. The code for this is still being optimized (see Coffee_Shop_Nash) and the final version will be available shortly.

Note that this code (encluding Nash Equilibrium) can be directly used by visiting the below heroku apps:
- Interactive Neighborhood Scores Map (May take 10-20 seconds to load)<br>
  https://interactive-neighb-score-maps.herokuapp.com/
- Interactive Neighborhood Cluster Map (May take 10-20s to load)<br>
  https://interactive-neighb-cluster-map.herokuapp.com/

For manually running this code the files should be run in the following order:
1) ScrapeAPI
2) obtain_google_search_counts
3) Mega_Data_Compiler
4) NLP
5) Feature_engineering_and_Analysis
6) Coffee_Shop_Nash- This is still being optimized,hence may need modifications!!

Data Sources
1) Yelp API - https://api.yelp.com
2) Uber Movement - https://movement.uber.com/cities?lang=en-US
3) Zillow API - https://www.zillow.com/howto/api/APIOverview.htm
4) US Census API - https://www.census.gov/developers/
5) Google.com - Scraped search results for rating
