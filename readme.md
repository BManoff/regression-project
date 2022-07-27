### Project Goals

The project goal is to Predict home value based on features from the Zillow dataframe.


### Project Description

Home value and housing estimates are a controversial subject. It is important to be able to accurately predict home prices in any given area. The features included in the Zillow data may lead us to create a model that will allow for those, hopefully, accurate predictions.


### Initial Questions

How does location effect value?

How does size effect price?

How does home age relate to value?

### Data Dictionary

| Variable                                      Meaning         
| -----------                                  -----------       
|  |                  
|              |         
|                 |         
|                       |         
|                  |           
|                    |       
|           |        
                              


### Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the telco table. Store that env file locally in the repository. Use a gitignore file to hide your env.

2. clone my repo (including the wrangle.py and explore.py files)

3. libraries used are pandas, matplotlib, seaborn, stats, numpy, sklearn and scipy

4. Run my wrange/ explore / final_report files to reproduce work

### The Plan

1.) Initial questions
2.) Data aquisition
3.) Data cleaning
4.) Exploration
5.) Modeling explored data
6.) drawing conclusions
7.) Making suggestions based on models
8.) Making suggestions for next steps

### Key Findings

size, age and location were all significantly correlated with home value:

- More square footage, bathrooms, and bedrooms increased home value
- Newer homes were higher in value
- location was correlated with home value

#### Recommendations

- Utilize the polynomial linear regression model to make price predictions

#### Next steps

- Look at several other features to include proximity to entertainment, schools, or other social and geographical factors