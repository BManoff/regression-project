from pydataset import data # importing librabries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import env
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split

# turn off warnings
import warnings
warnings.filterwarnings("ignore")

# connect to the mysql server
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Splitting the data into train, validate and test 
def train_validate_test_split(df, seed=123):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
    )
    return train, validate, test

def get_wrangle_zillow():
    #Get local cached file if it's there
    filename = "zillow.csv" 

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql(
        ''' 
        SELECT
        bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, fips, yearbuilt
        FROM 
        properties_2017
        JOIN
        predictions_2017 USING (id)
        JOIN
        propertylandusetype USING (propertylandusetypeid)
        WHERE
        propertylandusedesc = "Single Family Residential"
        AND
        transactiondate like "2017-__-__"
        ;

        '''
        
        , get_connection('zillow')
        )


        # Return the dataframe after pulling the query
        return df  

def handle_nulls(df):
      
    #dropping all the na values
    df = df.dropna() 

    # returning the dataframe once all changes have been made
    return df




def type_change(df):
    df["fips"] = df["fips"].astype(int) # Changing fips feature to an integer data type

    df["yearbuilt"] = df["yearbuilt"].astype(int) # Changing yearbuilt feature to an integer data type

    df["bedroomcnt"] = df["bedroomcnt"].astype(int)  # Changing bedroomcnt feature to an integer data type 

    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int) # Changing taxvaluedollar feature to an integer data type

    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int) # Changing calculatedfinishedsquarefeet feature to an integer data type

    return df # returning the dataframe once all changes have been made

def handle_outliers(df):
    
    # defining quartiles to get rid of outliers outside of the inter-quartile range (IQR)
    cols = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'yearbuilt']
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # setting hard limits on bathroom counts, bedroom counts, and value upper limit, just in case they weren't filtered out by the IQR range
    df = df[df.bathroomcnt <= 6]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 2_000_000]

    # return the dataframe after all changes have been made
    return df



def wrangle_zillow():

   df = get_wrangle_zillow() # calling the original wrangle function to pull from the Zillow data

   df = handle_nulls(df) # calling function to get rid of nulls

   df = type_change(df) # calling change to integers for features that need it

   df = handle_outliers(df) # calling function to remove outliers in data

   df.to_csv('zillow.csv', index=False) # exporting the dataframe to a csv file

   return df # returning the df after wrangling and cleaning the data


def scale_data(train, validate, test, features_to_scale):
    
    # Fit the scaler to train data only
    scaler = MinMaxScaler()
    scaler.fit(train[features_to_scale])
    
    # Generate a list of the new column names with _scaled added on
    scaled_columns = [col+'_scaled' for col in features_to_scale]
    
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[features_to_scale])
    scaled_validate = scaler.transform(validate[features_to_scale])
    scaled_test = scaler.transform(test[features_to_scale])
    
    # Concatenate the scaled data to the original unscaled data
    train_scaled = pd.concat([train, pd.DataFrame(scaled_train,index=train.index, columns = scaled_columns)],axis=1)
    validate_scaled = pd.concat([validate, pd.DataFrame(scaled_validate,index=validate.index, columns = scaled_columns)],axis=1)
    test_scaled = pd.concat([test, pd.DataFrame(scaled_test,index=test.index, columns = scaled_columns)],axis=1)

    # returning the  split and scaled data
    return train_scaled, validate_scaled, test_scaled


