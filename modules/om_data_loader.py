import modules.om_logging as oml
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import hashlib
import json
import os
import csv

class OMDataLoader():
    def __init__(s):
        return
    
    def convert_date(s,date_str:str,date_format:str='%Y-%m-%d',header_row_value:str='date',c:str='a',d:str='b')->dt.datetime:
        if(date_str=='date'):
            return date_str
        try: 
            return dt.datetime.strptime(date_str, date_format)
        except ValueError:
            raise Exception(f"Invalid date [{date_str}]")
        except Exception as ex:
            oml.error(f"Invalid date [{date_str}]")
            return None
        
    def split_data(s,df,predict_col):
        train_data, eval_data = train_test_split(df, test_size=0.2, shuffle=False)
        oml.debug(f"split_data().train_data is {train_data}")
        train_data_features = train_data.drop(predict_col, axis=1).values
        train_data_target = train_data[predict_col].values
        eval_data_features = eval_data.drop(predict_col, axis=1).values
        eval_data_target = eval_data[predict_col].values
        np.set_printoptions(precision=2)
        return train_data_features, train_data_target, eval_data_features, eval_data_target
    
    def read_from_cache(s,searchUrl):
        url_hash = hashlib.md5(searchUrl.encode()).hexdigest()
        file_name=f"../data/housing/{url_hash}.json"
        with open(file_name, 'r', encoding='utf-8') as f:
            data=json.load(f)
        return data
    
    def write_to_local_cache_as_json(s,file_name:str,data):
        #url_hash = hashlib.md5(searchUrl.encode()).hexdigest()
        #file_name=f"../data/housing/{url_hash}.json"
        full_path=f"cache/{file_name}"
        if not os.path.exists(full_path):
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def write_to_local_cache_as_csv(s,file_name:str,data):
        #url_hash = hashlib.md5(searchUrl.encode()).hexdigest()
        #file_name=f"../data/housing/{url_hash}.json"
        full_path=f"cache/{file_name}"
        with open(full_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if(type(data) is np.ndarray): 
                # convert array into dataframe
                df=pd.DataFrame(data)                
                #df=df.tail(-1)
                #df=df.iloc[1:]
                #df.drop(index=df.index[0], axis=0, inplace=True)
                #oml.debug(f"df={df}")
                df.to_csv(full_path,index=False,header=False)
                #writer.writerows(data)
            else:
                df=pd.DataFrame(data)
                df.to_csv(full_path,index=False,header=False)
                #for i in range(len(data)):
                    #row = data[i]
                    #writer.writerow(row)

    def is_cached(s,file_name:str):
        #url_hash = hashlib.md5(searchUrl.encode()).hexdigest()
        #file_name=f"../data/housing/{url_hash}.json"
        full_path=f"cache/{file_name}"
        return os.path.exists(full_path)    
    
    def load_trade_data(s,data_path:str,data_file:str,predict_col:str):
        oml.debug("load_trade_data called!")
        data_file='data/stock_prices.csv'
        #target_col="future_quote" # remove the column that we will predict from the input data set
        df=s.load_file_as_pd(data_file,date_col="date",index_col="date",date_format='%y-%m-%d')
        return s.split_data(df,predict_col=predict_col)
    
    def load_fav_animal_data(s,data_path:str,data_file:str,predict_col:str):
        oml.debug("load_fav_animal_data() called!")
        file_path=os.path.join(data_path,data_file)
        df=s.load_file_as_pd(file_path)
        oml.debug("fav_animal_data after load()")
        print(df)
        return s.split_data(df,predict_col=predict_col)    

    def load_mnist_housing_data(s):
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        housing = fetch_california_housing()
        #print(housing)
        s.write_to_local_cache_as_csv("mnist_housing_feature_data.csv",housing['data'])
        s.write_to_local_cache_as_csv("mnist_housing_target_data.csv",housing['target'])
        s.write_to_local_cache_as_csv("mnist_housing_feature_names.csv",housing['feature_names'])
        X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        np.random.seed(42)
        tf.random.set_seed(42)
        return X_train, y_train, X_valid, y_valid    

    def load_file_as_ndarr(s,data_file:str, date_col:str=None, index_col:str=None,encoding:str="UTF8",header_index:int=0)->np.ndarray:
        df=s.load_file_as_pd(data_file, date_col, index_col,encoding,header_index)
        return df.values

    def load_file_as_pd(s,data_file:str, date_col:str=None, index_col:str=None,encoding:str="UTF8",header_index:int=0,date_format:str='%m/%d/%y')->pd.DataFrame:
        oml.debug("load_file_as_pd called!")
        df = pd.read_csv(data_file,encoding=encoding,header=header_index,skipinitialspace=True)
        # Strip leading spaces from column names
        if(date_col!=None): df[date_col] = df[date_col].apply(s.convert_date)
        if(index_col!=None): df.set_index(index_col, inplace=True)
        return df