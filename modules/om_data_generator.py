import csv
import random 
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


class OMDataGenerator():
    def generate_fav_animal_data(s,data_path:str,file_name:str,num_rows:int=10000):
        header = ['age', 'gender', 'handedness', 'fav_animal']
        rows = []
        for i in range(num_rows):
            age = random.randint(1, 100)
            gender = random.choice(['male', 'female'])
            handedness = random.choice(['left', 'right'])
            
            if gender == 'male' and handedness == 'right' and age > 30:
                fav_animal = 'dog'
            elif gender == 'male' and handedness == 'right' and age <= 30:  
                fav_animal = 'cat'
            elif gender == 'female' and handedness == 'right' and age > 30:
                fav_animal = 'tiger'
            elif gender == 'female' and handedness == 'right' and age <= 30:
                fav_animal = 'bunny'
            elif gender == 'male' and handedness == 'left' and age > 50:
                fav_animal = 'elephant'
            elif gender == 'male' and handedness == 'left' and age <= 50:
                fav_animal = 'giraffe'
            rows.append([age, gender, handedness, fav_animal])

        with open(os.path.join(data_path,file_name), 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f) 
            writer.writerow(header)
            writer.writerows(rows)
        return

    def prepare_fav_animal_data(s,data_path:str,file_name:str,prepared_file_name:str):
        df=pd.read_csv(os.path.join(data_path,file_name),encoding='utf-8')
        #X = df.drop('fav_animal', axis=1) 
        #y = df['fav_animal']
        #scaler = StandardScaler()
        #X_scaled = scaler.fit_transform(X)
        categorical = ['gender', 'handedness', 'fav_animal']
        df[categorical] = df[categorical].astype('category')
        df[categorical] = df[categorical].apply(lambda x: x.cat.codes)
        print("df cats")
        print(df)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        print("numeric cols")
        print(numeric_cols)
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print("final df")
        print(df)
        df.to_csv(os.path.join(data_path,prepared_file_name), index=False, encoding='utf-8')

        #df_prepared = pd.DataFrame(X_scaled, columns=X.columns)
        #df_prepared['fav_animal'] = y
        #df_prepared.to_csv('prepared_animal_data.csv', index=False, encoding='utf-8')
