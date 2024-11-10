import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils import Utils

class ETAModel():
    def __init__(self, file_path, keyword, show_loss=False) -> None:
        self.file_path = file_path
        self.keyword = keyword
        self.show_loss = show_loss

    def process_hope_production(self):
        skip_rows = Utils.find_header_row(self.file_path, keyword=self.keyword, sheet_name="HOPE PRODUCTION")
        df = pd.read_excel(self.file_path, sheet_name="HOPE PRODUCTION", skiprows=skip_rows)
        df = df.dropna(subset=['BATCH no.']) # removing NA values in batch numbers
        df["start_time"] = None
        for index, row in df.iterrows():
            start_time = row[1] + timedelta(hours=np.random.randint(9,11), minutes=np.random.randint(0,60), seconds=np.random.randint(0,60))
            df.at[index, 'start_time'] = start_time
        df['start_time'] = pd.to_datetime(df['start_time'])

        skip_rows2 = Utils.find_header_row(file_path=self.file_path, keyword=self.keyword, sheet_name="HOPE STORAGE AFTER COOKING")
        df2 = pd.read_excel(self.file_path, sheet_name="HOPE STORAGE AFTER COOKING", skiprows=skip_rows2)
        df2 = df2.dropna(subset=['BATCH no.'])
        df2["end_time"] = None
        for index, row in df2.iterrows():
            end_time = row[1] + timedelta(hours=np.random.randint(13,16), minutes=np.random.randint(0,60), seconds=np.random.randint(0,60))
            df2.at[index, 'end_time'] = end_time
        df2['end_time'] = pd.to_datetime(df2['end_time'])

        df_sheet1 = df[["BATCH no.", "PRODUCTION DATE", "BATCH WEIGHT (kg) BEFORE COOKING", "BATCH WEIGHT (kg) AFTER COOKING", "start_time"]]
        df_sheet2 = df2[["BATCH no.", "BATCH INTO STORAGE", "end_time"]]

        merged_df = pd.merge(
            df_sheet1,
            df_sheet2,
            left_on = ["BATCH no.", "PRODUCTION DATE"],
            right_on = ["BATCH no.", "BATCH INTO STORAGE"],
            how = "inner"
        )
        merged_df["production_type"] = 0
        return merged_df

    def process_faith_production(self):
        skip_rows = Utils.find_header_row(self.file_path, keyword=self.keyword, sheet_name="FAITH PRODUCTION")
        df = pd.read_excel(self.file_path, sheet_name="FAITH PRODUCTION", skiprows=skip_rows)
        df = df.dropna(subset=['BATCH no.'])
        df["start_time"] = None
        for index, row in df.iterrows():
            start_time = row[1] + timedelta(hours=np.random.randint(9,11), minutes=np.random.randint(0,60), seconds=np.random.randint(0,60))
            df.at[index, 'start_time'] = start_time
        df['start_time'] = pd.to_datetime(df['start_time'])

        skip_rows2 = Utils.find_header_row(file_path=self.file_path, keyword=self.keyword, sheet_name="FAITH STORAGE AFTER COOKING")
        df2 = pd.read_excel(self.file_path, sheet_name="FAITH STORAGE AFTER COOKING", skiprows=skip_rows2)
        df2 = df2.dropna(subset=['BATCH no.'])
        df2["end_time"] = None
        for index, row in df2.iterrows():
            end_time = row[1] + timedelta(hours=np.random.randint(13,16), minutes=np.random.randint(0,60), seconds=np.random.randint(0,60))
            df2.at[index, 'end_time'] = end_time
        df2['end_time'] = pd.to_datetime(df2['end_time'])

        df_sheet1 = df[["BATCH no.", "PRODUCTION DATE", "BATCH WEIGHT (kg) BEFORE COOKING", "BATCH WEIGHT (kg) AFTER COOKING", "start_time"]]
        df_sheet2 = df2[["BATCH no.", "BATCH INTO STORAGE", "end_time"]]

        merged_df = pd.merge(
            df_sheet1,
            df_sheet2,
            left_on = ["BATCH no.", "PRODUCTION DATE"],
            right_on = ["BATCH no.", "BATCH INTO STORAGE"],
            how = "inner"
        )
        merged_df["production_type"] = 1
        return merged_df

    def remove_outlier(self, df, column_name, lower_bound=None, upper_bound=None):
        lower_bound = df[column_name].quantile(lower_bound)
        upper_bound = df[column_name].quantile(upper_bound)
        final_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
        return final_df
        
    def merge_df(self, df1, df2):
        return pd.concat([df1, df2], ignore_index=True)
    
    def build_time_data(self, df):
        final_df = df[["BATCH no.", "BATCH WEIGHT (kg) BEFORE COOKING", "start_time", "end_time", "production_type"]]
        final_df['start_time'] = pd.to_datetime(final_df['start_time'])
        final_df['end_time'] = pd.to_datetime(final_df['end_time'])

        final_df['time_elapsed'] = (final_df['end_time'] - final_df['start_time']).dt.total_seconds() / 60

        final_df['start_hour'] = final_df['start_time'].dt.hour
        final_df['start_minute'] = final_df['start_time'].dt.minute
        final_df['day_of_week'] = final_df['start_time'].dt.dayofweek
        return final_df
    
    def split_data(self, df, features, target):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X, y, X_train, X_test, y_train, y_test

    def train_test_model(self, X, y, X_train, X_test, y_train, y_test):

        model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                objective='reg:squarederror'
            )
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        if self.show_loss:
            print(f'Mean Absolute Error on Test Set: {mae:.2f} minutes')

        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        if self.show_loss:
            print(f'Mean Cross-Validation MAE: {-np.mean(scores):.2f} minutes')

        return model

    def predict(self, model, weight, start_time, production_type):
        start_time = pd.to_datetime(start_time)
        start_hour = start_time.hour
        start_minute = start_time.minute
        day_of_week = start_time.dayofweek
        
        new_data = pd.DataFrame({
            'BATCH WEIGHT (kg) BEFORE COOKING': [weight],
            'start_hour': [start_hour],
            'start_minute': [start_minute],
            'day_of_week': [day_of_week],
            'production_type': [production_type]
        })
        
        predicted_duration = model.predict(new_data)[0]
        print(f"Predicted duration: {Utils.convert_minutes_to_hours(predicted_duration)}")

        eta = start_time + pd.Timedelta(minutes=predicted_duration)
        print(f"Calculated ETA: {eta}")

        current_time = pd.Timestamp.now()
        
        time_diff = eta - current_time
        if time_diff < pd.Timedelta(0):
            time_diff = -time_diff  # Make the time difference positive

        if time_diff < pd.Timedelta(hours=1):
            minutes_left = int(time_diff.total_seconds() / 60)
            time_left = f'{minutes_left} minutes'
        else:
            hours_left = int(time_diff.total_seconds() // 3600)
            minutes_left = int((time_diff.total_seconds() % 3600) // 60)
            time_left = f'{hours_left} hours {minutes_left} minutes'

            print(f'Time left: {time_left}')

    def save_model(self, model, model_name):
        model.save_model(model_name)
        print(f"Model saved as {model_name}")

    def save_joblib_model(self, model, model_name):
        joblib.dump(model, model_name)
    

if __name__ == "__main__":
    model = ETAModel("HKFoods.xlsx", "BATCH no.")
    hope_df = model.process_hope_production()
    hope_df = model.remove_outlier(hope_df, "BATCH WEIGHT (kg) BEFORE COOKING", 0.01, 0.99)
    
    faith_df = model.process_faith_production()
    faith_df = model.remove_outlier(faith_df, "BATCH WEIGHT (kg) BEFORE COOKING", 0.01, 0.99)
    
    merged_df = model.merge_df(hope_df, faith_df)
    dummy_time_df = model.build_time_data(merged_df)

    features = ['BATCH WEIGHT (kg) BEFORE COOKING', 'start_hour', 'start_minute', 'day_of_week', "production_type"]
    target = ["time_elapsed"]
    X, y, X_train, X_test, y_train, y_test = model.split_data(dummy_time_df, features, target)

    trained_model = model.train_test_model(X, y, X_train, X_test, y_train, y_test)

    model.predict(trained_model, 190, '2024-11-09 10:30:00', 0)

    #.save_model(trained_model, "eta_model")

    model.save_joblib_model(trained_model, "eta_model.joblib")
