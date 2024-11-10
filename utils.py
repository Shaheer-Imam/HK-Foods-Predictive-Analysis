import pandas as pd

class Utils():

    @staticmethod
    def find_header_row(file_path, keyword, sheet_name):
        return_index = None
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        for index, row in df.iterrows():
            if row.astype(str).str.contains(keyword, case=False, na=False).any():
                return_index = index
                break
        return return_index
    
    @staticmethod
    def convert_minutes_to_hours(minutes):
        minutes = int(minutes)

        if minutes >= 60:
            hours = int(minutes // 60)
            remaining_minutes = int(minutes % 60)
            return f'{hours} hours {remaining_minutes} minutes'
        else:
            return f'{int(minutes)} minutes'