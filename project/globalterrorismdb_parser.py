import pandas as pd
import json
import os


class GlobalTerrorismDBParser():
    # Reads in the csv file and stores it as DataFrame object.
    def __init__(self):
        self.data_dir = "data"
        self.data_filename = "globalterrorismdb_0617dist.csv"
        self.data_path = os.path.join(self.data_dir, self.data_filename)

        self.data = pd.read_csv(self.data_path,
                                encoding="latin1",
                                low_memory=False)

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Must be instance of DataFrame, but got " + type(self.data) + ".")


    # Stores a json with the content of each column in the csv file.
    def to_json(self):
        for column in self.data.columns:
            with open(os.path.join(self.data_dir, column + ".json"), "w") as f:
                json.dump(obj=list(self.data.get(column)), fp=f, sort_keys=True, indent=4)


gt_parser = GlobalTerrorismDBParser()
gt_parser.to_json()
