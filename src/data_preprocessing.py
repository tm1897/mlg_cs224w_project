from pathlib import Path

import pandas as pd


class TrainTestGenerator:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def prepare_data(self):
        # Read the raw data
        df_user_artists = pd.read_table(self.data_dir / "hetrec2011-lastfm-2k" / "user_artists.dat")
        df_tagged = pd.read_table(self.data_dir / "hetrec2011-lastfm-2k" / "user_taggedartists-timestamps.dat")

        # Remove duplicate tags (keep first tag)
        df_tagged = df_tagged.groupby(["userID", "artistID", "tagID"])["timestamp"].min().reset_index()

        # Merge the datasets - to have weights and timestamps
        df = pd.merge(
            df_user_artists,
            df_tagged,
            on=["userID", "artistID"]
        ).drop_duplicates(subset=["userID", "artistID"])

        # Parse datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        # Filter data
        df = df[df["timestamp"].dt.year > 2000]
        df = df.reset_index(drop=True)

        return df

    def forward_chaining(self):
        data = self.prepare_data()

        for test_year in range(2008, 2010+1):
            train = data[data["timestamp"].dt.year < test_year]
            test = data[data["timestamp"].dt.year == test_year]

            yield test_year, train, test


