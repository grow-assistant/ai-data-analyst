import pandas as pd
import litellm
from .prompt import SCHEMA_PROFILING_PROMPT
import json

class Profiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_row_count(self):
        return len(self.df)

    def get_null_percentages(self):
        return (self.df.isnull().sum() / len(self.df) * 100).to_dict()

    def get_cardinality(self):
        return self.df.nunique().to_dict()

    def get_top_values(self, top_n=5):
        top_values = {}
        for col in self.df.columns:
            top_values[col] = self.df[col].value_counts().nlargest(top_n).index.tolist()
        return top_values

    def profile_dataframe(self):
        return {
            "row_count": self.get_row_count(),
            "null_percentage": self.get_null_percentages(),
            "cardinality": self.get_cardinality(),
            "top_values": self.get_top_values()
        }

def profile_dataframe(df: pd.DataFrame):
    profiler = Profiler(df)
    return profiler.profile_dataframe()

def get_semantic_profile(profile: dict):
    prompt = SCHEMA_PROFILING_PROMPT.format(profile=json.dumps(profile, indent=2))
    
    response = litellm.completion(
        model="gemini/gemini-pro",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def full_profile(df: pd.DataFrame):
    statistical_profile = profile_dataframe(df)
    semantic_profile = get_semantic_profile(statistical_profile)
    
    return {
        "statistical": statistical_profile,
        "semantic": semantic_profile
    }
