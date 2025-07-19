import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from tableauhyperapi import HyperProcess, Telemetry, Connection, TableName
from pathlib import Path
import tempfile
import logging
import os

logger = logging.getLogger(__name__)

class TdsxReader:
    def __init__(self, tdsx_path):
        self.tdsx_path = Path(tdsx_path)
        self.temp_dir = tempfile.TemporaryDirectory()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.temp_dir.cleanup()

    def _extract_tdsx(self):
        logger.info(f"Extracting TDSX file: {self.tdsx_path}")
        with zipfile.ZipFile(self.tdsx_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir.name)
        logger.info(f"Extracted to: {self.temp_dir.name}")

    def get_schema(self):
        self._extract_tdsx()
        tds_file = self._find_tds_file()
        if not tds_file:
            raise ValueError("No .tds file found in the archive.")
        
        tree = ET.parse(tds_file)
        root = tree.getroot()
        
        columns = []
        for column in root.findall('.//column'):
            columns.append({
                'name': column.get('name'),
                'datatype': column.get('datatype'),
                'role': column.get('role'),
            })
        return columns

    def _find_tds_file(self):
        for root, _, files in os.walk(self.temp_dir.name):
            for file in files:
                if file.endswith(".tds"):
                    return os.path.join(root, file)
        return None

    def _find_hyper_file(self):
        for root, _, files in os.walk(self.temp_dir.name):
            for file in files:
                if file.endswith(".hyper"):
                    return os.path.join(root, file)
        return None

    def get_dataframe(self):
        self._extract_tdsx()
        hyper_file = self._find_hyper_file()
        if not hyper_file:
            raise ValueError("No .hyper file found in the archive.")

        with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
            with Connection(hyper.endpoint, hyper_file) as connection:
                table_name = connection.catalog.get_table_names()[0]
                table = connection.catalog.get_table_definition(table_name)
                data = connection.execute_list_query(f"SELECT * FROM {table.table_name}")
                
                columns = [col.name.value.unescaped for col in table.columns]
                
                df = pd.DataFrame(data, columns=columns)
                return df

def read_tdsx_schema(tdsx_path):
    with TdsxReader(tdsx_path) as reader:
        return reader.get_schema()

def read_tdsx_dataframe(tdsx_path):
    with TdsxReader(tdsx_path) as reader:
        return reader.get_dataframe()
