import logging
import os
from DataPipeline.AzureDataPipeline import AzureSearchDataPipeline
import azure.functions as func 
from configs.config import data_cfg
logging.basicConfig(level=logging.INFO)


def main(mytimer: func.TimerRequest) -> None:
    logging.info("Timer triggered — starting pipeline...")

    pipeline = AzureSearchDataPipeline(data_cfg)
    pipeline.run_update()
    
    logging.info("Pipeline finished successfully")

