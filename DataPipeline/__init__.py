import logging
import azure.functions as func
from DataPipeline.pipeline import RagDataPipeline
from hydra import initialize, compose

def main(mytimer: func.TimerRequest) -> None:
    logging.info("Timer triggered — starting pipeline...")


    with initialize(config_path="../configs"):
        cfg = compose(config_name="config")

    pipeline = RagDataPipeline(config=cfg)
    pipeline.update_vectorstore()

    logging.info("Pipeline finished successfully")


