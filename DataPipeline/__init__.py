# import logging
# import azure.functions as func
# from data_pipeline.pipeline import RagDataPipeline
# from omegaconf import DictConfig

# @hydra.main(config_path="../configs", config_name="config", version_base=None)
# def load_config(cfg: DictConfig) -> DictConfig:
#     return cfg

# def main(mytimer: func.TimerRequest) -> None:
#     logging.info("Timer triggered — starting pipeline...")
    
#     # Load Hydra config once
#     config = load_config()  # now Hydra gives you a DictConfig
    
#     # Run your pipeline
#     pipeline = RagDataPipeline(config=config)
#     print(config)
#     pipeline.scan()

#     logging.info("Pipeline finished successfully")



# import logging
# import azure.functions as func
# from data_pipeline.pipeline import RagDataPipeline
# import hydra
# from omegaconf import DictConfig

# # Hydra entrypoint (NOT the Azure function)
# @hydra.main(config_path="../configs", config_name="config", version_base=None)
# def hydra_entry(cfg: DictConfig) -> None:
#     pipeline = RagDataPipeline(config=cfg)
#     pipeline.scan()

# def main(mytimer: func.TimerRequest) -> None:
#     logging.info("Timer triggered — starting pipeline...")

#     # Instead of Hydra decorating main(), call Hydra programmatically
#     hydra_entry()

#     logging.info("Pipeline finished successfully")

import logging
import azure.functions as func
from DataPipeline.pipeline import RagDataPipeline
from hydra import initialize, compose

def main(mytimer: func.TimerRequest) -> None:
    logging.info("Timer triggered — starting pipeline...")

    # Load Hydra configs programmatically
    with initialize(config_path="../configs"):
        cfg = compose(config_name="config")

    # Run your pipeline
    pipeline = RagDataPipeline(config=cfg)
    pipeline.update_vectorstore()

    logging.info("Pipeline finished successfully")


