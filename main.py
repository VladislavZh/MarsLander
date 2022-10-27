from scripts import PipelineMarsLanderDQN, PipelineMarsLanderAC

pipeline = PipelineMarsLanderAC()
#pipeline = PipelineMarsLanderDQN()
pipeline.execute_pipeline()
