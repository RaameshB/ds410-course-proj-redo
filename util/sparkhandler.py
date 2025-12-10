from pyspark.sql import SparkSession
class SparkHandler:
    def __init__(self, driver_mem = 4, executor_mem = 4, mem_overhead = 4, available_cores=None):
        if available_cores is not None:
            self.spark_session = SparkSession.builder.appName("DataProc")\
                .config("spark.executor.cores", f"{available_cores}")\
                .config("spark.driver.memory", f"{driver_mem}g")\
                .config("spark.executor.memory", f"{executor_mem}g")\
                .config("spark.executor.memoryOverhead", f"{mem_overhead}g")\
                .getOrCreate()
        else:
            self.spark_session = SparkSession.builder.appName("DataProc")\
                .config("spark.driver.memory", f"{driver_mem}g")\
                .config("spark.executor.memory", f"{executor_mem}g")\
                .config("spark.executor.memoryOverhead", f"{mem_overhead}g")\
                .getOrCreate()
        self.spark_context = self.spark_session.sparkContext

    def get_spark_session(self):
        return self.spark_session

    def get_spark_context(self):
        return self.spark_context

    def stop(self):
        self.spark_session.stop()