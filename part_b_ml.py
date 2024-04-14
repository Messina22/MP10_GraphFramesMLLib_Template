from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
import pyspark.sql.functions as F

############################################
#### PLEASE USE THE GIVEN PARAMETERS     ###
#### FOR TRAINING YOUR KMEANS CLUSTERING ###
#### MODEL                               ###
############################################

NUM_CLUSTERS = 4
SEED = 0
MAX_ITERATIONS = 100
INITIALIZATION_MODE = "random"

sc = SparkContext()
sqlContext = SQLContext(sc)


def get_clusters(df, num_clusters, max_iterations, initialization_mode, seed):
    # TODO:
    # Use the given data and the cluster pparameters to train a K-Means model
    # Find the cluster id corresponding to data point (a car)
    # Return a list of lists of the titles which belong to the same cluster
    # For example, if the output is [["Mercedes", "Audi"], ["Honda", "Hyundai"]]
    # Then "Mercedes" and "Audi" should have the same cluster id, and "Honda" and
    # "Hyundai" should have the same cluster id
    kmeans = (
        KMeans()
        .setK(num_clusters)
        .setSeed(seed)
        .setMaxIter(max_iterations)
        .setInitMode(initialization_mode)
    )
    model = kmeans.fit(df)
    transformed = model.transform(df)
    clusters = transformed.groupBy("prediction").agg(F.collect_list("car_name"))
    return clusters.rdd.map(lambda x: x[1]).collect()


def parse_line(line):
    # TODO: Parse data from line into an RDD
    # Hint: Look at the data format and columns required by the KMeans fit and
    # transform functions
    parts = line.split(",")
    return parts[0], Vectors.dense([float(x) for x in parts[1:]])


if __name__ == "__main__":
    f = sc.textFile("dataset/cars.data")

    rdd = f.map(parse_line)

    # TODO: Convert RDD into a dataframe
    df = sqlContext.createDataFrame(rdd, ["car_name", "features"])

    clusters = get_clusters(df, NUM_CLUSTERS, MAX_ITERATIONS, INITIALIZATION_MODE, SEED)
    for cluster in clusters:
        print(",".join(cluster))
