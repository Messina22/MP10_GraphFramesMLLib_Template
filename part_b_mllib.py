from pyspark import SparkContext
from numpy import array
from pyspark.mllib.clustering import KMeans, KMeansModel

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


def get_clusters(
    data_rdd,
    num_clusters=NUM_CLUSTERS,
    max_iterations=MAX_ITERATIONS,
    initialization_mode=INITIALIZATION_MODE,
    seed=SEED,
):
    # TODO:
    # Use the given data and the cluster pparameters to train a K-Means model
    # Find the cluster id corresponding to data point (a car)
    # Return a list of lists of the titles which belong to the same cluster
    # For example, if the output is [["Mercedes", "Audi"], ["Honda", "Hyundai"]]
    # Then "Mercedes" and "Audi" should have the same cluster id, and "Honda" and
    # "Hyundai" should have the same cluster id
    model = KMeans.train(
        data_rdd.map(lambda x: x[1]),
        num_clusters,
        maxIterations=max_iterations,
        initializationMode=initialization_mode,
        seed=seed,
    )
    clusters = data_rdd.map(lambda x: (model.predict(x[1]), x[0]))
    return clusters.groupByKey().map(lambda x: x[1]).collect()


def parse_line(line):
    parts = line.split(",")
    car_name = parts[0]
    features = array([float(x) for x in parts[1:]])
    return car_name, features


if __name__ == "__main__":
    f = sc.textFile("dataset/cars.data")

    # TODO: Parse data from file into an RDD

    data_rdd = f.map(parse_line)
    clusters = get_clusters(data_rdd)

    for cluster in clusters:
        print(",".join(cluster))
