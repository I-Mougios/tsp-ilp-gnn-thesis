import sys
import os
from pathlib import Path
project_dir = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_dir))
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm
from utils import TSPRecord, insert_one

from tsp import TSP, generate_clustered_tsp_data

project_dir = Path(__file__).parent.parent.resolve()
env_path = project_dir / "container" / "mongo.env"
load_dotenv(env_path)
mongo_username = os.getenv("MONGO_INITDB_ROOT_USERNAME", "root")
mongo_password = os.getenv("MONGO_INITDB_ROOT_PASSWORD", "secret")

# Both application and MongoDB run on docker container
# then I just use the name of the service and no need to expose any port
# Application run in container and the MongoDb run locally then
#  I have to use host.docker.internal as a special DNS that resolves the host's machine IP address
# (no need to expose any port since application is the client connecting out to MongoDb)

# When MongoDB run in container but application run in my local machine then
# I have to expose the port 27017 so client(application) can send request to MongoDb
# which is hosted within a docker container
mongo_uri = f"mongodb://{mongo_username}:{mongo_password}@localhost:27017"


# MongoDB setup
client = MongoClient(mongo_uri)  # Create MongoDB client
db_name = "tsp_database"  # Database name
db = client[db_name]  # Select database
tsp_collection = db["tsp_solutions"]  # Select collection

number_of_uniform_instances = 5000
number_of_clustered_instances = 5000

for size in [20, 35, 50]:

    # Uniform instances
    for i in tqdm(range(number_of_uniform_instances), desc="Processing uniform instances"):
        coords = np.random.uniform(low=0, high=100, size=(size, 2))

        tsp = TSP(coords)
        tsp.solve(verbose=False)
        tsp_record = TSPRecord.model_validate(tsp.to_dict())
        insert_one(collection=tsp_collection, tsp_record=tsp_record)

    # Clustered instances
    for i in tqdm(range(number_of_clustered_instances), desc="Processing clustered instances"):
        num_of_clusters = np.random.randint(low=2, high=6)
        cluster_std = np.random.randint(low=2, high=5)

        coords = generate_clustered_tsp_data(
            num_cities=size,
            num_clusters=num_of_clusters,
            cluster_std=cluster_std,
        )

        tsp = TSP(coords)
        tsp.solve(verbose=False)
        tsp_record = TSPRecord.model_validate(tsp.to_dict())
        insert_one(collection=tsp_collection, tsp_record=tsp_record)
