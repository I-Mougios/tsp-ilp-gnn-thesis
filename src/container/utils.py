from pydantic import BaseModel


class TSPRecord(BaseModel):
    number_of_points: int
    distance_metric: str
    coordinates: list[tuple[float, float]]
    decision_variables: list[int]
    optimal_tour: list[int]
    minimum_distance: float
    subtour_revisions: int
    elapsed_time: float


def insert_one(collection, tsp_record: TSPRecord):
    try:
        collection.insert_one(tsp_record.model_dump())
    except Exception as e:
        print(e)
        raise e
