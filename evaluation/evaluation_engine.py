import pandas as pd 

def get_recall_K(
    number_of_relevent_assessments_in_top_k,
    total_relevent_assessments_for_the_query
):
    return number_of_relevent_assessments_in_top_k / \
            total_relevent_assessments_for_the_query
