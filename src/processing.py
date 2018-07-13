import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer


def get_data_from_json(path):
    """Reads json files, and returns arrays of cuisines and ingredients."""

    train_data = os.path.join(path, "train.json")
    test_data = os.path.join(path, "test.json")

    recipes = pd.read_json(train_data).set_index('id')
    ingredients = recipes.ingredients.str.join(' ')

    cv = CountVectorizer()
    cv.fit(ingredients)
    num_unique_ingredients = len(cv.get_feature_names())
    num_recipes = len(recipes)
    train_x = pd.DataFrame(cv.transform(ingredients).todense())
    train_y = recipes.cuisine
    num_cuisines = recipes.cuisine.nunique()
    print "{0} unique ingredients used in {1} recipes from {2} different cuisines around the world.". \
        format(num_unique_ingredients, num_recipes, num_cuisines)

    recipes_test = pd.read_json(test_data).set_index('id')
    test_x = cv.transform(recipes_test.ingredients.str.join(' '))
    index = recipes_test.index
    return train_x, train_y, test_x, index


def write_predictions_to_file(index, test_y, path):
    """Writes the predictions on test data to a .csv file in suitable format for submission."""

    submission_df = pd.Series(test_y, index=index, name='cuisine')
    submission_df.to_csv(path=os.path.join(path, "submission.csv"), header=True)
