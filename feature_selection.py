import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


data_dir = "/home/paul/workspace/data-science/my-projects/data/diabetes/"


def load_data():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data = pd.read_csv(data_dir + "diabetes.csv")
    print("Data has been loaded...")
    return data


def main():
    df = load_data()
    # split data inot X and y
    X = df.drop(["Outcome"], axis=1)
    y = df["Outcome"]

    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    clf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, class_weight="balanced", max_depth=5
    )

    # define Boruta feature selection method
    feat_selector = BorutaPy(clf, n_estimators="auto", random_state=23)

    # find all relevant features - 5 features should be selected
    feat_selector = feat_selector.fit(X.values, y.values)

    # number of selected features
    print("\n Number of selected features: ")
    print(feat_selector.n_features_)

    # setting up a dataframe for ranking features
    feature_df = pd.DataFrame(X.columns.tolist(), columns=["features"])
    feature_df["rank"] = feat_selector.ranking_
    feature_df = feature_df.sort_values("rank", ascending=True).reset_index(
        drop=True
    )  # noqa

    print("\n Top %d features:" % feat_selector.n_features_)
    print(feature_df.head(feat_selector.n_features_))

    # save feature ranking on a csv file
    feature_df.to_csv(data_dir + "boruta-feature-ranking.csv", index=False)


if __name__ == "__main__":
    main()
