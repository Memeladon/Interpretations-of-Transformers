from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

def probe_task(X_layers, y, task_type="style"):
    results = {}
    for i, X in enumerate(X_layers):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if task_type == "style":
            score = accuracy_score(y_test, y_pred)
        else:  # semantic
            score = -mean_squared_error(y_test, y_pred)

        results[i] = score
    return results
