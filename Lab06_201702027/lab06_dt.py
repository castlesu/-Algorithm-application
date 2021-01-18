import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import export_graphviz
import pydot
import re


def File_Open(path):
    data = pd.read_csv(path)

    return data

def Data_Pro(train, test):

    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    common_value = 'S'
    genders = {"male": 0, "female": 1}
    ports = {"S": 0, "C": 1, "Q": 2}

    data = [train, test]

    for dataset in data:
        dataset['Cabin'] = dataset['Cabin'].fillna("U0")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(int)

        mean = train["Age"].mean()
        std = test["Age"].std()
        is_null = dataset["Age"].isnull().sum()
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        dataset["Age"] = age_slice
        dataset["Age"] = train["Age"].astype(int)

        dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
        dataset['Embarked'] = dataset['Embarked'].map(ports)

        dataset['Fare'] = dataset['Fare'].fillna(0)
        dataset['Fare'] = dataset['Fare'].astype(int)

        dataset['Sex'] = dataset['Sex'].map(genders)



    train["Age"].isnull().sum()
    train = train.drop(['Name','Ticket','Cabin'], axis=1)
    test = test.drop(['Name','Ticket','Cabin'], axis=1)

    # print(train,test)

    #---------data change
    data = [train, test]
    for dataset in data:
        dataset['Age'] = dataset['Age'].astype(int)
        dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
        dataset.loc[dataset['Age'] > 66, 'Age'] = 6

        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare'] = 3
        dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare'] = 4
        dataset.loc[dataset['Fare'] > 250, 'Fare'] = 5
        dataset['Fare'] = dataset['Fare'].astype(int)

    return train, test

def Draw_Graph(train, test):

    X_train = train.drop("Survived", axis=1)
    Y_train = train["Survived"]
    X_test = test.copy()

    train_dt = tree.DecisionTreeClassifier()
    train_dt.fit(X_train, Y_train)
    Y_test = train_dt.predict(X_test)

    dt = tree.DecisionTreeClassifier()
    dt.fit(X_test,Y_test)

    print('Score: {}'.format(dt.score(X_test,Y_test)))
    features = test.columns.values.tolist()

    export_graphviz(dt, out_file="dt.dot", class_names=['No','Yes'],
                                    feature_names=features, impurity=False,filled=True)
    (graph, ) = pydot.graph_from_dot_file('dt.dot', encoding='utf8')
    graph.write_png('tree.png')

    return X_test, Y_test

def CSV_Save(data, survived):
    csv =[[]for i in range(2)]
    csv[0] = data["PassengerId"]
    csv[1] = survived
    csv = np.array(csv)
    csv=csv.T
    # print(csv)

    csv_df = pd.DataFrame(csv)
    csv_df.rename(columns={0:'PassengerId',1:'Survived'},inplace=True)
    # csv_df.reset_index(drop=True,inplace=False)
    csv_df.to_csv('test_tree.csv',index=False)
    # print(csv_df)

if __name__ == '__main__':
    train_df =File_Open('train.csv')
    test_df = File_Open('test.csv')
    train,test = Data_Pro(train_df, test_df)
    test,test_survived = Draw_Graph(train,test)
    CSV_Save(test,test_survived)

