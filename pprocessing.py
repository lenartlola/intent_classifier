import pandas as pd
import pickle
import os

data_dir="atis"

def read_data(file="atis.train.pkl"):
    with open(os.path.join(data_dir, file), "rb") as path:
        dataset, dicts = pickle.load(path)
    #print("Loaded data from", file)
    #print(" samples: {:4d}".format(len(dataset["query"])))
    #print(" vocab_size: {:4d}".format(len(dicts["token_ids"])))
    #print(" slot count: {:4d}".format(len(dicts["slot_ids"])))
    #print(" intent count: {:4d}".format(len(dicts["intent_ids"])))

    return dataset, dicts

def create_dataframe(label, words, slots):
    df = pd.DataFrame(
            {
                "labels": label,
                "queries": words,
                "slots": slots,
            },
            columns = ["labels", "queries", "slots"]
        )
    return df


def pre_process():
    train_dataset, dicts = read_data("atis.train.pkl")
    test_dataset, dicts = read_data("atis.test.pkl")

    # Get all the ids
    word2idx, slot2idx, intent2idx = dicts["token_ids"], dicts["slot_ids"], dicts["intent_ids"]

    # Get the different components
    intent_train, query_train, slot_train = train_dataset["intent_labels"],train_dataset["query"],train_dataset["slot_labels"]
    intent_test, query_test, slot_test = test_dataset["intent_labels"], test_dataset["query"], test_dataset["slot_labels"]

    # Get components with their respective ids
    idx2word = {word2idx[i]:i for i in word2idx}
    idx2slot = {slot2idx[i]:i for i in slot2idx}
    idx2intent = {intent2idx[i]:i for i in intent2idx}

    # Create lists
    words_train = [ list(map(lambda x: idx2word[x], w)) for w in query_train]
    slots_train = [ list(map(lambda x: idx2slot[x], w)) for w in slot_train]
    labels_train = [ list(map(lambda x: idx2intent[x], y)) for y in intent_train]

    words_test = [ list(map(lambda x: idx2word[x], w)) for w in query_test]
    slots_test = [ list(map(lambda x: idx2slot[x], w)) for w in slot_test]
    labels_test = [ list(map(lambda x: idx2intent[x], y)) for y in intent_test]

    # Get labels for each
    y_train = [word[0] for word in labels_train]
    y_test = [word[0] for word in labels_test]

    df_train = create_dataframe(y_train, words_train, slots_train)
    df_test = create_dataframe(y_test, words_test, slots_test)

    df_train.labels = df_train.labels.apply(lambda x: x.split('+')[0] if '+' in x else x)
    df_test.labels = df_test.labels.apply(lambda x: x.split('+')[0] if '+' in x else x)
    y_train = df_train.labels
    y_test = df_test.labels
