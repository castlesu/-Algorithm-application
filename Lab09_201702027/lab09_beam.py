import numpy as np

def data_make(seed):
    data = []
    np.random.seed(seed)
    for i in range(5):
        data.append(np.random.random())
    data=np.array(data)
    return data

def greedy_search(data):
    index_data = []
    score = 1
    for i in range(10):
        max = np.argmax(data)
        index_data.append(max)
        score = score*data[max]
        seed = int(score*10)
        data = data_make(seed)

    greedy_data = [index_data,score]
    print("greedy search")
    print(greedy_data)
    return

def beam_search(data,k,seed1):
    sequences = [[list(),1]]
    for i in range(10):
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            seed = int(score *10)
            if (len(sequences) == 1):
                data = data_make(seed1)
            else :
                data = data_make(seed)
            for j in range(len(data)):
                candidate = [seq + [j],score*(data[j])]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates,key=lambda tup:tup[1],reverse=True)
        sequences = ordered[:k]

    print("beam serarch")
    for i in range(k):
        print(sequences[i])
    return

if __name__ == '__main__':
    seed = 9
    k=3
    data =data_make(seed)
    beam_search(data, k,seed)
    greedy_search(data)


