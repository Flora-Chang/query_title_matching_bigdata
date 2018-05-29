# encoding: utf-8
import tensorflow as tf
import pandas as pd
import numpy as np

def test_pointwise(sess, model, testing_set, filename=None):
    total_data = 0
    right = 0
    top = 0
    precision_bottom = 0
    recall_bottom = 0
    cnt = 0
    flag = 1
    for batch_data in testing_set.next_batch():
        queries, titles, labels,_ = batch_data
        if flag == 1:
            flag = 0

        fd = {model.train_query: queries,
              model.train_title: titles,
              model.train_labels: labels}

        #res = sess.run([model.score], fd)
        res = sess.run([model.eval_predictions], fd)
        res = list(zip(queries, titles, labels, res[0].tolist()))
        df = pd.DataFrame(res, columns=['query', 'title', 'label', 'score'])
        total_data += len(df)
        for i in range(0, len(df)):
            label = df['label'][i]
            score = df['score'][i]
            '''
            if label == 1:
                recall_bottom += 1
            if score >= 0.5:
                precision_bottom += 1
            if label ==1 and score >= 0.5:
                top += 1
            if (label == 0 and score < 0.5) or (label == 1 and score >= 0.5):
                right += 1
            '''
            if label >= 1:
                recall_bottom += 1
            if score >=1:
                precision_bottom += 1
            if label >=1 and score>=1:
                top += 1
            if label == score:
                right += 1

        if filename is not None:
            if cnt ==0:
                df.to_csv(filename, mode='a', index=False)
                cnt += 1
            else:
                df.to_csv(filename, mode='a', index=False, header=False)
    print("accuracy：", right / (total_data + 0.001))
    print("precision:",top / (precision_bottom + 0.001))
    print("recall:", top / (recall_bottom + 0.001))
    print("=" * 60)
    return right / (total_data + 0.001), top / (precision_bottom + 0.001), top / (recall_bottom + 0.001)



def test_pairwise(sess, model, testing_set, filename=None):
    total_data = 0
    right = 0

    cnt = 0
    flag = 1
    for batch_data in testing_set.next_batch():
        queries, titles, labels,_ = batch_data
        if flag == 1:
            flag = 0

        fd = {model.train_query: queries,
              model.train_title: titles,
              model.train_labels: labels}

        #res = sess.run([model.score], fd)
        res = sess.run([model.eval_score], fd)
        res = np.array(res[0])
        res = res.reshape([-1, 2])
        total_data += len(res)
        for i in range(0, len(res)):
            pos_score = res[i][0]
            neg_score = res[i][1]
            #total_data += 1
            if pos_score > neg_score:
                right += 1
        '''
        if filename is not None:
            if cnt ==0:
                df.to_csv(filename, mode='a', index=False)
                cnt += 1
            else:
                df.to_csv(filename, mode='a', index=False, header=False)
        '''
    print("accuracy：", right / (total_data + 0.001))
    print("right: ", right, "total_data: ", total_data)
    print("=" * 60)
    return right / (total_data + 0.001)


