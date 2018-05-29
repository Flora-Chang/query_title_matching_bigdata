import sys

with open(sys.argv[1], 'r') as f:
    dict_label = {}
    count = 0.0
    flag = 0
    for line in f:
        line = line.strip().split('\t')
        try:
            label = int(line[2])
        except Exception as e:
            flag = 1
            print(line)
        if flag == 0:
            dict_label[label] = dict_label.get(label, 0) + 1
            count += 1
    for key in dict_label:
        print(key, dict_label[key]/count)
'''
    (0, 0.07827682272202215)
    (1, 0.13297547054426853)
    (2, 0.5173608751783652)
    (3, 0.27138683155534415)
'''

