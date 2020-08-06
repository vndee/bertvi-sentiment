import os

with open(os.path.join('../outputs', 'submission.csv'), 'r') as stream:
    data = stream.read().split('\n')

    file_writer = open(os.path.join('../outputs', 'submiss.csv'), 'w+')
    for line in data:
        line = line.split(',')
        if line.__len__() < 3:
            file_writer.write('id,label\n')
        else:
            # print(line)
            file_writer.write((line[1] if line[1][:4] == 'test' else 'test_' + line[1]) + ',' + line[2] + '\n')
    file_writer.close()
