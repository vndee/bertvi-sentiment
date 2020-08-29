import os

data = 'Hotel'
sign = 'train'

if __name__ == '__main__':
    with open(os.path.join('data', 'VLSP2018', f'VLSP2018-SA-{data}-{sign}.txt'), mode='r', encoding='utf-8-sig') as stream:
        file = stream.read()

    file_writer = open(os.path.join('data', 'VLSP2018', f'VLSP2018-SA-{data}-{sign}.prod'), mode='w+', encoding='utf-8-sig')
    file = file.split('\n\n')
    for lines in file:
        lines = lines.split('\n')
        label = [line.strip() for line in lines[2].split('},')]

        for x in label:
            if x[0] == '{':
                x = x[1:]

            if x[-1] == '}':
                x = x[:-1]
            file_writer.write(f'{lines[1]}\n{x}\n\n')

    file_writer.close()
