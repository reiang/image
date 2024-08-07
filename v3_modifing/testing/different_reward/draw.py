import csv
import matplotlib.pyplot as plt
import os
import ast

def read_algorithm_data(filename, algorithm_name):
    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['algorithm'] == algorithm_name:
                return row
    return None

data_dir = '111'
fieldnames = [
    'algorithm', 'h0_reward', 'h1_reward', 'reward', 'bpp', 'mse',
    'cover_scores', 'message_density', 'uiqi', 'rs', 'psnr', 'ssim', 'consumption','generated_scores'
    ]
filename = 'data111.csv'

algs = ['Basic', 'Origin', 'Residual', 'Dense']
data_set = {alg: {fieldname: [] for fieldname in fieldnames} for alg in algs}

for alg in algs:
    data = read_algorithm_data(filename, alg)
    if data is not None:
        for fieldname in fieldnames:
            if fieldname == 'algorithm':
                data_set[alg][fieldname] = alg
            else:
                data_str = data.get(fieldname, "")
                try:
                    data_list = ast.literal_eval(data_str) if data_str else []
                except (ValueError, SyntaxError):
                    data_list = []
                data_set[alg][fieldname] = data_list
    else:
        print(f"No data of algorithm {alg}!!!")
        continue


for y_label in fieldnames:
    if y_label == 'algorithm':
        continue
    plt.figure()
    for alg in algs:
        plt.plot(data_set[alg][y_label], label=alg)
    plt.xlabel('Time slot')
    plt.ylabel(y_label)
    plt.legend()
    plt.title(data_dir.upper())
    plt.tight_layout()
    plt.savefig(os.path.join('.', 'data', data_dir, '111All_' + y_label + '.png'))
    plt.close()