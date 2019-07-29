import numpy as np

def merge_log(model, seg_len):
    accs = []
    aucs = []
    time = []
    for i in range(10):
        with open(f'./log_2/{model}/{seg_len}{i+1}.log', 'r') as f:
            log = [v.rstrip('\n') for v in f.readlines()[-3:]]
        accs.append(float(log[0][10:]))
        aucs.append(float(log[1][10:]))
        time.append(float(log[2][14:]))

    with open(f'./log_2/{model}/{seg_len}all.log', 'w') as f:
        f.write(f'Test acc mean: {np.mean(accs)}\n')
        f.write(f'Test acc var: {np.std(accs)}\n')
        f.write(f'Test auc mean: {np.mean(aucs)}\n')
        f.write(f'Test auc var: {np.std(aucs)}\n')
        f.write(f'Execute time: {np.mean(time)}')

def main():
    model = 'segment-4-0'
    for i in range(3, 16):
        merge_log(model, f'{i}.')

if '__main__' == __name__:
    main()
