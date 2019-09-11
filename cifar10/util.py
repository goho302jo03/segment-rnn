import numpy as np

def merge_log(model, seg_len):
    accs = []
    aucs = []
    epoch = []
    time = []
    wtime = []
    for i in range(10):
        with open(f'./log/{model}/{seg_len}{i+1}.log', 'r') as f:
            log = [v.rstrip('\n') for v in f.readlines()[-5:]]
        if 'resample' == model[:8]:
            accs.append(float(log[-5][10:]))
            aucs.append(float(log[-4][10:]))
            epoch.append(int(log[-3][15:]))
            time.append(float(log[-2][14:]))
            wtime.append(float(log[-1][18:]))
        else:
            accs.append(float(log[-4][10:]))
            aucs.append(float(log[-3][10:]))
            epoch.append(int(log[-2][15:]))
            time.append(float(log[-1][14:]))

    with open(f'./log/{model}/{seg_len}all.log', 'w') as f:
        f.write(f'Test acc mean: {np.mean(accs)}\n')
        f.write(f'Test acc var: {np.std(accs)}\n')
        f.write(f'Test auc mean: {np.mean(aucs)}\n')
        f.write(f'Test auc var: {np.std(aucs)}\n')
        f.write(f'Execute epoch: {np.mean(epoch)}\n')
        f.write(f'Execute time: {np.mean(time)}')
        if 'resample' == model[:8]:
            f.write(f'\n(W) Execute time: {np.mean(wtime)}')

def main():
    model = 'full'
    merge_log(model, '')

    model = 'resample_lstm/segment-1'
    for i in range(2, 31, 2):
        merge_log(model, f'{i}.')

    model = 'non_resample_lstm/segment-1'
    for i in range(2, 31, 2):
        merge_log(model, f'{i}.')

    # model = 'segment-2-0'
    # for i in range(4, 21, 2):
    #     merge_log(model, f'{i}.')

if '__main__' == __name__:
    main()
