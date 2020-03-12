import numpy as np

def merge_log(model, seg_len):
    accs = []
    aucs = []
    mccs = []
    epoch = []
    time = []
    wtime = []
    mode = model.split('/')[1]
    for i in range(10):
        with open(f'./log/{model}/{seg_len}{i+1}.log', 'r') as f:
            log = [v.rstrip('\n') for v in f.readlines()[-6:]]
        if 'resample' == mode:
            # accs.append(float(log[-6][32:]))
            aucs.append(float(log[-5][10:]))
            mccs.append(float(log[-4][10:]))
            epoch.append(int(log[-3][15:]))
            time.append(float(log[-2][14:]))
            wtime.append(float(log[-1][18:]))
        else:
            # accs.append(float(log[-5][10:]))
            aucs.append(float(log[-4][10:]))
            mccs.append(float(log[-3][10:]))
            epoch.append(int(log[-2][15:]))
            time.append(float(log[-1][14:]))

    with open(f'./log/{model}/{seg_len}all.log', 'w') as f:
        f.write(f'Test acc mean: {np.mean(accs)}\n')
        f.write(f'Test acc var: {np.std(accs)}\n')
        f.write(f'Test auc mean: {np.mean(aucs)}\n')
        f.write(f'Test auc var: {np.std(aucs)}\n')
        f.write(f'Test mcc mean: {np.mean(mccs)}\n')
        f.write(f'Test mcc var: {np.std(mccs)}\n')
        f.write(f'Execute epoch: {np.mean(epoch)}\n')
        f.write(f'Execute time: {np.mean(time)}\n')
        f.write(f'Execute time: {np.std(time)}')
        if 'resample' == mode:
            f.write(f'\n(W) Execute time: {np.mean(wtime)}')
            f.write(f'\n(W) Execute time: {np.std(wtime)}')

def main():
    neuron = 'simpleRnn'

    merge_log(f'{neuron}/full', '')

    model = f'{neuron}/resample/segment-1'
    for i in range(5, 91, 5):
        merge_log(model, f'{i}.')

if '__main__' == __name__:
    main()
