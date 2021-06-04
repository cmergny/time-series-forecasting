
import matplotlib.pyplot as plt
import numpy as np
import torch

def PlotPredictions(X, Y, P, batch, mode, title='', name=''):
    if type(X) == torch.Tensor:
        X, Y, P = np.array(X.to("cpu").detach()), np.array(Y.to("cpu").detach()), np.array(P.to("cpu").detach())
    len_x = len(X[:, batch, mode])
    ow = Y.shape[0]
    target_len = P.shape[0]
    X, Y, P = X[:, batch, mode], Y[:, batch, mode], P[:, batch, mode]
    figure, ax = plt.subplots()
    ax.plot(range(len_x), X, label='Data')
    ax.plot(range(len_x, len_x+ow), Y, color='green')
    ax.plot(range(len_x, len_x+ow), Y,'x', color='green',label='target')
    ax.plot(range(len_x, len_x+target_len), P, color='orange')
    ax.plot(range(len_x, len_x+target_len), P, 'x', color='orange', label='predictions')
    ax.set_ylabel('amplitude')
    ax.set_xlabel('timesteps')
    plt.title(title)
    plt.legend()
    if name != '':
        savepath = f'Figures/{name}'
        figure.savefig(savepath, dpi=300)
        print('Saved figure at : ' + savepath)
    plt.show()

if __name__ == '__main__':
    print('e')
    # dic = {}
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         dic[name] = param.to("cpu").detach()
    #         print(name)

    # figure, ax = plt.subplots(figsize=(5,8))
    # plt.pcolormesh(dic['encoder.lstm.weight_ih_l0'], cmap='Blues')
    # ax.set_aspect('equal')
    # ax.set_xlabel('modes')
    # ax.set_ylabel('weight of first layer')
    # plt.title(f"LSTM Encoder first layer, {k}ème mode")
    # plt.colorbar()
    # plt.legend()
    # plt.show()
    