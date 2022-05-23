import time
epochs = 3
n_batch = 6
for i in range(epochs):
    for j in range(n_batch):
        time.sleep(0.5)
        loss =1/((i+1)*(j+1))
        print("\rEpoch: {:d} batch: {:d} loss: {:.4f} ".format(i+1, j+1, loss), end='')
    print("\rEpoch: {:d}/{:d} epoch_loss: {:.4f} ".format(i+1, epochs, loss, end='\n'))
