import torch
import re
from functools import reduce, partial
import numpy as np
import toolz as tz
from itertools import chain
import pendulum as pm

def eval_model(model, data):
    data = list(map(torch.from_numpy,data))
    with torch.no_grad():
        return model(data)

def batch_yielder(batch_size, x,y):
    num_samples = list(set([t.shape[0] for t in x]) | set([t.shape[0] for t in y]))
    assert len(num_samples) == 1, "Varying number of samples in batch: {}".format(num_samples)
    for inds in tz.partition_all(batch_size, range(num_samples[0])):
        yield [t[inds[0]:inds[-1]] for t in x], [t[inds[0]:inds[-1]] for t in y]

def build_scalers(scalespec, keys, all_data, allow_sample=True, no_fit_scale_constant=-1):
    """ Build scalers and train on data """
    assert len(keys) == len(all_data), ""
    from sklearn.preprocessing import RobustScaler

    def scalespec_key_to_scaler(scaler_key):
        import re
        matching_keys = [value for key,value in scalespec if re.match(key,scaler_key)]
        if len(matching_keys) == 0:
            raise Exception("Unable to find matching scaler for key: {}".format(scaler_key))
        else:
            return matching_keys[0]
        
    def apply_scalespec(scaler_key, data):
        if allow_sample:
            inds = np.random.permutation(np.arange(len(data)))[:1000]
            data = data[inds]
            
        data_scaler = scalespec_key_to_scaler(scaler_key)
        print("Applying scaler: {} -> {}".format(scaler_key, data_scaler))

        if data_scaler == "robust":
            scaler = RobustScaler()
            data = data[np.all((data != no_fit_scale_constant).reshape(data.shape[0],-1), axis=1)]
            scaler.fit(data.reshape(data.shape[0],-1))
            
            def transform(x):
                return scaler.transform(x.reshape(x.shape[0],-1)).reshape(x.shape)
            def inverse_transform(x):
                return scaler.inverse_transform(x.reshape(x.shape[0],-1)).reshape(x.shape)
            return transform,inverse_transform
        else:
            raise "Unknown scaler: {}".format(data_scaler)

    return list(zip(*map(lambda idx_key: apply_scalespec(idx_key[1], all_data[idx_key[0]]), enumerate(keys))))




def scale_data(scalers,data):
    return list(map(lambda t: t[0](t[1]), zip(scalers, data)))


def test( model, device, loss_model, test_x, test_y, batch_size=128):
    test_loader = batch_yielder(batch_size, test_x, test_y)

    model.eval()
    test_loss = 0    
    with torch.no_grad():
        for data, target in test_loader:
            data = [torch.from_numpy(a).float().to(device) for a in data]
            target = [torch.from_numpy(a).float().to(device) for a in target]

            output = model(data)
            if not isinstance(output, (list,tuple)): output = [output]
            assert len(output) == len(target), "Output/target length mismatch: Output={} vs Target={}".format(len(output), len(target))
            test_loss += sum(map(lambda t: loss_model(*t), zip(output,target)))

            # # release memory
            for d in chain(data,target):
                del d
            del output

    avg_loss = test_loss 
    # / len(test_y[0])
    return avg_loss

def train_model(model, train_data, validation_data, epochs, batch_size=256, use_cuda=True, keep_best=True, verbosity=1, patience=20, loss_func="xentropy"):
    """
    Arguments:
        - train_data: (train_x, train_y)
        - validation_data: (valid_x, valid_y)
    """
    if verbosity > 0:
        print("Training model: {} samples, {} validation".format(len(train_data[0][0]), len(validation_data[0][0])))

    # train_data = (list(map(torch.from_numpy,train_data[0])), list(map(torch.from_numpy,train_data[0])))
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device) 

    if loss_func == "xentropy":
        criterion_underlying = torch.nn.CrossEntropyLoss() # torch.nn.L1Loss() # torch.nn.MSELoss(reduction='sum')
        def criterion(a,b):
            b = torch.max(b,1)[1].long()
            return criterion_underlying(a,b)
    else:
        raise Exception("Loss func not supported: {}".format(loss_func))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    total_items_in_x0 = reduce(lambda s,x: s*x, train_data[1][0].shape)

    def numpy_to_device(a):
        nonlocal device
        return torch.from_numpy(a).to(device)
    
    best_tracker = None # (loss,weights)
    last_checkpoint_epoch = 0
    for epoch_idx in range(epochs):
        model.train()
        if verbosity > 0: print("Epoch {} ..".format(epoch_idx), end="", flush=True)
        epoch_start = pm.now()

        # Forward pass: Compute predicted y by passing x to the model
        losses = []
        for x,y in batch_yielder(batch_size, train_data[0], train_data[1]):
            x,y = list(map(numpy_to_device,x)), list(map(numpy_to_device,y))
            y_pred = model(x)

            loss = reduce(lambda a,b: a+b, [criterion(a,b) for a,b in zip(y_pred, y)]) / len(y_pred)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute loss
        loss = test(model, device, criterion, validation_data[0], validation_data[1])
        if verbosity > 0: print("Loss={}. Took={}s".format(loss, (pm.now() - epoch_start).total_seconds()), end="")
        if keep_best and (best_tracker == None or loss < best_tracker[0]):
            if verbosity > 0: print("(checkpointed)".format(loss), end="")
            best_tracker = (loss,model.state_dict())
            last_checkpoint_epoch = epoch_idx
        elif epoch_idx - last_checkpoint_epoch > patience:
            if verbosity > 0: print("\n - Patience limit reached - giving up")            
            break

        if verbosity > 0: print("")
    
    if best_tracker != None:
        model.load_state_dict(best_tracker[1])
    return model.cpu()


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
        targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


def split_data(vecs, split_factor=0.1):
    num_samples = set([len(k) for k in vecs])
    assert len(num_samples) == 1, "All vectors must be of equal length"
    num_samples = list(num_samples)[0]
    split_idx = num_samples - int(num_samples*split_factor)

    return [t[:split_idx] for t in vecs], [t[split_idx:] for t in vecs]


def load_iris_dataset(validation_split_factor=0.2):
    from sklearn import datasets
    from sklearn.preprocessing import label_binarize, robust_scale
    from sklearn.utils import shuffle

    iris = datasets.load_iris()
    x = [iris.data[:, :4]] 
    y = [label_binarize(iris.target, classes=[0,1,2])]

    # scale the input data
    x = [robust_scale(kx) for kx in x]
    
    # Shuffle the data
    shuffle_index = np.arange(len(x[0]))
    np.random.shuffle(shuffle_index)
    x = [kx[shuffle_index] for kx in x]
    y = [ky[shuffle_index] for ky in y]

    # Pick 
    validation_split_idx  =20
    x,x_valid = [kx[:-validation_split_idx].astype(np.float32) for kx in x], [kx[-validation_split_idx:].astype(np.float32) for kx in x] # split_data(x, split_factor=validation_split_factor) #
    y,y_valid = [ky[:-validation_split_idx].astype(np.float32) for ky in y], [ky[-validation_split_idx:].astype(np.float32) for ky in y] # split_data(y, split_factor=validation_split_factor) #
    return (x,y), (x_valid,y_valid)