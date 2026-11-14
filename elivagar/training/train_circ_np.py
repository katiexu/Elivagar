import numpy as np
import torch

from quantumnat.quantize import PACTActivationQuantizer


class TQCeLoss(torch.nn.Module):
    def __init__(self, num_useful_qubits, num_meas_qubits, device):
        super().__init__()
        
        self.filter_matrix = np.zeros((num_meas_qubits, num_useful_qubits))
        
        for i in range(num_useful_qubits):
            self.filter_matrix[i, i] = 1
            
        self.filter_matrix = torch.from_numpy(self.filter_matrix).to(torch.float32).to(device)
            
    def forward(self, preds, labels):
        useful_preds = torch.matmul(preds, self.filter_matrix)
        # [MNIST-10 NLL] Keep class targets one-dimensional even when the final batch has one sample.
        loss = torch.nn.functional.cross_entropy(useful_preds, labels.long().reshape(-1))
            
        return loss


class TQMseLoss(torch.nn.Module):
    def __init__(self, num_useful_qubits, num_meas_qubits, device=None):
        super().__init__()
        
        self.filter_matrix = np.zeros((num_meas_qubits, num_useful_qubits))
        
        if device is None:
            device = torch.device('cpu')
        
        for i in range(num_useful_qubits):
            self.filter_matrix[i, i] = 1
            
        self.filter_matrix = torch.from_numpy(self.filter_matrix).to(torch.float32).to(device)
            
    def forward(self, preds, labels):
        useful_preds = torch.matmul(preds, self.filter_matrix)
        
        loss = torch.mean(torch.sum(torch.pow(torch.sub(useful_preds, labels), 2), 1))
            
        return loss


def compute_tq_model_acc(preds, labels, num_meas_qubits):
    return torch.sum(torch.eq(torch.sum(torch.gt(torch.multiply(preds[:, :num_meas_qubits], labels), 0), 1), num_meas_qubits).to(torch.float))


def compute_tq_model_acc_ce(preds, labels, num_meas_qubits):
    # [MNIST-10 NLL] Normalize label shape so both (batch,) and (batch, 1) targets compare sample-wise.
    labels = labels.reshape(-1, 1)
    return torch.sum(torch.eq(torch.argmax(preds[:, :num_meas_qubits], dim=1, keepdim=True), labels))
 

def train_tq_model(model, num_meas_qubits, opt, loss, data_loader, test_data_loader,
                   epochs, print_freq=1, loss_window=50, acc_fn=None,
                   use_test=True, return_train_losses=False, quantize=False,
                   noise_strength=0.1, device=torch.device('cpu'),
                   train_metrics_data_loader=None):
    device = torch.device('cpu')
    
    if acc_fn == None:
        if isinstance(loss, TQCeLoss):
            acc_fn= compute_tq_model_acc_ce
        else:
            acc_fn = compute_tq_model_acc
        
    quantizer = PACTActivationQuantizer(
        model, precision=4, alpha=1.0, backprop_alpha=False,
        device=device, lower_bound=-5, upper_bound=5
    )
    
    if quantize:
        model.set_noise_injection(True)
    
    losses = []
    valid_losses = []
    
    for epoch in range(epochs):
        quantizer.register_hook()
        
        for step, (x, y) in enumerate(data_loader):
            x = x.to(torch.float).to(device)
            y = y.to(device)

            opt.zero_grad()
            
            out = model(x)
    
            batch_loss = loss(out, y)
            batch_loss.backward()
            
            losses.append(batch_loss.detach().item())

            opt.step()

            if not (step % print_freq):
                batch_acc = acc_fn(out, y, num_meas_qubits) / y.shape[0]
                print(f'Epoch {epoch + 1} | Step {step + 1} | Loss: {(losses[-1]):7.5} | Acc: {batch_acc:7.5}')
#             else:
#                 print(acc_fn(out, y, num_meas_qubits) / y.shape[0])
                
        quantizer.remove_hook()

    model.set_noise_injection(False)
        
    test_losses = []
    test_accs = []
    
    if use_test:
        num_test_samples = 0

        quantizer.register_hook()
        
        for x, y in test_data_loader:
            x = x.to(torch.float).to(device)
            y = y.to(device)
            
            preds = model(x)

            test_losses.append(loss(preds, y).detach().numpy())
            test_accs.append(acc_fn(preds, y, num_meas_qubits).detach().numpy())
            
            num_test_samples += x.shape[0]
            
        quantizer.remove_hook()

        ret_val = [np.mean(test_losses), np.sum(test_accs) / num_test_samples]
    else:
        ret_val = []
        
    if return_train_losses:
        ret_val.append(losses)

    if train_metrics_data_loader is not None:
        # [TrainMetrics] Evaluate the final model over every training sample with the same quantized hook as test evaluation.
        train_loss_sum = 0.0
        train_correct = 0.0
        num_train_samples = 0

        # [TrainMetrics] Restore Torch RNG state so this extra pass cannot affect subsequent run initialization/training.
        with torch.random.fork_rng(devices=[]):
            quantizer.register_hook()
            try:
                with torch.no_grad():
                    for x, y in train_metrics_data_loader:
                        x = x.to(torch.float).to(device)
                        y = y.to(device)

                        preds = model(x)
                        batch_size = x.shape[0]
                        train_loss_sum += loss(preds, y).item() * batch_size
                        train_correct += acc_fn(preds, y, num_meas_qubits).item()
                        num_train_samples += batch_size
            finally:
                quantizer.remove_hook()

        if num_train_samples == 0:
            raise ValueError('Cannot compute final training metrics for an empty training dataset.')

        ret_val.append({
            'loss': train_loss_sum / num_train_samples,
            'accuracy': train_correct / num_train_samples,
        })
        
    return ret_val