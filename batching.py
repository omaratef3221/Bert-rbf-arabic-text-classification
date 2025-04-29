from torch.utils.data import DataLoader, TensorDataset
import torch 
def process_batch(model, tokenizer, X_data, batch_size, device):
    """Process a batch of data through the BERT model."""
    inputs = tokenizer(X_data.tolist(), max_length=150, truncation=True, padding='max_length', return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to device
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[:, 0, :]  # Get the [CLS] token representation
    return output

def process_data_in_batches(model, tokenizer, X_data, batch_size, device):
    """Process data in smaller batches to avoid memory overload."""
    all_outputs = []
    num_batches = len(X_data) // batch_size + (1 if len(X_data) % batch_size != 0 else 0)
    
    for i in range(num_batches):
        # Select a batch
        batch_data = X_data[i * batch_size : (i + 1) * batch_size]
        # Process the batch
        batch_output = process_batch(model, tokenizer, batch_data, batch_size, device)
        all_outputs.append(batch_output)
        
        # Clear GPU memory after each batch
        del batch_data
        torch.cuda.empty_cache()

    # Concatenate all outputs into a single tensor
    return torch.cat(all_outputs, dim=0)

