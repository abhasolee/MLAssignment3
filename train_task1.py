import torch
import torch.nn as nn
from models.rnn_models import TextGenerator
from utils.data_loader import get_shakespeare_dataloader
from utils.metrics import calculate_perplexity

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataloader, vocab_size, w2v_weights = get_shakespeare_dataloader(seq_length=20, batch_size=128, max_vocab=3000)
    w2v_weights = w2v_weights.to(device)

    hidden_size = 128
    num_layers = 1
    epochs = 3 
    
    results = {}

    for rnn_type in ['LSTM', 'GRU']:
        for embed_type in ['Word2Vec', 'One-Hot']:
            print(f"\n{'='*40}")
            print(f"Training Model: {rnn_type} | Embeddings: {embed_type}")
            print(f"{'='*40}")
            
            use_one_hot = (embed_type == 'One-Hot')
            model = TextGenerator(vocab_size, hidden_size, num_layers, rnn_type, use_one_hot, w2v_weights).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

            model.train()
            final_perplexity = 0
            
            for epoch in range(epochs):
                total_loss = 0
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs, _ = model(inputs)
                    
                    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                avg_loss = total_loss / len(dataloader)
                final_perplexity = calculate_perplexity(avg_loss)
                print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Perplexity: {final_perplexity:.4f}")
            
            results[f"{rnn_type} + {embed_type}"] = final_perplexity

    print("\n\n" + "*"*40)
    print("TASK 1 EXPERIMENTAL RESULTS (PERPLEXITY)")
    print("*"*40)
    for model_name, score in results.items():
        print(f"{model_name:<20}: {score:.4f}")

if __name__ == "__main__":
    main()