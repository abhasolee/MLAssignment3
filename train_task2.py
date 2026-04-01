import torch
import torch.nn as nn
from models.seq2seq import Encoder, Decoder, Seq2Seq
from utils.data_loader import get_task2_dataloader
from utils.metrics import calculate_bleu

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Real Multi30k Data and Word2Vec Embeddings
    dataloader, dataset = get_task2_dataloader(batch_size=64, max_vocab=3000)
    
    src_vocab = dataset.src_vocab_size
    trg_vocab = dataset.trg_vocab_size
    w2v_src = dataset.src_weights.to(device)
    w2v_trg = dataset.trg_weights.to(device)
    
    hidden_size = 256
    epochs = 5
    results = {}

    for rnn_type in ['LSTM', 'GRU']:
        for embed_type in ['Word2Vec', 'One-Hot']:
            print(f"\n{'='*40}")
            print(f"Training Machine Translation: {rnn_type} | {embed_type}")
            print(f"{'='*40}")
            
            use_one_hot = (embed_type == 'One-Hot')
            encoder = Encoder(src_vocab, hidden_size, rnn_type, use_one_hot, w2v_src)
            decoder = Decoder(trg_vocab, hidden_size, rnn_type, use_one_hot, w2v_trg)
            model = Seq2Seq(encoder, decoder, device).to(device)
            
            criterion = nn.CrossEntropyLoss(ignore_index=0) 
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

            final_bleu = 0
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                all_targets = []
                all_predictions = []
                
                for batch_idx, (src, trg) in enumerate(dataloader):
                    src, trg = src.to(device), trg.to(device)
                    
                    optimizer.zero_grad()
                    output = model(src, trg)
                    predictions = output.argmax(dim=-1)
                    
                    output_dim = output.shape[-1]
                    output_flat = output[:, 1:].reshape(-1, output_dim)
                    trg_flat = trg[:, 1:].reshape(-1)
                    
                    loss = criterion(output_flat, trg_flat)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                    # Convert indices back to German words for BLEU score
                    # Skip 0:PAD, 1:SOS, 2:EOS, 3:UNK
                    special_tokens = [0, 1, 2, 3]
                    for i in range(trg.shape[0]):
                        ref_words = [dataset.de_i2w[idx.item()] for idx in trg[i, 1:] if idx.item() not in special_tokens]
                        pred_words = [dataset.de_i2w[idx.item()] for idx in predictions[i, 1:] if idx.item() not in special_tokens]
                        
                        all_targets.append([ref_words]) 
                        all_predictions.append(pred_words)
                    
                avg_loss = total_loss / len(dataloader)
                final_bleu = calculate_bleu(all_predictions, all_targets)
                print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | BLEU Score: {final_bleu:.4f}")
            
            results[f"Seq2Seq {rnn_type} + {embed_type}"] = final_bleu

    print("\n\n" + "*"*40)
    print("TASK 2 EXPERIMENTAL RESULTS (BLEU SCORE)")
    print("*"*40)
    for model_name, score in results.items():
        print(f"{model_name:<30}: {score:.4f}")

if __name__ == "__main__":
    main()