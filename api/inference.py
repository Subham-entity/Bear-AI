import torch

def generate_answer(model, vocab, tensor_image, question="describe the image", beam_width=3, max_length=30, device="cpu"):
    model.eval()
    with torch.no_grad():
        encoder_features = model.encoder(tensor_image.unsqueeze(0).to(device))
        
        start_token = vocab.stoi["<START>"]
        end_token = vocab.stoi["<END>"]
        sep_idx = vocab.stoi.get("<SEP>", vocab.stoi["<PAD>"])
        
        # Build initial sequence: [<START>] + [question tokens] + [<SEP>]
        seq = [start_token] + vocab.numericalize(question) + [sep_idx]
        score = 0.0
        beams = [(seq, score)]
        
        for _ in range(max_length):
            new_beams = []
            for current_seq, current_score in beams:
                if current_seq[-1] == end_token:
                    new_beams.append((current_seq, current_score))
                    continue
                
                captions_tensor = torch.tensor(current_seq, dtype=torch.long).unsqueeze(0).to(device)
                logits = model.decoder(captions_tensor, encoder_features)
                
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                topk_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    next_token = topk_indices[i].item()
                    next_score = topk_probs[i].item()
                    new_beams.append((current_seq + [next_token], current_score + next_score))
            
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(seq[-1] == end_token for seq, _ in beams):
                break
                
        best_seq = beams[0][0]
        
        # Extract only the generated answer tokens coming after the <SEP> token
        sep_pos = best_seq.index(sep_idx) if sep_idx in best_seq else 0
        answer_tokens = best_seq[sep_pos + 1:]
        
        processed_seq = []
        for idx in answer_tokens:
            if idx == end_token:
                break
            processed_seq.append(idx)
            
        generated_words = [vocab.itos[idx] for idx in processed_seq]
        return " ".join(generated_words)