import torch
import copy
import math
import torch.nn as nn
from torch.nn.functional import pad
import sacrebleu


## Dummy functions defined to use the same function run_epoch() during eval
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe.unsqueeze(0)
        x = x + pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)



def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size, end_idx):
    """
    Implement beam search decoding with 'beam_size' width
    """

    #To-do: Encode source input using the model
    memory = model.encode(src, src_mask)

    # Initialize decoder input and scores
    ys = torch.full((1, 1), start_symbol, dtype=torch.long).cuda()  # Single sequence start
    scores = torch.zeros(1).cuda()  # Initial score for the single beam (set to 0)

    # Store completed sequences and their scores
    completed_sequences = []
    completed_scores = []

    for i in range(max_len - 1):
        # Expand the sequences and memory for beam size (at the first iteration)
        if i == 0:
            memory = memory.expand(beam_size, -1, -1)
            ys = ys.expand(beam_size, -1)
            scores = scores.expand(beam_size)

        # Create a target mask for the sequence
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src_mask.data).cuda()

        # Calculate probabilities for the next token
        out = model.decode(ys, memory, src_mask, tgt_mask)
        prob = torch.softmax(model.generator(out[:, -1]), dim=-1)

        # Set probabilities of end token to 0 (except when already ended)
        if i > 0:  # Skip at the start since no sequence has ended
            end_mask = ys[:, -1] == end_idx  # True if the sequence has ended
            prob[end_mask] = 0
            prob[end_mask, end_idx] = 1  # Keep the probability of the end token for ended sequences

        # Update scores
        total_scores = scores.unsqueeze(-1) + torch.log(prob)

        # Get top-k scores and indices
        top_scores, top_indices = total_scores.view(-1).topk(beam_size, 0)

        # TODO: Extract beam indices and token indices from top-k scores
        beam_indices = top_indices // prob.size(1)  # Replace with torch.divide(indices, vocab_size, rounding_mode='floor')
        token_indices = top_indices % prob.size(1)  # Replace with torch.divide(indices, vocab_size, rounding_mode='floor')

        # Prepare next decoder input
        next_ys = []
        for beam_idx, token_idx in zip(beam_indices, token_indices):
            next_seq = torch.cat([ys[beam_idx], token_idx.view(1)], dim=0)
            next_ys.append(next_seq)
            # If sequence ends with the end token, add it to completed sequences
            if token_idx.item() == end_idx:
                completed_sequences.append(next_seq)
                completed_scores.append(top_scores[beam_idx])


        # Update ys
        ys = torch.stack(next_ys, dim=0)
        scores = top_scores

        # Check if all beams are finished, exit
        if len(completed_sequences) >= beam_size:
            break
        
    if len(completed_sequences) == 0:
        completed_sequences = ys
        completed_scores = scores
    
    # Return the top-scored sequence
    best_idx = torch.argmax(torch.tensor(completed_scores))
    best_sequence = completed_sequences[best_idx].tolist()

    # convert the top scored sequence to a list of text tokens
    decoded_words = [model.vocab.itos[idx] for idx in best_sequence if idx != end_idx]

    return " ".join(decoded_words)

def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for s in batch:
        _src = s['de']
        _tgt = s['en']
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def remove_start_end_tokens(sent):

    if sent.startswith('<s>'):
        sent = sent[3:]

    if sent.endswith('</s>'):
        sent = sent[:-4]

    return sent


def compute_corpus_level_bleu(refs, hyps):

    refs = [remove_start_end_tokens(sent) for sent in refs]
    hyps = [remove_start_end_tokens(sent) for sent in hyps]

    bleu = sacrebleu.corpus_bleu(hyps, [refs])

    return bleu.score

