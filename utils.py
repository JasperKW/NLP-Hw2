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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encode the source input using the model
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask).to(device)

    # Initialize decoder input with the start symbol for all beams
    ys = torch.ones(beam_size, 1).fill_(start_symbol).type(torch.long).to(device)
    scores = torch.zeros(beam_size).to(device)  # Scores for all beams

    finished = [False] * beam_size
    sequences = [ys.clone() for _ in range(beam_size)]  # Sequences for each beam

    for i in range(max_len - 1):
        if i == 0:  # Expand for beam size
            memory = memory.expand(beam_size, -1, -1)
            src_mask = src_mask.expand(beam_size, -1, -1)
            tgt_mask = tgt_mask.expand(beam_size, -1, -1)
    
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data).to(device)
        out = model.decode(memory, src_mask, torch.cat(sequences, dim=0).to(device), tgt_mask)

        # Get probabilities for the last token in each beam sequence
        prob = model.generator(out[:, -1]).to(device)

        if beam_size == 1:  # If beam_size is 1, act like greedy decoding
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0].item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(ys).fill_(next_word)], dim=1)
            continue

        # Add the previous scores to the current token log-probabilities
        prob = prob + scores.view(beam_size, 1)

        # Flatten prob to get top k token probabilities across all beams
        vocab_size = prob.size(-1)
        topk_scores, topk_indices = torch.topk(prob.view(-1), beam_size)

        # Extract beam indices and token indices
        beam_indices = torch.div(topk_indices, vocab_size, rounding_mode='floor')
        token_indices = topk_indices % vocab_size

        # Update sequences and scores
        next_sequences = []
        next_scores = []

        for beam_idx, token_idx, score in zip(beam_indices, token_indices, topk_scores):
            seq = torch.cat([sequences[beam_idx], token_idx.view(1, 1)], dim=1).to(device)
            next_sequences.append(seq)
            next_scores.append(score)

            # Mark the beam as finished if the end token is generated
            if token_idx.item() == end_idx:
                finished[beam_idx] = True

        sequences = next_sequences
        scores = torch.stack(next_scores).to(device)

        if all(finished):  # Stop if all beams have finished
            break

    # Return the sequence with the highest score
    best_sequence = sequences[scores.argmax().item()].squeeze(0).tolist()

    return best_sequence






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
