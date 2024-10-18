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
    ys = torch.full((beam_size, 1), start_symbol, dtype=torch.long).cuda()  # start_symbol is usually the index for <sos>
    scores = torch.zeros(beam_size).cuda()

    for i in range(max_len - 1):
        # TODO: Decode using the model, memory, and source mask
        tgt_mask = torch.triu(torch.ones((ys.size(1), ys.size(1))), 1).type_as(src_mask.data).bool()
        tgt_mask = tgt_mask.cuda()

        # Calculate probabilities for the next token
        out = model.decode(ys, memory, src_mask, tgt_mask)
        prob = torch.softmax(model.generator(out[:, -1]), dim=-1)

        # Set probabilities of end token to 0 (except when already ended)
        prob[:, end_idx] = prob[:, end_idx].where(ys[:, -1] == end_idx, torch.tensor(0.).cuda())

        # Update scores
        total_scores = scores.unsqueeze(-1) + torch.log(prob)

        # Get top-k scores and indices
        top_scores, top_indices = total_scores.view(-1).topk(beam_size, 0)

        # TODO: Extract beam indices and token indices from top-k scores
        vocab_size = prob.size(1)
        beam_indices = torch.divide(top_indices, vocab_size, rounding_mode='floor')  # Replace with torch.divide(indices, vocab_size, rounding_mode='floor')
        token_indices = torch.divide(top_indices, vocab_size, rounding_mode='floor')  # Replace with torch.divide(indices, vocab_size, rounding_mode='floor')

        # Prepare next decoder input
        next_decoder_input = []
        for beam_index, token_index in zip(beam_indices, token_indices):
            # TODO: Handle end token condition and append to next_decoder_input
            next_seq = torch.cat([ys[beam_index], token_index.unsqueeze(0)], dim=0)
            next_decoder_input.append(next_seq)

        # Update ys
        ys = torch.stack(next_decoder_input, dim=0)
        scores = top_scores

        # Check if all beams are finished, exit
        if (ys[:, -1] == end_idx).all():
            break

        # Expand encoder output for beam size (only once)
        if i == 0:
            memory = memory.repeat(beam_size, 1, 1)
    
    # Return the top-scored sequence
        top_seq = ys[top_scores.argmax()].tolist()
    # convert the top scored sequence to a list of text tokens
    decoded_words = [model.vocab.itos[idx] for idx in top_seq if idx != end_idx]

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

