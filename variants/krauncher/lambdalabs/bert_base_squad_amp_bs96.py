"""Krauncher variant of tasks/lambdalabs/bert_base_squad_amp_bs96.py (the neutral form, with
the source and the reduction it documents); the expected forecast and its
sources are in tasks/lambdalabs/bert_base_squad_amp_bs96.json.

The task function is the neutral one; model and dataset come through the
hf:// data bridge (/data/<org>__<name>) instead of the Hub. FUNC is the task,
OPTIONS the arguments of its @client.task decorator, KWARGS the arguments of
the call.
"""

OPTIONS = dict(
    timeout=3600,
    data_urls=["hf://datasets/rajpurkar/squad", "hf://models/google-bert/bert-base-uncased"],
)

KWARGS = dict(batch_size=96, max_steps=100, max_seq_length=384, doc_stride=128)


def finetune_bert_base_squad(batch_size: int = 96, max_steps: int = 100,
                             max_seq_length: int = 384, doc_stride: int = 128):
    import time

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader, RandomSampler, TensorDataset
    from transformers import AutoTokenizer, BertForQuestionAnswering

    device = torch.device("cuda")
    model_id = "/data/google-bert__bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = BertForQuestionAnswering.from_pretrained(model_id).to(device)

    ds = load_dataset("/data/rajpurkar__squad", split="train")
    enc = tokenizer(ds["question"], ds["context"], truncation="only_second",
                    max_length=max_seq_length, stride=doc_stride, padding="max_length",
                    return_overflowing_tokens=True, return_offsets_mapping=True)
    starts, ends = [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        ex = enc["overflow_to_sample_mapping"][i]
        a_start = ds[ex]["answers"]["answer_start"][0]
        a_end = a_start + len(ds[ex]["answers"]["text"][0])
        seq = enc.sequence_ids(i)
        ctx = [k for k, s in enumerate(seq) if s == 1]
        s_tok = e_tok = 0
        if offsets[ctx[0]][0] <= a_start and offsets[ctx[-1]][1] >= a_end:
            s_tok = next(k for k in ctx if offsets[k][1] > a_start)
            e_tok = next(k for k in reversed(ctx) if offsets[k][0] < a_end)
        starts.append(s_tok)
        ends.append(e_tok)
    data = TensorDataset(torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"]),
                         torch.tensor(enc["token_type_ids"]), torch.tensor(starts), torch.tensor(ends))
    loader = DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(d in n for d in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(d in n for d in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=0.0, fused=True)
    scaler = torch.cuda.amp.GradScaler()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=max_seq_length)

    model.train()
    step, t0 = 0, time.time()
    while step <= max_steps:
        for input_ids, mask, seg, s_pos, e_pos in loader:
            if step > max_steps:
                break
            input_ids, mask, seg = input_ids.to(device), mask.to(device), seg.to(device)
            s_pos, e_pos = s_pos.to(device).clamp_(0, max_seq_length), e_pos.to(device).clamp_(0, max_seq_length)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=input_ids, attention_mask=mask, token_type_ids=seg)
                loss = (loss_fct(out.start_logits, s_pos) + loss_fct(out.end_logits, e_pos)) / 2
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            loss_value = loss.item()
            step += 1
    elapsed = time.time() - t0
    return {"features": len(data), "steps": step, "elapsed_sec": round(elapsed, 2),
            "sequences_per_sec": round(step * batch_size / elapsed, 1), "last_loss": round(loss_value, 3)}



FUNC = finetune_bert_base_squad
