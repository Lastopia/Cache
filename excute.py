import os
import time
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from selfTF import SelfTransformer

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

from datasets import load_dataset
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
V = tokenizer.vocab_size
print(f"WikiText-103 loaded, Vocab size: {V}")


BATCH_SIZE = 4
TRAIN_LEN = 64
TEST_LENS = [256, 512]
EPOCHS = 2
LR = 5e-4
RUN_ROPE = False   # 是否跑 RoPE 实验（与 Std/ALiBi 并列）
RUN_SAIBI1 = False # 是否跑 SAiBi1 实验
RUN_SAIBI2 = False# 是否跑 SAlibi2 实验
RUN_SALIBI4 = True # 是否跑 SAlibi4 实验

tf = SelfTransformer(N=4, d_embed=256, d_ff=1024, h=4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

def generate_data(batch_size, seq_len, dataset, tokenizer):
    
    total_tokens = batch_size * seq_len * 2
    texts = []
    tokens_len = 0
    
    # 采样足够长文本
    while tokens_len < total_tokens:
        idx = torch.randint(0, len(dataset), (1,)).item()
        text = dataset[idx]['text']
        if len(text.strip()) > 50:
            texts.append(text)
            tokens_len += len(tokenizer(text)['input_ids'])
    
    # 合并&tokenize
    full_text = ' '.join(texts[:8])  # 8篇文章
    encoding = tokenizer(full_text, truncation=True, 
                        max_length=total_tokens*2, return_tensors="pt")
    tokens = encoding['input_ids'][0]  # [seq_max]
    
    # 切动态batch
    input_batch, target_batch = [], []
    for _ in range(batch_size):
        start = torch.randint(0, max(1, len(tokens)-seq_len*2), (1,)).item()
        seq = tokens[start:start+seq_len*2]
        if len(seq) < seq_len*2:
            pad_len = seq_len*2 - len(seq)
            seq = torch.cat([seq, torch.full((pad_len,), tokenizer.pad_token_id)])
        input_batch.append(seq[:-1])
        target_batch.append(seq[1:])
    
    return torch.stack(input_batch).to(device), torch.stack(target_batch).to(device)

def train_short(model, train_len, epochs, name="Model"):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    losses = []
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_yscale('log')
    
    for epoch in tqdm(range(epochs), desc=f"Train {name}"):
        epoch_loss = 0
        num_batches = 30  # 真实数据/epoch batches
        pbar = tqdm(range(num_batches), leave=False, desc=f"E{epoch+1}")
        
        for _ in pbar:
            inputids, targets = generate_data(BATCH_SIZE, train_len,dataset,tokenizer)
            mask = tf.sub_mask(inputids).to(device)
            logits = model(inputids, mask)
            loss = criterion(logits.reshape(-1, V), targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"{name} E{epoch+1}: Loss={avg_loss:.3f}")
        
        # 动态Loss图
        ax.clear()
        ax.plot(range(1,len(losses)+1), losses, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)'); ax.set_title(f'{name} Training')
        ax.grid(True, alpha=0.3); plt.pause(0.3)
    
    plt.ioff()
    plt.savefig(f'plots/train_loss_{name.lower()}.png', dpi=150)
    plt.close()
    torch.save(model.state_dict(), f'plots/{name.lower()}_model.pt')  # 保存模型
    return losses, avg_loss

def eval_ppl(model, test_len, num_batches=20, name=""):
    model.eval()
    total_loss, n_tokens = 0, 0
    pbar = tqdm(range(num_batches), desc=f"PPL {name} L={test_len}")
    
    with torch.no_grad():
        for _ in pbar:
            inputids, targets = generate_data(BATCH_SIZE, test_len,dataset,tokenizer)
            mask = tf.sub_mask(inputids).to(device)
            logits = model(inputids, mask)
            loss = criterion(logits.reshape(-1, V), targets.reshape(-1))
            total_loss += loss.item() * targets.numel()
            n_tokens += targets.numel()
            pbar.set_postfix({'bPPL': f"{math.exp(loss.item()):.1f}"})
    
    ppl = math.exp(total_loss / n_tokens)
    print(f"{name} L={test_len} PPL: {ppl:.1f}")
    return ppl

# === 实验主流程 ===
os.makedirs('plots', exist_ok=True)
print(f"\n=== ALiBi/SAiBi/RoPE 实验: WikiText-103 Train={TRAIN_LEN} Test={TEST_LENS} ===")

results = {}  # 收集每个模型的训练时间和各长度 PPL
lengths = [TRAIN_LEN] + TEST_LENS

# 1. Std Sinusoidal
print("\n1. Std (PositionalEmbedding)")
model_std = tf.deonly_model(V)
t0 = time.perf_counter()
losses_std, train_loss_std = train_short(model_std, TRAIN_LEN, EPOCHS, "Std")
train_time_std = time.perf_counter() - t0
ppls_std = [train_loss_std] + [eval_ppl(model_std, L, name="Std") for L in TEST_LENS]
results["Std"] = {
    "train_time": train_time_std,
    "ppls": {L: p for L, p in zip(lengths, ppls_std)},
    "losses": losses_std,
}

# 2. ALiBi
print("\n2. ALiBi")
model_alibi = tf.alibi_model(V)
t0 = time.perf_counter()
losses_alibi, train_loss_alibi = train_short(model_alibi, TRAIN_LEN, EPOCHS, "ALiBi")
train_time_alibi = time.perf_counter() - t0
ppls_alibi = [train_loss_alibi] + [eval_ppl(model_alibi, L, name="ALiBi") for L in TEST_LENS]
results["ALiBi"] = {
    "train_time": train_time_alibi,
    "ppls": {L: p for L, p in zip(lengths, ppls_alibi)},
    "losses": losses_alibi,
}

# 3. SAiBi1（可选）
if RUN_SAIBI1:
    print("\n3. SAiBi1")
    model_saibi1 = tf.salibi_model_one(V)
    t0 = time.perf_counter()
    losses_saibi1, train_loss_saibi1 = train_short(model_saibi1, TRAIN_LEN, EPOCHS, "SAiBi1")
    train_time_saibi1 = time.perf_counter() - t0
    ppls_saibi1 = [train_loss_saibi1] + [eval_ppl(model_saibi1, L, name="SAiBi1") for L in TEST_LENS]
    results["SAiBi1"] = {
        "train_time": train_time_saibi1,
        "ppls": {L: p for L, p in zip(lengths, ppls_saibi1)},
        "losses": losses_saibi1,
    }

# 4. SAlibi2（可选）
if RUN_SAIBI2:
    print("\n4. SAlibi2")
    model_saibi2 = tf.salibi_model_2(V)
    t0 = time.perf_counter()
    losses_saibi2, train_loss_saibi2 = train_short(model_saibi2, TRAIN_LEN, EPOCHS, "SAlibi2")
    train_time_saibi2 = time.perf_counter() - t0
    ppls_saibi2 = [train_loss_saibi2] + [eval_ppl(model_saibi2, L, name="SAlibi2") for L in TEST_LENS]
    results["SAlibi2"] = {
        "train_time": train_time_saibi2,
        "ppls": {L: p for L, p in zip(lengths, ppls_saibi2)},
        "losses": losses_saibi2,
    }

# 5. SAlibi4（可选）
if RUN_SALIBI4:
    print("\n5. SAlibi4")
    model_saibi4 = tf.salibi_model_4(V, a=1000.0)
    t0 = time.perf_counter()
    losses_saibi4, train_loss_saibi4 = train_short(model_saibi4, TRAIN_LEN, EPOCHS, "SAlibi4")
    train_time_saibi4 = time.perf_counter() - t0
    ppls_saibi4 = [train_loss_saibi4] + [eval_ppl(model_saibi4, L, name="SAlibi4") for L in TEST_LENS]
    results["SAlibi4"] = {
        "train_time": train_time_saibi4,
        "ppls": {L: p for L, p in zip(lengths, ppls_saibi4)},
        "losses": losses_saibi4,
    }

# 6. RoPE（可选）
if RUN_ROPE:
    print("\n6. RoPE")
    model_rope = tf.rope_model(V)
    t0 = time.perf_counter()
    losses_rope, train_loss_rope = train_short(model_rope, TRAIN_LEN, EPOCHS, "RoPE")
    train_time_rope = time.perf_counter() - t0
    ppls_rope = [train_loss_rope] + [eval_ppl(model_rope, L, name="RoPE") for L in TEST_LENS]
    results["RoPE"] = {
        "train_time": train_time_rope,
        "ppls": {L: p for L, p in zip(lengths, ppls_rope)},
        "losses": losses_rope,
    }

# 7. 数据分析表（按模型汇总）
print("\n=== 模型整体表现汇总（PPL & 训练时间） ===")
test_256, test_512 = 256, 512

col_names = ["Model", f"PPL@{TRAIN_LEN}", f"PPL@{test_256}", f"PPL@{test_512}", "Train Time (s)"]
col_widths = [10, 12, 12, 12, 16]

def fmt_cell(val, width):
    if val is None:
        s = "-"
    elif isinstance(val, (float, int)):
        s = f"{val:.2f}"
    else:
        s = str(val)
    return s.ljust(width)

header_line = " ".join(fmt_cell(n, w) for n, w in zip(col_names, col_widths))
print(header_line)
print("-" * len(header_line))

model_order = ["Std", "ALiBi", "SAiBi1", "SAlibi2", "SAlibi4", "RoPE"]
for name in model_order:
    if name not in results:
        continue
    info = results[name]
    ppls = info["ppls"]
    row_vals = [
        name,
        ppls.get(TRAIN_LEN),
        ppls.get(test_256),
        ppls.get(test_512),
        info["train_time"],
    ]
    print(" ".join(fmt_cell(v, w) for v, w in zip(row_vals, col_widths)))

# 8. 最终可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
epochs_range = range(1, EPOCHS + 1)
ax1.plot(epochs_range, losses_std, 'o-', label='Std', linewidth=3, markersize=8)
ax1.plot(epochs_range, losses_alibi, 's-', label='ALiBi', linewidth=3, markersize=8)
if RUN_ROPE:
    ax1.plot(epochs_range, losses_rope, '^-', label='RoPE', linewidth=3, markersize=8)
if RUN_SAIBI1:
    ax1.plot(epochs_range, losses_saibi1, 'x-', label='SAiBi1', linewidth=3, markersize=8)
if RUN_SAIBI2:
    ax1.plot(epochs_range, losses_saibi2, 'd-', label='SAlibi2', linewidth=3, markersize=8)
if RUN_SALIBI4:
    ax1.plot(epochs_range, losses_saibi4, 'v-', label='SAlibi4', linewidth=3, markersize=8)
ax1.set_yscale('log'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (WikiText-103)'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.semilogy(lengths, ppls_std, 'o-', label='Std PPL', linewidth=3, markersize=10)
ax2.semilogy(lengths, ppls_alibi, 's-', label='ALiBi PPL', linewidth=3, markersize=10)
if RUN_ROPE:
    ax2.semilogy(lengths, ppls_rope, '^-', label='RoPE PPL', linewidth=3, markersize=10)
if RUN_SAIBI1:
    ax2.semilogy(lengths, ppls_saibi1, 'x-', label='SAiBi1 PPL', linewidth=3, markersize=10)
if RUN_SAIBI2:
    ax2.semilogy(lengths, ppls_saibi2, 'd-', label='SAlibi2 PPL', linewidth=3, markersize=10)
if RUN_SALIBI4:
    ax2.semilogy(lengths, ppls_saibi4, 'v-', label='SAlibi4 PPL', linewidth=3, markersize=10)
ax2.axvline(TRAIN_LEN, color='k', ls='--', lw=2, alpha=0.7, label='Train Len')
ax2.set_xlabel('Sequence Length'); ax2.set_ylabel('PPL (log)')
ax2.set_title('Train Short → Test Long'); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/alibi_wikitext_full.png', dpi=200, bbox_inches='tight')
plt.show()