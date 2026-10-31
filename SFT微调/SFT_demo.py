import os
import time
from numpy import dtype
os.system('cls' if os.name == 'nt' else 'clear')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dataclasses import dataclass
import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer,AutoModelForCausalLM, PreTrainedTokenizer
from typing import List
# from rich import print

@dataclass
class SFTConfig:
    """SFT过程当中，所有的相关配置"""
    train_data_size:int = 2000
    max_lr: float = 3e-5
    min_lr: float = 1e-6
    warmup_ratio: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = r"D:\ML_Project\LLM\runs_sft"
    log_iter: int = 10
    save_dir:str = "./finetuned/Qwen3-0.6B-SFT"

tokenizer = AutoTokenizer.from_pretrained(r"D:\ML_Project\LLM\SFT微调\model\Qwen3-0.6B-Base")
model = AutoModelForCausalLM.from_pretrained(r"D:\ML_Project\LLM\SFT微调\model\Qwen3-0.6B-Base",dtype=torch.bfloat16)


def get_data_ultrachat_200k(config: SFTConfig):
    ultrachat_200k_data = datasets.load_dataset("./SFT微调/data/ultrachat_200k")
    train_data = []
    i = 0
    while True:
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        msgs += list(ultrachat_200k_data["train_sft"][i]["messages"])
        input_ids = tokenizer.apply_chat_template(      # ← 直接用全局的
            msgs, tokenize=True, add_generation_prompt=False,
            truncation=True, max_length=1024
        )
        train_data.append(input_ids)
        i += 1  
        if i % 1000 == 0:
            print(f"已经处理了 {i} 条数据")
        if i == config.train_data_size:
            break
    return train_data

class SFTDataset(Dataset):
    """SFT数据集"""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """将一个batch的样本拼接成一个tensor，并返回input_ids和labels"""
    max_len = max(len(x) for x in batch) # 最大长度
    pad_id = tokenizer.pad_token_id
    old = torch.full((len(batch), max_len), pad_id, dtype=torch.long) 
    for i, x in enumerate(batch):
        old[i,:len(x)] = torch.tensor(x, dtype=torch.long)
    input_idx = old[:, :-1]  # 去掉最后一个token
    labels = old[:, 1:]  # 去掉第一个token
    return input_idx, labels

def my_dataloader(dataset, batch_size=8, shuffle=True):
    """自定义 dataloader"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def _parse_conversation_turns(eos_position):
    """解析对话轮次，返回 user / assistant 各自的结束位置（指向 <|im_end|> 后面那个 token）"""
    user_ends      = [p + 1 for p in eos_position[::2]]
    assistant_ends = [p + 1 for p in eos_position[1::2]]
    return user_ends, assistant_ends

def creat_answer_mask(input_ids, tokenizer: PreTrainedTokenizer):
    """标出"哪些位置参与 loss"，返回与 input_ids 同形状的 0/1 张量。

    :param input_ids: 必须是【已经位移过的】input_idx = old[:, :-1]，
                      对应的 labels = old[:, 1:]
    :param tokenizer: 用来取 <|im_end|> 的 id

    ── 先看模板的 token 排布，所有偏移量都从这来 ──
        ... <|im_end|>  "\n"  <|im_start|>  "assistant"  "\n"  答案内容...  <|im_end|>
              im_end     +0        +1           +2         +3     +4
        用户那句完了：user_end 指向 "\n"（= im_end + 1）
        助手那句完了：assistant_end 指向 <|im_end|> + 1

    ── 关键前提：mask 的下标 p 表示"用位置 p 的 logits 预测位置 p+1 的 token" ──
        因为 labels 已经右移了一位。
        所以 mask 天然要比答案本身【早一格】—— 这就是 "+3 看着少跳了一个
        token、结果却正确" 的全部原因。
    """
    answer_mask = torch.zeros_like(input_ids)        # 全 0，稍后把答案段置 1
    eos_id = tokenizer.encode("<|im_end|>")[0]       # 对话轮次分隔符的 token id

    for i, seq in enumerate(input_ids):              # 逐条样本（同一 batch 已 pad 成等长）
        # 1. 找出这条样本里所有 <|im_end|> 的位置
        eos_position = torch.where(seq == eos_id)[0].tolist()

        # 2. 第 1 个 <|im_end|> 属于 system prompt，对话轮次从第 2 个才开始
        eos_position = eos_position[1:]

        # 3. 交替切分：偶数下标是 user 结束、奇数下标是 assistant 结束（内部已 +1）
        user_ends, assistant_ends = _parse_conversation_turns(eos_position)

        # 4. 三种情况决定"哪些轮次能配对"
        if len(user_ends) == len(assistant_ends):
            # 正常：user 和 assistant 一样多，正好两两配对
            pairs = list(zip(user_ends, assistant_ends))

        elif len(user_ends) == len(assistant_ends) + 1:
            # 被 max_length 截断：最后一轮 user 在，但 assistant 的回答被切断，
            # 没有收尾的 <|im_end|>，因此拿不到 assistant_end，配不上对
            pairs = list(zip(user_ends[:-1], assistant_ends))   # 前面完整的轮次照常配对
            # 剩下的半截回答没有结束位置，直接标到序列末尾。
            # 末尾的 pad 不用怕 —— 训练时会再 & 一个 valid_mask 把 pad 去掉。
            # 这一行必须在循环体内、用 i，否则只会标到最后一行
            answer_mask[i, user_ends[-1] + 3:] = 1

        else:
            # 理论上不会出现（assistant 比 user 还多）。真出现就整条跳过，宁可不标也别标错
            pairs = []

        # 5. 把每一对完整轮次的答案段置 1
        for u, a in pairs:
            # u = user 的 <|im_end|> + 1，再 +3 跳过 "\n" "<|im_start|>" "assistant"，
            #     落点是 assistant 后面那个 "\n"。因为 mask 语义是"预测下一个 token"，
            #     从这一格开始，监督到的第一格正好是答案的第一个字。
            # a - 1 = <|im_end|> 自身的位置；切片左闭右开，
            #     所以最后一格 a-2 监督的正是 <|im_end|> —— 结束符也在训练范围内，
            #     模型才学得会"什么时候停"。
            answer_mask[i, u + 3 : a - 1] = 1

    return answer_mask

def compute_loss(logits, labels, answer_mask):
    """计算 loss
    :param logits: 模型输出的 logits [B, seq_len, vocab_size]
    :param labels: 真实的标签 [B, seq_len]
    :param answer_mask: answer_mask [B, seq_len]
    """
    log_prob = torch.log_softmax(logits, dim=-1)  # [B, seq_len, vocab_size]
    labels = labels.unsqueeze(-1)  # [B, seq_len, 1]
    loss = -log_prob.gather(-1, labels).squeeze(-1)  # [B, seq_len]
    mask_token_loss = loss * answer_mask  # [B, seq_len]
    loss = mask_token_loss.sum() / answer_mask.sum()  # 平均 loss
    return loss

def cosine_decay(total_steps, current_step, warmup_steps, min_lr, max_lr):
    """余弦退火学习率衰减
    :param total_steps: 总训练步数
    :param current_step: 当前训练步数
    :param warmup_steps: 预热步数
    :param min_lr: 最小学习率
    :param max_lr: 最大学习率
    """
    if current_step < warmup_steps:
        return max_lr * current_step / warmup_steps # 快速上升
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    cos_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793))) # 余弦退火
    decayed = (max_lr - min_lr) * cos_decay + min_lr
    return decayed.item()

def train():
    config = SFTConfig()
    writer = SummaryWriter(log_dir=config.log_dir)
    train_data = get_data_ultrachat_200k(config)
    dataset = SFTDataset(train_data)
    dataloader = my_dataloader(dataset, batch_size=2, shuffle=True)
    model.train()
    model.to(config.device) # pyright: ignore[reportArgumentType]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr)
    total_steps = len(dataloader)
    total_loss = []
    for step, batch in enumerate(dataloader):
        input_ids, labels = batch
        input_ids = input_ids.to(config.device) # pyright: ignore[reportArgumentType]
        labels = labels.to(config.device) # pyright: ignore[reportArgumentType]
        answer_mask = creat_answer_mask(input_ids, tokenizer)
        valid_mask = torch.where(labels != tokenizer.pad_token_id, 1, 0)
        really_answer_mask = answer_mask & valid_mask
        if(really_answer_mask.sum(dim=-1).min().item() == 0): # 有一个无效label的样本，跳过
            print("有一个无效label的样本，跳过")
            continue
        logits = model(input_ids).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none').view(labels.shape)
        loss = (loss * really_answer_mask).sum() / really_answer_mask.sum()
        loss.backward()
        total_loss.append(loss.item())

        current_learning_rate = cosine_decay(
            current_step=step,
            warmup_steps=int(config.warmup_ratio * total_steps),
            total_steps=total_steps,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
        )
        writer.add_scalar("train_lr", current_learning_rate, step)
        optimizer.param_groups[0]['lr'] = current_learning_rate
        optimizer.step()
        optimizer.zero_grad()

        should_log = (step + 1) % config.log_iter == 0 or (step + 1) == total_steps
        if should_log:
            recent_losses = total_loss[-config.log_iter:]
            recent_average_loss = sum(recent_losses) / len(recent_losses)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{step+1}/{total_steps}] loss={recent_average_loss:.4f} lr={current_learning_rate:.2e}")
            writer.add_scalar("train_loss",recent_average_loss,step)
        
    print("训练完成")
    model.save_pretrained(config.save_dir)
    tokenizer.save_pretrained(config.save_dir)


if __name__ == "__main__":
    train()