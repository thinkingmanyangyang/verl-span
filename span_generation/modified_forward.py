import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import random
import numpy as np

# 设置随机种子为42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)


import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model
from transformers import AutoTokenizer
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from .span_utils import repeat_batch_starts_ends, predict_span_by_similarity


import torch

import torch

import torch
from typing import Tuple

import torch
from typing import Tuple

def find_span_ends(span_starts_ids: torch.LongTensor) -> torch.LongTensor:
    """
    辅助函数，根据每个 token 的 span 起始 ID，计算其 span 的结束 ID (exclusive)。
    例如: [0, 0, 0, 3, 3, 6, 6, 6] -> [3, 3, 3, 6, 6, 9, 9, 9]
    """
    # 找到每个 span 的边界 (下一个 span 开始的地方)
    # 在末尾添加一个极大值作为哨兵，方便计算最后一个 span 的结尾
    boundaries = torch.cat(
        [span_starts_ids[1:], torch.tensor([float('inf')], device=span_starts_ids.device)]
    )
    
    # 找到每个 token 所在位置的下一个 span 的起始 id
    # span_ends_ids[i] 的值是 span_starts_ids[i] 这个 span 的结束位置
    span_ends_ids = boundaries[span_starts_ids.unique(return_inverse=True)[1]]
    
    # 最后一个 span 的 end 应该是序列总长度
    span_ends_ids[span_ends_ids == float('inf')] = len(span_starts_ids)

    return span_ends_ids.long()


def create_attention_masks_from_spans(
    position_ids: torch.LongTensor,
    kv_cache_len: int,
    dtype: torch.dtype = torch.float32,
    intra_span_causal: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据 position_ids (span starts) 和参考代码的逻辑，创建两个独立的 Attention Mask。

    Args:
        position_ids (torch.LongTensor): 一维张量，定义了当前输入序列中每个 token 所属的 span 的起始位置。
                                       例如：[0, 0, 0, 3, 3, 3, 6, 6, 6]
        kv_cache_len (int): 在当前输入之前的 Key-Value Cache 的长度。
        dtype (torch.dtype): mask 的数据类型。
        intra_span_causal (bool): 若为 True，span 内部为因果注意力；若为 False，为双向注意力。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - span_to_kv_mask: 形状为 (query_len, kv_cache_len) # query shape, kv shape
            - intra_span_mask: 形状为 (query_len, query_len) # query shape, kv shape
    """
    device = position_ids.device
    query_len = position_ids.shape[0]

    # --- 1. 准备 Queries 和 Keys 的位置信息 ---
    
    # Queries (当前输入序列)
    q_span_starts = position_ids.view(-1, 1)  # 形状: (query_len, 1)
    
    # Keys for KV Cache
    k_kv_positions = torch.arange(kv_cache_len, device=device).view(1, -1) # 形状: (1, kv_cache_len)
    
    # Keys for Intra-Span (当前输入序列)
    k_span_starts = position_ids.view(1, -1)  # 形状: (1, query_len)

    # 规则 a: span start 可以注意到所有之前 position id 的 kv cache
    mask_prev_spans = q_span_starts >= k_kv_positions
    
    # 规则 b: span 可以注意到所有在同一个 span 内的 Key
    mask_same_span = q_span_starts == k_span_starts
    
    # 如果需要 span 内因果，则增加额外约束
    if intra_span_causal:
        q_indices = torch.arange(query_len, device=device).view(-1, 1)
        k_indices = torch.arange(query_len, device=device).view(1, -1)
        causal_constraint = q_indices >= k_indices
        mask_same_span = mask_same_span & causal_constraint
    
    return mask_prev_spans, mask_same_span

def batch_create_attention_masks_from_spans(
    batch_position_ids: torch.LongTensor,
    kv_cache_len: int,
    dtype: torch.dtype = torch.float32,
    intra_span_causal: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = batch_position_ids.shape[0]
    mask_prev_spans = []
    mask_same_span = []
    min_dtype = torch.finfo(dtype).min
    for i in range(batch_size):
        mask_prev_spans_i, mask_same_span_i = create_attention_masks_from_spans(batch_position_ids[i], kv_cache_len, dtype, intra_span_causal)
        mask_prev_spans_i = mask_prev_spans_i[None, None, :, :]
        mask_same_span_i = mask_same_span_i[None, None, :, :]
        mask_prev_spans_i = torch.where(mask_prev_spans_i, torch.tensor(0.0, device=mask_prev_spans_i.device, dtype=dtype), min_dtype)
        mask_same_span_i = torch.where(mask_same_span_i, torch.tensor(0.0, device=mask_same_span_i.device, dtype=dtype), min_dtype)
        mask_prev_spans.append(mask_prev_spans_i)
        mask_same_span.append(mask_same_span_i)

    return torch.cat(mask_prev_spans, dim=0), torch.cat(mask_same_span, dim=0)

def get_topk_threshold(entropy: torch.Tensor, attention_mask: torch.Tensor, topk: float = 0.2):
    """
    获取attention_mask有效部分的entropy的top k阈值
    
    Args:
        entropy: 形状为 (batch_size, seq_len) 的熵值张量
        attention_mask: 形状为 (batch_size, seq_len) 的注意力掩码
        topk: float, 0.0-1.0, 表示取前百分之多少作为阈值
    
    Returns:
        threshold: 形状为 (batch_size,) 的阈值张量
    """
    batch_size, seq_len = entropy.shape
    thresholds = []
    
    for i in range(batch_size):
        # 获取当前batch中attention_mask为1的位置
        valid_mask = attention_mask[i] == 1
        
        # 提取有效位置的entropy值
        valid_entropy = entropy[i][valid_mask]
        
        if valid_entropy.numel() == 0:
            # 如果没有有效位置，使用默认阈值
            thresholds.append(torch.tensor(0.0, device=entropy.device))
        else:
            # 计算top k的索引数量
            k = max(1, int(valid_entropy.numel() * topk))
            
            # 获取top k的阈值（第k大的值）
            topk_values, _ = torch.topk(valid_entropy, k, largest=True)
            threshold = topk_values[-1]  # 取第k大的值作为阈值
            thresholds.append(threshold)
    
    return torch.stack(thresholds)


def create_modified_forward():
    """
    创建修改后的forward函数
    这里可以添加你需要的自定义逻辑
    """
    def modified_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """
        修改后的forward函数
        在这里添加你的自定义逻辑
        """
        print("=== 使用修改后的forward函数 ===")
        # 原始forward函数的逻辑
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
  
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 准备mask参数
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # 创建位置嵌入
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 遍历decoder层
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
    
    return modified_forward


def create_modified_causal_lm_forward():
    def modified_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        do_span: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """
        修改后的CausalLM forward函数，支持span生成
        """
        print("=== 开始执行修改后的CausalLM forward函数 ===")
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        use_cache = True
        past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)
  
        # 调用模型的transformer部分
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            # inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        # 只计算必要的logits，如果不计算loss则不上转换为float
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if do_span:
            # entropy cut span
            probs = torch.softmax(logits, dim=-1)
            # batch size, seq len, vocab size
            ground_probs = probs * torch.log(probs)
            entropy = -torch.sum(ground_probs, dim=-1)
            entropy = entropy[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()
            entropy_topk = get_topk_threshold(entropy, attention_mask, topk=0.2)
            entropy_threshold = entropy_topk
            
            batch_starts, batch_ends = predict_span_by_similarity(hidden_states, threshold_method="mean", threshold_sensitivity=1.0, min_span_len=1, max_span_len=1, threshold=-entropy_threshold, similarity=-entropy) 
            
            batch_starts, batch_ends, batch_position_ids, batch_span_position = repeat_batch_starts_ends(batch_starts, batch_ends)
            
            batch_starts = torch.tensor(batch_starts, device=input_ids.device)
            
            span_to_kv_mask, intra_span_mask = batch_create_attention_masks_from_spans(batch_starts, past_key_values.get_seq_length(), dtype=torch.float32, intra_span_causal=False)
            
            # span hidden states
            batch_span_position = torch.tensor(batch_span_position, dtype=torch.long, device=input_ids.device)
            span_hidden_states = self.span_embedding(batch_span_position)
            span_attention_mask = torch.cat([span_to_kv_mask, intra_span_mask], dim=-1)
            original_attn_implementation = self.config._attn_implementation
            self.config._attn_implementation = "eager"
            # last layer span attention
            span_hidden_states = self.model.layers[-1](
                span_hidden_states,
                # attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                attention_mask=span_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            self.config._attn_implementation = original_attn_implementation
            span_logits = self.lm_head(span_hidden_states)
        
        loss = None
        if labels is not None:
            print("计算损失...")
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            if do_span:
                span_loss = self.loss_function(logits=span_logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
                print(f"损失值: {loss}")
                print(f"span损失值: {span_loss}")
                loss = loss + 0.1 * span_loss

        if do_span:
            # batch size, seq len, vocab size
            logits = torch.cat([logits, span_logits], dim=1)
        print("=== CausalLM forward函数执行完成 ===")
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    return modified_forward


def initialize_model_and_replace_forward(model_path: str, device: str = "cuda:0"):
    """
    初始化模型并替换forward函数
    
    Args:
        model_path: 模型路径
        device: 设备
    """
    print(f"正在从 {model_path} 加载模型...")
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = Qwen2ForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = Qwen2ForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model_device = model.device
    model_dtype = model.dtype
    print("模型加载完成！")
    
    # 获取模型的transformer部分（Qwen2Model）
    transformer_model = model.model
    
    # 保存原始的forward函数（可选）
    original_forward = transformer_model.forward
    
    # 创建修改后的forward函数
    modified_forward_func = create_modified_forward()
    
    # 将修改后的forward函数绑定到模型实例
    # 使用types.MethodType来正确绑定方法
    import types
    transformer_model.forward = types.MethodType(modified_forward_func, transformer_model)
    
    print("Forward函数已成功替换为修改后的版本！")

    # 同样替换CausalLM的forward函数
    modified_causal_lm_forward = create_modified_causal_lm_forward()
    model.forward = types.MethodType(modified_causal_lm_forward, model)
    
    # 给模型加入span embedding 参数
    max_span_len = 128
    embedding_dim = model.config.hidden_size
    # model.span_embedding = torch.nn.Parameter(torch.randn(max_span_len, embedding_dim))
    model.span_embedding = nn.Embedding(max_span_len, embedding_dim, dtype=model_dtype, device=model_device)
    # 使用合理的初始化方法初始化span_embedding
    # torch.nn.init.normal_(model.span_embedding, mean=0.0, std=0.02)
    
    # 验证span_embedding是否可训练
    print(f"span_embedding是否可训练: {model.span_embedding.weight.requires_grad}")
    print(f"span_embedding形状: {model.span_embedding.weight.shape}")
    print("CausalLM Forward函数也已成功替换为修改后的版本！")
    
    return model, tokenizer, original_forward


def test_modified_model():
    """
    测试修改后的模型
    """
    model_path = "/data/yangzhenfei/llm_checkpoint/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # 初始化模型并替换forward函数
    model, tokenizer, original_forward = initialize_model_and_replace_forward(model_path)
    
    # 测试文本
    text = [
    '''天津是中国四大直辖市之一，位于华北平原东部、海河流域下游，环渤海地区的中心。天津是一座具有深厚历史文化底蕴的城市，有着"津门故里"的美誉。这里既有五大道的欧式建筑群，又有古文化街的传统风貌；既有现代化的滨海新区，又有古色古香的天津古城。天津不仅以美食闻名，特色小吃如狗不理包子、耳朵眼炸糕享誉全国，还以相声、评书等传统曲艺而著称。作为北方重要的港口城市，天津在中国近现代史上扮演了重要角色。''',
    '''北京是中国的首都，位于华北平原北部，是中国的政治、文化中心。北京有着悠久的历史和丰富的文化遗产，是明清两代的帝都，留下了许多著名的古建筑和历史遗迹。北京也是现代中国的政治、文化和国际交流中心，拥有众多的高校、研究机构和国际组织。北京以其独特的历史风貌、现代化的城市建设和丰富的文化活动而闻名于世。''',
#     '''
# 1. **Define Variables:**
# Let the volume of the first container be $A$ and the volume of the second container be $B$.

# 2. **Set Up the Equation:**
# Since Alicia poured $\frac{5}{6}$ of the first container into the second container, and this amount filled $\frac{3}{4}$ of the second container, we can write the equation:
# \[
# \frac{5}{6}A = \frac{3}{4}B
# \]

# 3. **Solve for the Ratio $\frac{A}{B}$:**
# To find the ratio of the volume of the first container to the volume of the second container, we rearrange the equation:
# \[
# \frac{A}{B} = \frac{\frac{3}{4}B}{\frac{5}{6}A} = \frac{\frac{3}{4}}{\frac{5}{6}}
# \]
# Simplifying the right-hand side, we multiply by the reciprocal of $\frac{5}{6}$:
# \[
# \frac{3}{4} \times \frac{6}{5} = \frac{3 \times 6}{4 \times 5} = \frac{18}{20} = \frac{9}{10}
# \]

# 4. **Conclusion:**
# The ratio of the volume of the first container to the volume of the second container is $\boxed{\textbf{(D) }\frac{9}{10}}$. This ratio is less than 1, indicating that the first container is smaller than the second container.
#     '''
    ]
    
    # 将文本转换为模型输入格式
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True, padding=True, padding_side="left").to("cuda:0")
    
    print("\n开始测试修改后的模型...")
    # print(f"inputs: {inputs}")
    # 使用修改后的forward函数进行前向传播
    outputs = model(**inputs, labels=inputs["input_ids"])
    
    # 获取loss和logits
    loss = outputs.loss
    logits = outputs.logits
    
    print(f"\n测试结果:")
    print(f"Loss: {loss.item()}")
    print(f"Output shape: {logits.shape}")
    
    # 如果需要恢复原始forward函数
    # model.model.forward = original_forward
    # print("已恢复原始forward函数")
    
    return model, tokenizer


if __name__ == "__main__":
    # 运行测试
    model, tokenizer = test_modified_model() 