import torch
import torch.nn.functional as F

def calculate_similarity(auto_hidden):
    """
    计算 auto_hidden 中相邻向量之间的余弦相似度。

    Args:
        auto_hidden (torch.Tensor): 自动解码器隐藏状态，形状 (batch_size, sequence_length, hidden_size).

    Returns:
        torch.Tensor: 相似度矩阵，形状 (batch_size, sequence_length - 1).
    """
    # 当前token和下一个token的相似度
    # 计算余弦相似度
    print(f"auto_hidden shape: {auto_hidden.shape}")
    cos_similarity = F.cosine_similarity(auto_hidden[:, :-1, :], auto_hidden[:, 1:, :], dim=-1)
    # 计算点乘相似度
    dot_similarity = torch.mean(auto_hidden[:, :-1, :] * auto_hidden[:, 1:, :], dim=-1)
    # 将两种相似度结合
    # similarity = (cos_similarity + dot_similarity) / 2
    similarity = cos_similarity
    return similarity

def calculate_threshold(similarity, method='mean', sensitivity=1.0):
    """
    计算相似度阈值。

    Args:
        similarity (torch.Tensor): 相似度矩阵，形状 (batch_size, sequence_length - 1).
        method (str): 阈值计算方法，可选 'mean', 'median'.
        sensitivity (float): 阈值敏感度，用于调节阈值大小。

    Returns:
        torch.Tensor: 阈值向量，形状 (batch_size,).
    """
    if method == 'mean':
        threshold = similarity.mean(dim=-1) * sensitivity
    elif method == 'median':
        threshold = torch.median(similarity, dim=-1).values * sensitivity
    else:
        raise ValueError(f"Invalid threshold method: {method}")
    return threshold

def cut_span(similarity, threshold, min_span_len=2, max_span_len=5):
    """
    根据相似度和阈值切割 span。
    
    Args:
        similarity (torch.Tensor): 相似度向量，形状 (sequence_length - 1,).
                                  similarity[i] 表示位置 i 和 i+1 之间的相似度
        threshold (float): 阈值.
        min_span_len (int): 最小 span 长度.
        max_span_len (int): 最大 span 长度.
        
    Returns:
        tuple: (starts, ends) span 起始和结束位置列表。
    """
    if similarity is None or len(similarity) == 0:
        return [0], [1]
        
    sequence_length = len(similarity) + 1
    
    starts = []
    ends = []
    current_start = 0
    
    for i in range(len(similarity)):
        current_span_len = i + 1 - current_start
        
        # 确定切分条件
        should_cut = False
        # 条件1: 相似度低于阈值
        if similarity[i] < threshold:
            should_cut = True
        # 条件2: span长度达到最大限制
        if current_span_len >= max_span_len:
            should_cut = True
        
        # 如果满足切分条件，并且生成的span长度不小于最小长度，则进行切分
        if should_cut and current_span_len >= min_span_len:
            starts.append(current_start)
            ends.append(i + 1)
            current_start = i + 1
            
    # 处理最后一个 span
    if current_start < sequence_length:
        final_span_len = sequence_length - current_start
        
        if final_span_len >= min_span_len or not starts:
            # 最后一个span长度满足最小要求，或者这是唯一的span（必须保留）
            starts.append(current_start)
            ends.append(sequence_length)
        else:
            # 最后一个span太短，合并到前一个span
            ends[-1] = sequence_length
            
    return starts, ends


def predict_span_by_similarity(auto_hidden, threshold_method='mean', threshold_sensitivity=1.0, min_span_len=1, max_span_len=20, threshold = None, similarity = None):
    """
    基于 auto_hidden 的相邻向量相似度预测文本片段 (spans)。

    Args:
        auto_hidden (torch.Tensor): 自动解码器隐藏状态，形状 (batch_size, sequence_length, hidden_size).
        threshold_method (str): 阈值计算方法，可选 'mean', 'median'.
        threshold_sensitivity (float): 阈值敏感度。
        min_span_len (int): 最小 span 长度.
        max_span_len (int): 最大 span 长度.

    Returns:
        tuple: (batch_starts, batch_ends) 批次 span 起始和结束位置列表。
    """
    if similarity is None:
        similarity = calculate_similarity(auto_hidden)
    if threshold is None:
        threshold = calculate_threshold(similarity, method=threshold_method, sensitivity=threshold_sensitivity)

    batch_starts = []
    batch_ends = []
    for i in range(similarity.shape[0]):
        starts, ends = cut_span(similarity[i], threshold[i].item(), min_span_len, max_span_len)
        batch_starts.append(starts)
        batch_ends.append(ends)
    return batch_starts, batch_ends


def repeat_batch_starts_ends(batch_starts, batch_ends):
    batch_size = len(batch_starts)
    max_span_len = max(max(ends) - min(starts) for starts, ends in zip(batch_starts, batch_ends))
    new_batch_starts = []
    new_batch_ends = []
    new_batch_position_ids = []
    new_batch_span_position = []

    for starts, ends in zip(batch_starts, batch_ends):
        # 按照span长度复制start和end
        new_starts = []
        new_ends = []
        new_position_ids = []
        new_span_position = []
        for s, e in zip(starts, ends):
            span_len = e - s
            for idx in range(span_len):
                new_starts.append(s)
                new_ends.append(e)
                new_position_ids.append(s + idx)
                new_span_position.append(idx)

        # 如果当前span长度小于max_span_len, 需要padding到max_span_len
        while len(new_starts) < max_span_len:
            new_starts.append(0)
            new_ends.append(0)
            new_position_ids.append(0)
            new_span_position.append(0)

        new_batch_starts.append(new_starts)
        new_batch_ends.append(new_ends)
        new_batch_position_ids.append(new_position_ids)
        new_batch_span_position.append(new_span_position)
    return new_batch_starts, new_batch_ends, new_batch_position_ids, new_batch_span_position


