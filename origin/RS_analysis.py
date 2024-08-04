import PIL
import PIL.Image
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import pdb

# 示例阈值，这些阈值可以根据训练数据进行优化
thresholds = {
    'rm_threshold': 100,
    'sm_threshold': 50,
    'r_m_threshold': 100,
    's_m_threshold': 50
}

def read_image(image_path):
    # 读取图像并转换为灰度
    image = Image.open(image_path).convert('L')
    image = np.array(image)
    return image

def block_process(image, block_size):
    
    # 将图像分割成小块
    blocks = np.array([img_block for img_block in image.reshape(-1, block_size, block_size)])
    return blocks

def flip_bits(block):
    # 对块中的每个像素的最低有效位进行翻转
    block *= 255
    block_int = block.astype(np.uint8)
    # import pdb; pdb.set_trace()
    return np.bitwise_xor(block_int, 1)

def smoothness(block):
    # 计算平滑度函数
    return np.sum(np.abs(block[1:] - block[:-1]))

def estimate_steganography(rm, sm, r_m, s_m, thresholds):
    """
    根据RS分析的统计量估计图像是否被隐写。
    
    参数:
    - rm: 正则模型统计量
    - sm: 隐写模型统计量
    - r_m: 反转模型统计量
    - s_m: 隐写反转模型统计量
    - thresholds: 阈值字典，包含'rm_threshold', 'sm_threshold', 'r_m_threshold', 's_m_threshold'
    
    返回:
    - True 如果图像很可能被隐写
    - False 如果图像很可能未被隐写
    """
    # 设定阈值，这些阈值可以根据实际情况调整
    rm_threshold = thresholds.get('rm_threshold', 0)
    sm_threshold = thresholds.get('sm_threshold', 0)
    r_m_threshold = thresholds.get('r_m_threshold', 0)
    s_m_threshold = thresholds.get('s_m_threshold', 0)
    
    # 如果RM和r_m的值低于阈值，而SM和S-M的值高于阈值，则认为图像可能被隐写
    if (rm < rm_threshold) and (r_m < r_m_threshold) and (sm > sm_threshold) and (s_m > s_m_threshold):
        return True
    else:
        return False

def rs_analysis(image):
    
    # image 需要转化为 numpy 数组
    if isinstance(image, PIL.Image.Image):
        image_ = np.array(image)
    elif isinstance(image, torch.Tensor):
        image_ = image.clone().to('cpu').detach().numpy()
        # # 使用ToPILImage转换
        # transform_to_pil_image = transforms.ToPILImage()

        # # 将张量转换为PIL图像
        # image_PIL = transform_to_pil_image(image)
    else:
        image_ = image
    blocks = block_process(image_, 8)  # 假设块大小为8x8
    rm, sm, r_m, s_m = 0, 0, 0, 0
    for block in blocks:
        original_smoothness = smoothness(block)
        flipped_block = flip_bits(block)
        flipped_smoothness = smoothness(flipped_block)
        
        if original_smoothness > flipped_smoothness:
            rm += 1
        elif original_smoothness < flipped_smoothness:
            sm += 1
        
        if smoothness(flipped_block) > smoothness(block):
            r_m += 1
        elif smoothness(flipped_block) < smoothness(block):
            s_m += 1
    
    # 调用函数判断图像是否被隐写
    is_stegano = estimate_steganography(rm, sm, r_m, s_m, thresholds)
    
    return is_stegano


if __name__ == '__main__':
    image_path = 'path_to_your_image.jpg'
    image = read_image(image_path)
    
    results = rs_analysis(blocks, flip_bits)
    print(f"RM: {results[0]}, SM: {results[1]}, R-M: {results[2]}, S-M: {results[3]}")