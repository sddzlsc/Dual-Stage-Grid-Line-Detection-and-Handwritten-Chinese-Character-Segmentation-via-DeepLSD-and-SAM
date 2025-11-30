"""
使用DeepLSD检测图像中的横线和竖线 - 二次检测去干扰模式
基于DeepLSD-main文件夹中的代码

🔧 调整参数去除干扰线:
--------------------
在main()函数的配置区域可以调整以下参数：

1. GRAD_THRESH (梯度阈值)
   - 默认值: 3 (正常检测)
   - 推荐值: 5-10 (去除更多干扰线)
   - 说明: 值越高，检测越严格，短线和弱线会被过滤

2. MIN_LENGTH_RATIO (横线最小长度比例)
   - 默认值: 0.05 (图像对角线的5%)
   - 推荐值: 0.03-0.10
   - 说明: 二次检测时过滤短于此长度的横线

3. ENDPOINT_DISTANCE_THRESHOLD (竖线端点距离阈值)
   - 默认值: 10 像素
   - 推荐值: 5-15
   - 说明: 只保留上下端点都在此距离内接近横线的竖线
   
4. MERGE_LINES (合并相近线段)
   - 默认值: True
   - 说明: 是否自动合并距离很近的线段

5. SAM参数 (控制检测密度，避免过度分割)
   - SAM_POINTS_PER_SIDE: 采样点密度
     · 默认值: 40 (较密)
     · 推荐值: 16=稀疏，32=适中，40=较密，64=密集
     · 说明: 值越大检测越密集，mask数量越多
   
   - SAM_PRED_IOU_THRESH: 预测IOU阈值
     · 默认值: 0.82
     · 推荐值: 0.8-0.95
     · 说明: 值越高要求mask质量越高，过滤更多低质量mask
   
   - SAM_STABILITY_THRESH: 稳定性阈值
     · 默认值: 0.88
     · 推荐值: 0.85-0.95
     · 说明: 值越高过滤越严格
   
   - SAM_MIN_AREA: 最小mask面积
     · 默认值: 70 像素
     · 推荐值: 50-200
     · 说明: 过滤小于此面积的碎片mask

6. AREA_TOLERANCE (面积过滤容差)
   - 默认值: 0.5 (中位数的±50%)
   - 推荐值: 0.3-0.7
   - 说明: 根据面积中位数过滤异常mask
     · 0.5 表示保留面积在 [中位数×0.5, 中位数×1.5] 范围内的mask
     · 值越小过滤越严格，越大越宽松

7. BBOX_ALPHA (矩形框透明度)
   - 默认值: 0.6 (半透明)
   - 推荐值: 0.5-0.8
   - 说明: 所有矩形框的透明度
     · 0.0 = 完全透明（看不见）
     · 1.0 = 完全不透明（实心）
     · 所有检测框统一使用1px单像素宽 + 半透明效果

🎯 工作流程:
-----------
第1步: DeepLSD检测原图 → 得到所有线段
第2步: 绘制纯线段图 (all_lines_raw.png)
第3步: 分离横线和竖线，分别处理：
       - 横线图 → DeepLSD二次检测 + 长度过滤 → 去除短干扰线
       - 竖线图 → DeepLSD二次检测 + 端点过滤 → 只保留端点接近横线的竖线
第4步: 生成干净的线段图 (all_lines.png)
第5步: 使用SAM检测作文格 → 生成所有候选mask
第6步: 按面积中位数过滤异常mask → 得到终版检测框 (final_bboxes.png，蓝色)
第7步: 使用SAM box prompt在每个作文格内检测字符 (char_detection.png，红色)
       - 每个作文格框作为box prompt输入SAM
       - SAM在框内精确分割出字符mask
       - 过滤空白格子（无字符的不显示）
       - 用红色1px边框标注字符边界
第8步: 生成合并检测框图像 (combined_detection.png)
       - 蓝色框 = 所有作文格外框
       - 红色框 = 有字符的字符框（内框）
       - 空白格子无红色框

💡 竖线端点过滤原理:
-------------------
- 作文格的字格分隔线（短竖线）的特点：上下两端都与横线相交
- 干扰竖线的特点：至少有一端悬空，不接近横线
- 过滤方法：计算竖线的上下端点到所有横线的最短距离
- 判断标准：只保留上下两端都在阈值距离内接近横线的竖线
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import json

# 尝试导入PIL（支持TIF等格式）
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 尝试导入SAM
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ SAM未安装，跳过SAM处理")

# 添加DeepLSD路径到sys.path
DEEPLSD_PATH = os.path.join(os.path.dirname(__file__), "DeepLSD")
if os.path.exists(DEEPLSD_PATH):
    sys.path.insert(0, DEEPLSD_PATH)
    print(f"✅ 添加DeepLSD路径: {DEEPLSD_PATH}")
else:
    print(f"⚠️ DeepLSD路径不存在: {DEEPLSD_PATH}")
    DEEPLSD_PATH = None

# 尝试导入DeepLSD
DEEPLSD_AVAILABLE = False
try:
    from deeplsd.models.deeplsd_inference import DeepLSD
    DEEPLSD_AVAILABLE = True
    print("✅ DeepLSD库导入成功")
except ImportError as e:
    print(f"⚠️ DeepLSD导入失败: {e}")
    print("💡 提示: 可能需要安装依赖或模型权重文件")


def load_deeplsd_model(model_path=None, device='cuda', grad_thresh=3, merge_lines=False):
    """
    加载DeepLSD模型
    
    Args:
        model_path: 模型权重路径（.tar文件）
        device: 设备（'cuda' 或 'cpu'）
        grad_thresh: 梯度阈值，越高越严格（默认3，可设置5-10去除更多干扰）
        merge_lines: 是否合并相近的线段
    
    Returns:
        net: 加载好的模型
        device: 使用的设备
    """
    if not DEEPLSD_AVAILABLE:
        raise ImportError("DeepLSD未安装，无法使用")
    
    # 检查设备
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠️ CUDA不可用，使用CPU模式")
    
    device = torch.device(device)
    
    # 如果没有提供模型路径，尝试默认路径
    if model_path is None:
        possible_paths = [
            "DeepLSD/weights/deeplsd_wireframe.tar",
            "DeepLSD/weights/deeplsd_md.tar",
            "DeepLSD-main/weights/deeplsd_wireframe.tar",
            "DeepLSD-main/weights/deeplsd_md.tar",
            "weights/deeplsd_wireframe.tar",
            "weights/deeplsd_md.tar",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                "未找到DeepLSD模型权重文件。\n"
                "请下载模型到以下位置之一：\n"
                "  - DeepLSD/weights/deeplsd_md.tar (推荐)\n"
                "  - DeepLSD/weights/deeplsd_wireframe.tar\n"
                "\n下载地址: https://cvg-data.inf.ethz.ch/DeepLSD/"
            )
    
    print(f"📦 加载模型: {model_path}")
    
    # 模型配置
    conf = {
        'detect_lines': True,
        'line_detection_params': {
            'merge': merge_lines,  # 是否合并相近的线
            'filtering': True,  # 过滤异常线
            'grad_thresh': grad_thresh,  # 梯度阈值（可调整）
            'grad_nfa': True,  # 使用NFA评分
        }
    }
    
    # 加载模型
    ckpt = torch.load(str(model_path), map_location='cuda',weights_only=False)
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()
    
    print(f"✅ 模型加载完成 (grad_thresh={grad_thresh}, merge={merge_lines})")
    return net, device


def detect_lines_deeplsd(image_path, model, device, min_length=0, score_thresh=0.0, is_second_pass=False):
    """
    使用DeepLSD检测线段
    
    Args:
        image_path: 图像路径
        model: 加载好的DeepLSD模型
        device: 设备
        min_length: 最小线段长度（像素），0表示不限制
        score_thresh: 线段置信度阈值（0-1），0表示不限制
        is_second_pass: 是否为二次检测（用于纯线段图，会使用更严格的参数）
    
    Returns:
        lines: 检测到的线段，格式为 numpy array (N, 2, 2)
               每个线段是 [[x1, y1], [x2, y2]]
    """
    # 读取图像（支持png/tif等格式）
    img = cv2.imread(image_path)
    
    if img is None:
        # OpenCV加载失败，尝试PIL（支持更多格式如16位TIF）
        if PIL_AVAILABLE:
            try:
                if not is_second_pass:
                    print(f"      OpenCV加载失败，尝试PIL...")
                pil_image = PILImage.open(image_path)
                if not is_second_pass:
                    print(f"      PIL模式: {pil_image.mode}, 尺寸: {pil_image.size}")
                
                # 转换为RGB
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                if not is_second_pass:
                    print(f"      ✓ PIL加载成功")
            except Exception as e:
                raise ValueError(f"无法读取图像: {image_path}, 错误: {e}")
        else:
            raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为RGB灰度图
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    if not is_second_pass:
        print(f"🖼️  图像尺寸: {gray_img.shape[1]} x {gray_img.shape[0]}")
    
    # 准备输入
    img_tensor = torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.
    
    # 检测线段
    if is_second_pass:
        print(f"      🔍 开始二次检测（使用严格参数）...")
    else:
        print("🔍 开始检测线段...")
    
    with torch.no_grad():
        inputs = {'image': img_tensor}
        outputs = model(inputs)
        pred_lines = outputs['lines'][0]  # (N, 2, 2)
    
    # 过滤线段（根据长度和置信度）
    if min_length > 0 or score_thresh > 0:
        filtered_lines = []
        for i, line in enumerate(pred_lines):
            # 计算线段长度
            pt1, pt2 = line[0], line[1]
            length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            
            # 获取线段得分（如果有）
            score = outputs.get('line_scores', [1.0] * len(pred_lines))[i] if 'line_scores' in outputs else 1.0
            
            # 应用过滤条件
            if length >= min_length and score >= score_thresh:
                filtered_lines.append(line)
        
        pred_lines = np.array(filtered_lines) if len(filtered_lines) > 0 else np.array([]).reshape(0, 2, 2)
        
        if is_second_pass:
            print(f"      📊 二次检测结果: {len(pred_lines)} 条线段（过滤后）")
        else:
            print(f"✅ 检测到 {len(pred_lines)} 条线段（过滤后）")
    else:
        if is_second_pass:
            print(f"      📊 二次检测结果: {len(pred_lines)} 条线段")
        else:
            print(f"✅ 检测到 {len(pred_lines)} 条线段")
    
    return pred_lines, gray_img.shape


def convert_lines_format(lines):
    """
    将DeepLSD的输出格式转换为标准格式
    
    Args:
        lines: DeepLSD输出 (N, 2, 2)，每个线段是 [[x1, y1], [x2, y2]]
    
    Returns:
        lines_standard: 标准格式列表 [(x1, y1, x2, y2), ...]
    """
    lines_standard = []
    for line in lines:
        # line shape: (2, 2) -> [[x1, y1], [x2, y2]]
        pt1 = line[0]  # [x1, y1]
        pt2 = line[1]  # [x2, y2]
        lines_standard.append((float(pt1[0]), float(pt1[1]), 
                               float(pt2[0]), float(pt2[1])))
    return lines_standard


def filter_horizontal_vertical(lines_standard, angle_threshold=15):
    """
    将线段分类为横线和竖线
    
    Args:
        lines_standard: 线段列表 [(x1, y1, x2, y2), ...]
        angle_threshold: 角度阈值（度），默认15度
    
    Returns:
        horizontal_lines: 横线列表
        vertical_lines: 竖线列表
        other_lines: 其他方向的线段
    """
    horizontal_lines = []
    vertical_lines = []
    other_lines = []
    
    for x1, y1, x2, y2 in lines_standard:
        # 计算角度
        dx = x2 - x1
        dy = y2 - y1
        
        # 避免除零
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        
        # 计算角度（弧度转度）
        angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
        
        # 归一化角度到0-90度
        if angle > 90:
            angle = 180 - angle
        
        # 分类
        if angle < angle_threshold:  # 接近水平
            horizontal_lines.append((x1, y1, x2, y2))
        elif angle > (90 - angle_threshold):  # 接近垂直
            vertical_lines.append((x1, y1, x2, y2))
        else:
            other_lines.append((x1, y1, x2, y2))
    
    return horizontal_lines, vertical_lines, other_lines


def visualize_lines(image_path, horizontal_lines, vertical_lines, output_path):
    """
    可视化检测到的横线和竖线
    
    Args:
        image_path: 原始图像路径
        horizontal_lines: 横线列表
        vertical_lines: 竖线列表
        output_path: 输出图像路径
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 无法读取图像: {image_path}")
        return
    
    # 绘制横线（红色）
    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    # 绘制竖线（绿色）
    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)
    print(f"💾 可视化结果已保存: {output_path}")


def save_results(txt_path, horizontal_lines, vertical_lines, image_name):
    """保存检测结果到文本文件"""
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"# DeepLSD线段检测结果 - {image_name}\n")
        f.write(f"# 检测时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"# 横线数量: {len(horizontal_lines)}\n")
        f.write("# 格式: line_id, x1, y1, x2, y2\n")
        for i, (x1, y1, x2, y2) in enumerate(horizontal_lines):
            f.write(f"horizontal_{i:04d}: {x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}\n")
        
        f.write(f"\n# 竖线数量: {len(vertical_lines)}\n")
        f.write("# 格式: line_id, x1, y1, x2, y2\n")
        for i, (x1, y1, x2, y2) in enumerate(vertical_lines):
            f.write(f"vertical_{i:04d}: {x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}\n")
    
    print(f"💾 结果已保存: {txt_path}")


def filter_grid_lines(lines, line_type, img_shape):
    """
    过滤网格线：保留长的网格线，去掉方格内的短干扰线和斜线
    
    Args:
        lines: 线段列表 [(x1, y1, x2, y2), ...]
        line_type: 'horizontal' 或 'vertical'
        img_shape: 图像尺寸 (height, width)
    
    Returns:
        filtered_lines: 过滤后的线段列表
    """
    if len(lines) == 0:
        return []
    
    img_h, img_w = img_shape[:2]
    
    # 计算每条线段的长度和角度
    filtered_lines = []
    line_lengths = []
    
    for x1, y1, x2, y2 in lines:
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length == 0:
            continue
        
        # 计算角度（相对于水平方向的夹角）
        dx = x2 - x1
        dy = y2 - y1
        angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
        
        # 规范角度到0-90度
        if angle > 90:
            angle = 180 - angle
        
        # 判断是否为斜线（角度不在接近0度或90度的范围内）
        # 横线：角度应该接近0度（0-10度或170-180度）
        # 竖线：角度应该接近90度（80-100度）
        is_valid_angle = False
        if line_type == 'horizontal':
            # 横线：角度接近0度或180度（规范后0-10度或80-90度）
            is_valid_angle = angle <= 10 or angle >= 85
        else:  # vertical
            # 竖线：角度接近90度（规范后80-90度）
            is_valid_angle = angle >= 80
        
        if is_valid_angle:
            filtered_lines.append((x1, y1, x2, y2))
            line_lengths.append(length)
    
    if len(line_lengths) == 0:
        return []
    
    line_lengths = np.array(line_lengths)
    
    # 计算长度统计
    median_length = np.median(line_lengths)
    
    # 设置最小长度阈值为中位数的50%
    min_length = median_length * 0.5
    
    print(f"      中位数长度: {median_length:.1f}px, 最小阈值: {min_length:.1f}px ({'横线' if line_type == 'horizontal' else '竖线'})")
    
    # 再次筛选：保留长度足够长的线段
    final_filtered_lines = []
    for i, (x1, y1, x2, y2) in enumerate(filtered_lines):
        length = line_lengths[i]
        if length >= min_length:
            final_filtered_lines.append((x1, y1, x2, y2))
    
    return final_filtered_lines


def is_point_near_lines(point, lines, threshold=5):
    """
    检查点是否接近任何一条线
    
    Args:
        point: (x, y) 坐标
        lines: 线段列表 [(x1, y1, x2, y2), ...]
        threshold: 距离阈值（像素）
    
    Returns:
        bool: True如果点接近任何一条线
    """
    px, py = point
    
    for x1, y1, x2, y2 in lines:
        # 计算点到线段的最短距离
        # 使用向量投影方法
        line_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([px - x1, py - y1])
        
        line_len_sq = line_vec[0]**2 + line_vec[1]**2
        if line_len_sq == 0:
            # 线段退化为点
            dist = np.sqrt((px - x1)**2 + (py - y1)**2)
        else:
            # 投影参数t (0<=t<=1表示投影点在线段上)
            t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
            # 投影点坐标
            proj_x = x1 + t * line_vec[0]
            proj_y = y1 + t * line_vec[1]
            # 点到投影点的距离
            dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        
        if dist <= threshold:
            return True
    
    return False


def filter_vertical_lines_by_endpoints(vertical_lines, horizontal_lines, distance_threshold=10):
    """
    根据端点是否接近其他线来过滤竖线
    只保留上下两端都接近横线的竖线（作文格的字格分隔线）
    
    Args:
        vertical_lines: 竖线列表 [(x1, y1, x2, y2), ...]
        horizontal_lines: 横线列表 [(x1, y1, x2, y2), ...]
        distance_threshold: 端点到线的距离阈值（像素）
    
    Returns:
        filtered_lines: 过滤后的竖线列表
    """
    if len(vertical_lines) == 0:
        return []
    
    if len(horizontal_lines) == 0:
        print(f"      ⚠️ 没有横线，无法过滤竖线端点")
        return vertical_lines
    
    filtered_lines = []
    
    for x1, y1, x2, y2 in vertical_lines:
        # 确保y1是上端点，y2是下端点
        if y1 > y2:
            y1, y2 = y2, y1
        
        # 检查上端点和下端点是否都接近横线
        top_point = (x1, y1)
        bottom_point = (x2, y2)
        
        top_near = is_point_near_lines(top_point, horizontal_lines, distance_threshold)
        bottom_near = is_point_near_lines(bottom_point, horizontal_lines, distance_threshold)
        
        # 只保留上下两端都接近横线的竖线
        if top_near and bottom_near:
            filtered_lines.append((x1, y1, x2, y2))
    
    print(f"      📊 端点过滤: {len(vertical_lines)} -> {len(filtered_lines)} (保留上下端都接近横线的)")
    
    return filtered_lines


def filter_vertical_lines(vertical_lines, img_shape):
    """
    筛选竖线：保留长度几乎相等的短竖线和特长的竖线
    
    Args:
        vertical_lines: 竖线列表 [(x1, y1, x2, y2), ...]
        img_shape: 图像尺寸 (height, width)
    
    Returns:
        (short_lines, long_lines): (短竖线列表, 特长竖线列表)
    """
    if len(vertical_lines) == 0:
        return [], []
    
    img_h, img_w = img_shape[:2]
    
    # 计算每条竖线的长度
    line_lengths = []
    for x1, y1, x2, y2 in vertical_lines:
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        line_lengths.append(length)
    
    line_lengths = np.array(line_lengths)
    
    # 找出长度几乎相等的短竖线
    # 使用中位数长度作为基准
    median_length = np.median(line_lengths)
    print(f"   📏 竖线长度统计: 中位数={median_length:.1f}px, 范围=[{line_lengths.min():.1f}, {line_lengths.max():.1f}]px")
    
    # 筛选出长度接近中位数的短竖线（±30%）和特长竖线（≥2倍中位数）
    short_lines = []
    long_lines = []
    
    for i, (x1, y1, x2, y2) in enumerate(vertical_lines):
        length = line_lengths[i]
        # 短竖线：长度在0.7-1.3倍中位数之间
        if median_length * 0.7 <= length <= median_length * 1.3:
            short_lines.append((x1, y1, x2, y2))
        # 特长竖线：长度大于中位数的2倍（可能是页面边缘线）
        elif length >= median_length * 2.0:
            long_lines.append((x1, y1, x2, y2))
    
    print(f"   ✅ 短竖线: {len(short_lines)} 条 (长度≈{median_length:.1f}px)")
    print(f"   ✅ 特长竖线: {len(long_lines)} 条 (长度≥{median_length*2.0:.1f}px)")
    
    # 按X坐标排序
    short_lines.sort(key=lambda line: (line[0] + line[2]) / 2)
    long_lines.sort(key=lambda line: (line[0] + line[2]) / 2)
    
    return short_lines, long_lines




def load_sam_model(checkpoint_path=None, model_type="vit_h", device="cuda"):
    """
    加载SAM模型（延迟加载，只在需要时加载一次）
    """
    if not SAM_AVAILABLE:
        raise ImportError("SAM未安装，无法使用")
    
    # 如果没有指定checkpoint路径，尝试自动查找
    if checkpoint_path is None:
        possible_paths = [
            "sam_vit_h_4b8939.pth",
            "sam_vit_l_0b3195.pth",
            "sam_vit_b_01ec64.pth",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                if "vit_h" in path:
                    model_type = "vit_h"
                elif "vit_l" in path:
                    model_type = "vit_l"
                elif "vit_b" in path:
                    model_type = "vit_b"
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError("找不到SAM模型文件，请下载并放在项目根目录")
    
    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("   ⚠️ CUDA不可用，使用CPU")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return sam, device


def process_char_detection_with_sam(original_image_path, grid_bboxes, output_dir, image_name, bbox_alpha=0.6):
    """
    使用SAM的box prompt模式在每个作文格框内检测字符
    
    Args:
        original_image_path: 原图路径
        grid_bboxes: 作文格检测框列表 [{"x": x, "y": y, "width": w, "height": h}, ...]
        output_dir: 输出目录
        image_name: 图像名称
        bbox_alpha: 矩形框透明度（0.0-1.0）
    
    Returns:
        char_results: 字符检测结果列表
    """
    if not SAM_AVAILABLE:
        print(f"   ⚠️ SAM未安装，无法进行字符检测")
        return []
    
    # 加载SAM模型
    try:
        sam, device = load_sam_model()
    except Exception as e:
        print(f"   ❌ SAM模型加载失败: {e}")
        return []
    
    # 读取原图
    image = cv2.imread(original_image_path)
    if image is None:
        print(f"   ❌ 无法读取图像: {original_image_path}")
        return []
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建SAM预测器（用于box prompt）
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    
    print(f"   🔍 开始字符检测（共{len(grid_bboxes)}个作文格）...")
    
    char_results = []
    # 创建overlay层用于半透明绘制
    overlay = image.copy()
    
    for grid_idx, grid_bbox in enumerate(grid_bboxes):
        grid_x = grid_bbox['x']
        grid_y = grid_bbox['y']
        grid_w = grid_bbox['width']
        grid_h = grid_bbox['height']
        
        # 准备box prompt（格式：[x1, y1, x2, y2]）
        box_prompt = np.array([grid_x, grid_y, grid_x + grid_w, grid_y + grid_h])
        
        # 使用SAM预测
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_prompt[None, :],  # shape: (1, 4)
            multimask_output=False  # 只输出一个最佳mask
        )
        
        # 取得分最高的mask
        if len(masks) > 0 and len(scores) > 0:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]  # shape: (H, W)
            score = scores[best_idx]
            
            # 从mask计算字符的边界框
            # 找到mask中所有True的点
            points = np.argwhere(mask)
            
            if len(points) > 0:
                # 计算最小外接矩形
                y_coords = points[:, 0]
                x_coords = points[:, 1]
                
                char_x1 = int(np.min(x_coords))
                char_y1 = int(np.min(y_coords))
                char_x2 = int(np.max(x_coords))
                char_y2 = int(np.max(y_coords))
                
                char_w = char_x2 - char_x1
                char_h = char_y2 - char_y1
                
                # 计算字符mask的实际面积
                char_area = np.sum(mask)
                
                # 过滤空白格子：如果字符面积太小或尺寸太小，认为是空白
                # 阈值：面积至少10像素，且宽高至少3像素
                min_char_area = 10
                min_char_size = 3
                
                if char_area >= min_char_area and char_w >= min_char_size and char_h >= min_char_size:
                    # 保存字符检测结果
                    char_results.append({
                        "grid_id": grid_idx,
                        "grid_bbox": {
                            "x": float(grid_x),
                            "y": float(grid_y),
                            "width": float(grid_w),
                            "height": float(grid_h)
                        },
                        "char_bbox": {
                            "x": char_x1,
                            "y": char_y1,
                            "width": char_w,
                            "height": char_h
                        },
                        "char_area": int(char_area),
                        "confidence": float(score)
                    })
                    
                    # 在overlay层上绘制红色1px边框
                    cv2.rectangle(overlay, (char_x1, char_y1), (char_x2, char_y2), (0, 0, 255), 1)
    
    # 叠加半透明效果
    vis_img = cv2.addWeighted(overlay, bbox_alpha, image, 1 - bbox_alpha, 0)
    
    print(f"   ✅ 字符检测完成，检测到 {len(char_results)} 个字符（已过滤空白格子）")
    
    # 保存字符检测可视化
    char_vis_path = os.path.join(output_dir, f"{image_name}_char_detection.png")
    cv2.imwrite(char_vis_path, vis_img)
    print(f"   ✅ 字符检测框已保存: {char_vis_path} (红色1px半透明，仅显示有字符的格子)")
    
    # 保存字符检测JSON
    char_json_path = os.path.join(output_dir, f"{image_name}_char_detection.json")
    char_json_data = {
        "image": image_name,
        "total_grids": len(grid_bboxes),
        "detected_chars": len(char_results),
        "method": "SAM_box_prompt",
        "chars": char_results
    }
    
    with open(char_json_path, 'w', encoding='utf-8') as f:
        json.dump(char_json_data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 字符检测JSON已保存: {char_json_path}")
    
    return char_results


def filter_masks_by_area(masks, area_tolerance=0.5):
    """
    根据面积中位数过滤掉异常大小的mask
    
    Args:
        masks: SAM生成的masks列表
        area_tolerance: 面积容差范围（相对中位数的比例），默认0.5（即中位数的50%-150%）
    
    Returns:
        filtered_masks: 过滤后的masks列表
    """
    if len(masks) == 0:
        return []
    
    # 统计所有mask的面积
    areas = [m['area'] for m in masks]
    
    if len(areas) == 0:
        return []
    
    # 计算面积中位数
    median_area = np.median(areas)
    
    # 计算面积范围
    area_min = median_area * (1 - area_tolerance)
    area_max = median_area * (1 + area_tolerance)
    
    print(f"   📊 面积统计: 中位数={median_area:.0f}, 范围=[{min(areas):.0f}, {max(areas):.0f}]")
    print(f"   📊 过滤范围: [{area_min:.0f}, {area_max:.0f}] (中位数±{area_tolerance*100:.0f}%)")
    
    # 过滤mask
    filtered_masks = []
    for m in masks:
        area = m['area']
        if area_min <= area <= area_max:
            filtered_masks.append(m)
    
    print(f"   ✅ 面积过滤: {len(masks)} -> {len(filtered_masks)} 个mask")
    
    return filtered_masks


def filter_grid_masks(masks, image_name, target_rows=20, target_cols=16):
    """
    筛选出规则的网格mask（20×16或17×16）
    基于面积统计规律：作文格面积基本一致，找到面积最集中的区域
    
    Args:
        masks: SAM生成的masks列表
        image_name: 图像名称（用于判断类型）
        target_rows: 目标行数（20或17）
        target_cols: 目标列数（16）
    
    Returns:
        filtered_masks: 筛选后的masks列表
    """
    if len(masks) == 0:
        return []
    
    # 统计所有mask的面积
    areas = [m['area'] for m in masks]
    
    if len(areas) == 0:
        return []
    
    # 过滤掉异常大的面积（可能是多个方格合并）
    max_reasonable_area = np.percentile(areas, 95)  # 95分位数作为最大合理面积
    reasonable_masks = [m for m in masks if m['area'] <= max_reasonable_area]
    reasonable_areas = [m['area'] for m in reasonable_masks]
    
    if len(reasonable_areas) == 0:
        return []
    
    print(f"   📊 面积统计: 范围=[{min(areas)}, {max(areas)}], 中位数={np.median(areas):.0f}")
    print(f"   📊 合理面积范围: [0, {max_reasonable_area:.0f}], 候选mask: {len(reasonable_masks)}")
    
    # 找到面积最集中的区域（作文格应该面积基本一致）
    # 使用直方图找到峰值
    hist, bins = np.histogram(reasonable_areas, bins=100)
    peak_idx = np.argmax(hist)
    peak_area = (bins[peak_idx] + bins[peak_idx + 1]) / 2
    
    # 计算面积的中位数和四分位距
    median_area = np.median(reasonable_areas)
    q1_area = np.percentile(reasonable_areas, 25)
    q3_area = np.percentile(reasonable_areas, 75)
    iqr_area = q3_area - q1_area
    
    # 使用IQR方法找到集中区域：Q1到Q3之间
    # 但稍微扩展一点，保留更多候选
    area_lower = max(q1_area - 0.5 * iqr_area, min(reasonable_areas) * 0.5)
    area_upper = min(q3_area + 0.5 * iqr_area, max(reasonable_areas) * 2)
    
    print(f"   📊 面积峰值: {peak_area:.0f}, 中位数={median_area:.0f}, IQR=[{q1_area:.0f}, {q3_area:.0f}]")
    print(f"   📊 作文格面积范围: [{area_lower:.0f}, {area_upper:.0f}]")
    
    # 筛选符合作文格面积的mask（放宽筛选条件）
    candidate_masks = []
    for m in reasonable_masks:
        area = m['area']
        bbox = m['bbox']
        w, h = bbox[2], bbox[3]
        aspect_ratio = w / h if h > 0 else 0
        
        # 筛选条件：面积在集中范围内，宽高比合理（0.3-3.0），并且面积不能太小
        if area >= area_lower and area <= area_upper and area >= 50 and 0.3 <= aspect_ratio <= 3.0:
            candidate_masks.append(m)
    
    print(f"   📊 符合作文格面积的mask: {len(candidate_masks)} 个")
    
    if len(candidate_masks) == 0:
        print(f"   ⚠️ 没有找到符合作文格面积的mask")
        return []
    
    main_masks = candidate_masks
    
    # 按位置排序，尝试组成网格
    main_masks.sort(key=lambda m: (m['bbox'][1], m['bbox'][0]))  # 先按Y，再按X
    
    # 计算mask的平均尺寸（用于计算间距）
    if len(main_masks) > 0:
        sizes = [(m['bbox'][2], m['bbox'][3]) for m in main_masks]
        median_w = np.median([s[0] for s in sizes])
        median_h = np.median([s[1] for s in sizes])
    else:
        median_w = 50
        median_h = 50
    
    # 计算平均间距
    if len(main_masks) > 1:
        # 按Y坐标排序，计算垂直间距
        y_coords = [m['bbox'][1] + m['bbox'][3]/2 for m in main_masks]
        sorted_y = sorted(y_coords)
        y_spacings = [sorted_y[i+1] - sorted_y[i] 
                     for i in range(len(sorted_y)-1) 
                     if sorted_y[i+1] - sorted_y[i] > 0 and sorted_y[i+1] - sorted_y[i] < median_h * 3]
        row_spacing = np.median(y_spacings) if len(y_spacings) > 0 else median_h * 1.2
        
        # 按X坐标排序，计算水平间距
        x_coords = [m['bbox'][0] + m['bbox'][2]/2 for m in main_masks]
        sorted_x = sorted(x_coords)
        x_spacings = [sorted_x[i+1] - sorted_x[i] 
                     for i in range(len(sorted_x)-1) 
                     if sorted_x[i+1] - sorted_x[i] > 0 and sorted_x[i+1] - sorted_x[i] < median_w * 3]
        col_spacing = np.median(x_spacings) if len(x_spacings) > 0 else median_w * 1.2
    else:
        row_spacing = median_h * 1.2
        col_spacing = median_w * 1.2
    
    # 聚类行和列
    y_coords = [m['bbox'][1] + m['bbox'][3]/2 for m in main_masks]
    x_coords = [m['bbox'][0] + m['bbox'][2]/2 for m in main_masks]
    
    # 对Y坐标聚类（行）
    from sklearn.cluster import DBSCAN
    y_array = np.array(y_coords).reshape(-1, 1)
    if row_spacing > 0:
        y_eps = float(row_spacing * 0.4)
    else:
        y_eps = float(median_h * 0.3) if median_h > 0 else 10.0
    y_eps = max(y_eps, 2.0)
    y_clustering = DBSCAN(eps=y_eps, min_samples=1).fit(y_array)
    
    # 对X坐标聚类（列）
    x_array = np.array(x_coords).reshape(-1, 1)
    if col_spacing > 0:
        x_eps = float(col_spacing * 0.4)
    else:
        x_eps = float(median_w * 0.3) if median_w > 0 else 10.0
    x_eps = max(x_eps, 2.0)
    x_clustering = DBSCAN(eps=x_eps, min_samples=1).fit(x_array)
    
    # 获取聚类后的行数和列数
    unique_y_labels = [l for l in np.unique(y_clustering.labels_) if l != -1]
    unique_x_labels = [l for l in np.unique(x_clustering.labels_) if l != -1]
    
    detected_rows = len(unique_y_labels)
    detected_cols = len(unique_x_labels)
    
    print(f"   📊 检测到: {detected_rows}行 × {detected_cols}列 (目标: {target_rows}行 × {target_cols}列)")
    
    # 计算每行的平均Y坐标和每列的平均X坐标
    row_y_means = []
    for label in unique_y_labels:
        row_mask_indices = np.where(y_clustering.labels_ == label)[0]
        if len(row_mask_indices) > 0:
            row_y = np.mean([y_coords[i] for i in row_mask_indices])
            row_y_means.append((label, row_y))
    
    row_y_means.sort(key=lambda x: x[1])
    
    col_x_means = []
    for label in unique_x_labels:
        col_mask_indices = np.where(x_clustering.labels_ == label)[0]
        if len(col_mask_indices) > 0:
            col_x = np.mean([x_coords[i] for i in col_mask_indices])
            col_x_means.append((label, col_x))
    
    col_x_means.sort(key=lambda x: x[1])
    
    # 如果检测到的行列数不足，直接按位置排序选择
    if detected_rows < target_rows or detected_cols < target_cols:
        print(f"   ⚠️ 检测到的网格不完整，按位置排序选择...")
        # 按位置排序，保留最多target_rows * target_cols个
        main_masks.sort(key=lambda m: (m['bbox'][1], m['bbox'][0]))
        filtered_masks = main_masks[:target_rows * target_cols]
        print(f"   ✅ 筛选后: {len(filtered_masks)} 个mask")
        return filtered_masks
    
    # 如果检测到的行列数足够，筛选规则的网格
    # 选择中间部分组成目标网格
    if len(row_y_means) >= target_rows:
        # 选择中间的target_rows行
        excess = len(row_y_means) - target_rows
        start_idx = excess // 2
        selected_row_labels = [row_y_means[i][0] for i in range(start_idx, start_idx + target_rows)]
    elif len(row_y_means) > 0:
        # 如果行数不足，使用所有行
        selected_row_labels = [r[0] for r in row_y_means]
    else:
        selected_row_labels = []
    
    if len(col_x_means) >= target_cols:
        # 选择中间的target_cols列
        excess = len(col_x_means) - target_cols
        start_idx = excess // 2
        selected_col_labels = [col_x_means[i][0] for i in range(start_idx, start_idx + target_cols)]
    elif len(col_x_means) > 0:
        # 如果列数不足，使用所有列
        selected_col_labels = [c[0] for c in col_x_means]
    else:
        selected_col_labels = []
    
    # 筛选出在选中行列中的mask
    filtered_masks = []
    for i, m in enumerate(main_masks):
        y_label = y_clustering.labels_[i]
        x_label = x_clustering.labels_[i]
        if y_label in selected_row_labels and x_label in selected_col_labels:
            filtered_masks.append(m)
    
    # 如果筛选后数量不足，按位置排序补齐
    if len(filtered_masks) < target_rows * target_cols:
        print(f"   ⚠️ 筛选后数量不足({len(filtered_masks)})，按位置补齐...")
        # 按位置排序所有候选mask
        main_masks.sort(key=lambda m: (m['bbox'][1], m['bbox'][0]))
        # 如果候选mask足够，补齐到目标数量
        if len(main_masks) >= target_rows * target_cols:
            filtered_masks = main_masks[:target_rows * target_cols]
        else:
            filtered_masks = main_masks
    elif len(filtered_masks) > target_rows * target_cols:
        # 如果筛选后数量过多，按面积和位置进一步筛选
        print(f"   ⚠️ 筛选后数量过多({len(filtered_masks)})，进一步筛选...")
        # 计算平均面积
        filtered_areas = [m['area'] for m in filtered_masks]
        median_filtered_area = np.median(filtered_areas)
        
        # 按面积接近中位数和位置排序
        filtered_masks_with_score = []
        for m in filtered_masks:
            area = m['area']
            # 面积得分：越接近中位数得分越高
            area_score = 1.0 / (1.0 + abs(area - median_filtered_area) / median_filtered_area)
            filtered_masks_with_score.append((m, area_score))
        
        # 先按位置排序，然后按面积得分排序
        filtered_masks_with_score.sort(key=lambda x: (x[0]['bbox'][1], x[0]['bbox'][0], -x[1]))
        filtered_masks = [m for m, _ in filtered_masks_with_score[:target_rows * target_cols]]
    
    print(f"   ✅ 最终筛选后: {len(filtered_masks)} 个mask")
    return filtered_masks


def process_image_with_sam_everything(lines_image_path, original_image_path, output_dir, image_name,
                                       points_per_side=32, pred_iou_thresh=0.86, 
                                       stability_score_thresh=0.92, min_mask_area=100, bbox_alpha=0.6):
    """
    使用SAM everything模式处理线段图像，并在原图上绘制mask的矩形框
    
    Args:
        lines_image_path: 线段图像路径（all_lines.png）
        original_image_path: 原图路径（用于绘制矩形框）
        output_dir: 输出目录
        image_name: 图像名称
        points_per_side: SAM采样点密度
        pred_iou_thresh: 预测IOU阈值
        stability_score_thresh: 稳定性得分阈值
        min_mask_area: 最小mask面积
        bbox_alpha: 矩形框透明度（0.0-1.0）
    
    Returns:
        masks: SAM生成的masks列表
    """
    # 延迟加载SAM模型
    try:
        sam, device = load_sam_model()
    except Exception as e:
        print(f"   ❌ SAM模型加载失败: {e}")
        return []
    
    # 读取线段图像（all_lines.png）
    lines_img = cv2.imread(lines_image_path)
    if lines_img is None:
        print(f"   ❌ 无法读取线段图像: {lines_image_path}")
        return []
    
    lines_img_rgb = cv2.cvtColor(lines_img, cv2.COLOR_BGR2RGB)
    
    # 创建SAM自动mask生成器（使用可配置参数）
    from segment_anything.utils.transforms import ResizeLongestSide
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,  # 采样点密度
        pred_iou_thresh=pred_iou_thresh,  # 预测IOU阈值
        stability_score_thresh=stability_score_thresh,  # 稳定性得分阈值
        crop_n_layers=1,  # 减少裁剪层，减少细分
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_mask_area,  # 最小区域面积
    )
    
    print(f"      SAM参数: points={points_per_side}, iou={pred_iou_thresh}, stability={stability_score_thresh}, min_area={min_mask_area}")
    
    # 生成masks
    print(f"   🔄 正在生成masks（可能需要几分钟）...")
    masks = mask_generator.generate(lines_img_rgb)
    print(f"   ✅ 生成 {len(masks)} 个masks")
    
    # 读取原图
    original_img = cv2.imread(original_image_path)
    if original_img is None:
        print(f"   ❌ 无法读取原图: {original_image_path}")
        return masks
    
    # 保存mask可视化（整张图像的mask叠加显示）
    mask_vis_img = lines_img_rgb.copy()
    if len(masks) > 0:
        # 按面积排序（从大到小）
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # 为每个mask分配一个颜色（使用透明度叠加）
        overlay = mask_vis_img.copy().astype(np.float32)
        
        # 设置随机种子以获得可重复的颜色
        np.random.seed(42)
        
        for i, mask_info in enumerate(sorted_masks):
            mask = mask_info['segmentation']  # 2D boolean array (H x W)
            # 为每个mask生成一个随机颜色
            color = np.random.randint(0, 255, 3)
            
            # 只对mask区域着色（半透明）
            # mask是2D的，需要扩展到3D来匹配overlay的shape
            mask_3d = mask[:, :, np.newaxis]  # (H x W x 1)
            
            # 对每个通道分别应用mask
            overlay[mask, 0] = overlay[mask, 0] * 0.5 + color[0] * 0.5
            overlay[mask, 1] = overlay[mask, 1] * 0.5 + color[1] * 0.5
            overlay[mask, 2] = overlay[mask, 2] * 0.5 + color[2] * 0.5
        
        mask_vis_img = overlay.astype(np.uint8)
    
    mask_vis_path = os.path.join(output_dir, f"{image_name}_sam_masks_visualization.png")
    cv2.imwrite(mask_vis_path, cv2.cvtColor(mask_vis_img, cv2.COLOR_RGB2BGR))
    print(f"   ✅ SAM masks可视化已保存: {mask_vis_path}")
    
    # 暂时不过滤mask，直接使用所有检测到的mask
    print(f"   📝 暂不过滤mask，保留所有检测到的 {len(masks)} 个mask")
    
    # 在原图上绘制所有mask的矩形框（红色，1px，半透明）
    # 方法：直接使用SAM返回的bbox（轴对齐边界框）
    overlay = original_img.copy()
    
    for i, mask_info in enumerate(masks):
        # SAM自动计算的bbox：[x, y, width, height]
        bbox = mask_info['bbox']  
        x, y, w, h = bbox
        
        # 绘制矩形框（红色，1px）
        cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
    
    # 叠加半透明效果
    vis_img = cv2.addWeighted(overlay, bbox_alpha, original_img, 1 - bbox_alpha, 0)
    
    # 保存结果
    output_path = os.path.join(output_dir, f"{image_name}_sam_masks_bboxes.png")
    cv2.imwrite(output_path, vis_img)
    print(f"   ✅ 所有mask矩形框已保存: {output_path} (共{len(masks)}个，1px半透明α={bbox_alpha})")
    
    
    # 保存masks信息（JSON）
    masks_json_path = os.path.join(output_dir, f"{image_name}_sam_masks.json")
    masks_data = {
        "image": image_name,
        "mask_count": len(masks),
        "sam_config": {
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "crop_n_layers": 1,
            "min_mask_region_area": min_mask_area
        },
        "masks": []
    }
    
    for i, mask_info in enumerate(masks):
        bbox = mask_info['bbox']
        masks_data["masks"].append({
            "id": i,
            "bbox": {
                "x": float(bbox[0]),
                "y": float(bbox[1]),
                "width": float(bbox[2]),
                "height": float(bbox[3])
            },
            "area": int(mask_info['area']),
            "predicted_iou": float(mask_info.get('predicted_iou', 0)),
            "stability_score": float(mask_info.get('stability_score', 0))
        })
    
    with open(masks_json_path, 'w', encoding='utf-8') as f:
        json.dump(masks_data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ SAM masks信息已保存: {masks_json_path}")
    
    return masks


def save_json_results(json_path, horizontal_lines, vertical_lines, image_name, image_size):
    """保存检测结果到JSON文件"""
    result = {
        "image": image_name,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "method": "DeepLSD",
        "horizontal_lines": {
            "count": len(horizontal_lines),
            "lines": [
                {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                for x1, y1, x2, y2 in horizontal_lines
            ]
        },
        "vertical_lines": {
            "count": len(vertical_lines),
            "lines": [
                {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                for x1, y1, x2, y2 in vertical_lines
            ]
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"💾 JSON结果已保存: {json_path}")


def process_single_image(image_path, model, device, output_dir="deeplsd_results", 
                         min_length_ratio=0.05, endpoint_distance_threshold=10,
                         sam_points_per_side=32, sam_pred_iou_thresh=0.86, 
                         sam_stability_thresh=0.92, sam_min_area=100, area_tolerance=0.5, bbox_alpha=0.6):
    """
    处理单张图像（支持宽图像自适应）
    
    Args:
        image_path: 图像路径
        model: DeepLSD模型
        device: 设备
        output_dir: 输出目录
        min_length_ratio: 二次检测的最小线段长度比例（相对图像对角线）
        endpoint_distance_threshold: 竖线端点到横线的距离阈值（像素）
        sam_points_per_side: SAM采样点密度
        sam_pred_iou_thresh: SAM预测IOU阈值
        sam_stability_thresh: SAM稳定性阈值
        sam_min_area: SAM最小mask面积
        area_tolerance: 面积容差范围（相对中位数）
        bbox_alpha: 矩形框透明度（0.0-1.0）
    """
    print(f"\n{'='*60}")
    print(f"处理图像: {image_path}")
    print(f"{'='*60}")
    
    # 预先读取图像以检测宽高比
    test_img = cv2.imread(image_path)
    if test_img is None and PIL_AVAILABLE:
        try:
            pil_img = PILImage.open(image_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            test_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            pass
    
    # 自适应参数调整（针对宽图像）
    if test_img is not None:
        img_h, img_w = test_img.shape[:2]
        aspect_ratio = img_w / img_h
        
        print(f"📐 图像尺寸: {img_w} × {img_h} (宽高比: {aspect_ratio:.2f})")
        
        # 宽图像自适应调整
        if aspect_ratio > 2.5:
            # 超宽图像（如3列作文格，宽高比约3.0）
            print(f"⚙️  检测到超宽图像（宽高比 > 2.5），自动调整参数：")
            
            # 增加SAM采样点密度
            original_points = sam_points_per_side
            sam_points_per_side = min(int(sam_points_per_side * 2.0), 64)
            print(f"   SAM采样点: {original_points} → {sam_points_per_side} (x2倍)")
            
            # 调整端点距离阈值
            original_threshold = endpoint_distance_threshold
            endpoint_distance_threshold = int(endpoint_distance_threshold * 1.5)
            print(f"   端点距离阈值: {original_threshold} → {endpoint_distance_threshold}px (x1.5倍)")
            
            # 调整最小面积
            original_min_area = sam_min_area
            sam_min_area = int(sam_min_area * 2.0)
            print(f"   SAM最小面积: {original_min_area} → {sam_min_area}px² (x2倍)")
            
            # 放宽面积容差
            original_tolerance = area_tolerance
            area_tolerance = min(area_tolerance * 1.5, 0.8)
            print(f"   面积容差: {original_tolerance:.1f} → {area_tolerance:.1f} (x1.5倍)")
            
        elif aspect_ratio > 1.8:
            # 宽图像（如2列作文格）
            print(f"⚙️  检测到宽图像（宽高比 > 1.8），调整参数：")
            
            original_points = sam_points_per_side
            sam_points_per_side = min(int(sam_points_per_side * 1.5), 64)
            print(f"   SAM采样点: {original_points} → {sam_points_per_side} (x1.5倍)")
            
            original_threshold = endpoint_distance_threshold
            endpoint_distance_threshold = int(endpoint_distance_threshold * 1.3)
            print(f"   端点距离阈值: {original_threshold} → {endpoint_distance_threshold}px (x1.3倍)")
            
            original_min_area = sam_min_area
            sam_min_area = int(sam_min_area * 1.5)
            print(f"   SAM最小面积: {original_min_area} → {sam_min_area}px² (x1.5倍)")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像名称
    image_name = Path(image_path).stem
    
    try:
        # 检测线段
        lines, img_shape = detect_lines_deeplsd(image_path, model, device)
        
        # 转换格式
        lines_standard = convert_lines_format(lines)
        
        # 分类横线和竖线
        horizontal_lines, vertical_lines, other_lines = filter_horizontal_vertical(lines_standard)
        
        print(f"📊 分类结果:")
        print(f"   横线: {len(horizontal_lines)} 条")
        print(f"   竖线: {len(vertical_lines)} 条")
        if other_lines:
            print(f"   其他: {len(other_lines)} 条")
        
        # 合并所有线段（用于提取矩形框）
        all_lines = horizontal_lines + vertical_lines + other_lines
        print(f"   总线段: {len(all_lines)} 条")
        
        # 根据文件名确定目标网格尺寸
        filename = Path(image_path).stem
        if filename.endswith('_B_03'):
            target_rows = 17
            target_cols = 16
            print(f"   📋 检测到_B_03类型，目标: {target_rows}行 × {target_cols}列")
        else:
            target_rows = 20
            target_cols = 16
            print(f"   📋 目标网格: {target_rows}行 × {target_cols}列")
        
        # === 新思路：二次DeepLSD检测 ===
        print(f"\n📝 新思路：使用DeepLSD二次检测去除干扰线...")
        
        # 第一步：绘制所有原始线段到图像（不过滤）
        print(f"   步骤1: 绘制第一次检测的所有线段...")
        img_h, img_w = img_shape[:2]
        lines_image_raw = np.ones((img_h, img_w), dtype=np.uint8) * 255  # 白色背景
        
        # 绘制所有横线（黑色，1px）
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(lines_image_raw, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
        
        # 绘制所有竖线（黑色，1px）
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(lines_image_raw, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
        
        # 保存原始线段图像
        lines_raw_path = os.path.join(output_dir, f"{image_name}_all_lines_raw.png")
        cv2.imwrite(lines_raw_path, lines_image_raw)
        print(f"   ✅ 第一次检测的所有线段已保存: {lines_raw_path}")
        print(f"      (横线: {len(horizontal_lines)}, 竖线: {len(vertical_lines)})")
        
        # 第二步：用DeepLSD再次检测纯线段图像（横线和竖线分别处理）
        print(f"\n   步骤2: 使用DeepLSD二次检测纯线段图（分类处理模式）...")
        
        try:
            # 策略：横线用长度过滤，竖线不过滤（保留短竖线）
            
            # 2.1 绘制仅包含横线的图像
            print(f"      2.1 绘制纯横线图...")
            h_lines_image = np.ones((img_h, img_w), dtype=np.uint8) * 255
            for x1, y1, x2, y2 in horizontal_lines:
                cv2.line(h_lines_image, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
            h_lines_path = os.path.join(output_dir, f"{image_name}_horizontal_lines_raw.png")
            cv2.imwrite(h_lines_path, h_lines_image)
            
            # 2.2 绘制仅包含竖线的图像
            print(f"      2.2 绘制纯竖线图...")
            v_lines_image = np.ones((img_h, img_w), dtype=np.uint8) * 255
            for x1, y1, x2, y2 in vertical_lines:
                cv2.line(v_lines_image, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
            v_lines_path = os.path.join(output_dir, f"{image_name}_vertical_lines_raw.png")
            cv2.imwrite(v_lines_path, v_lines_image)
            
            # 2.3 二次检测横线（使用长度过滤）
            print(f"      2.3 二次检测横线（使用长度过滤）...")
            img_diag = np.sqrt(img_h**2 + img_w**2)
            min_h_length = img_diag * min_length_ratio  # 横线最小长度
            print(f"          横线参数: min_length={min_h_length:.1f}px")
            
            h_lines_2nd, _ = detect_lines_deeplsd(
                h_lines_path, model, device,
                min_length=min_h_length,
                score_thresh=0.0,
                is_second_pass=True
            )
            h_lines_2nd_standard = convert_lines_format(h_lines_2nd)
            horizontal_lines_2nd, _, _ = filter_horizontal_vertical(h_lines_2nd_standard)
            
            # 2.4 二次检测竖线（不使用长度过滤，保留短竖线）
            print(f"      2.4 二次检测竖线（不过滤长度，保留短竖线）...")
            v_lines_2nd, _ = detect_lines_deeplsd(
                v_lines_path, model, device,
                min_length=0,  # 不过滤长度
                score_thresh=0.0,
                is_second_pass=True
            )
            v_lines_2nd_standard = convert_lines_format(v_lines_2nd)
            _, vertical_lines_2nd, _ = filter_horizontal_vertical(v_lines_2nd_standard)
            
            # 2.5 根据端点过滤竖线（只保留上下两端都接近横线的）
            print(f"      2.5 根据端点过滤竖线（距离阈值={endpoint_distance_threshold}px）...")
            vertical_lines_2nd = filter_vertical_lines_by_endpoints(
                vertical_lines_2nd, 
                horizontal_lines_2nd,  # 使用二次检测后的横线
                distance_threshold=endpoint_distance_threshold  # 使用配置的阈值
            )
            
            print(f"   📊 第二次检测对比:")
            print(f"      横线: {len(horizontal_lines)} -> {len(horizontal_lines_2nd)} (长度过滤)")
            print(f"      竖线: {len(vertical_lines)} -> {len(vertical_lines_2nd)} (端点过滤)")
            
            # 使用第二次检测的结果
            filtered_horizontal_lines = horizontal_lines_2nd
            filtered_vertical_lines = vertical_lines_2nd
            
        except Exception as e:
            print(f"   ⚠️ 第二次检测失败，使用传统过滤方法: {e}")
            import traceback
            traceback.print_exc()
            # 如果第二次检测失败，回退到过滤方法
            filtered_horizontal_lines = filter_grid_lines(horizontal_lines, 'horizontal', img_shape)
            filtered_vertical_lines = filter_grid_lines(vertical_lines, 'vertical', img_shape)
        
        # 第三步：生成最终的线段图像（用于SAM处理）
        print(f"\n   步骤3: 生成最终线段图像...")
        lines_image = np.ones((img_h, img_w), dtype=np.uint8) * 255  # 白色背景
        
        # 绘制过滤后的横线（黑色，1px）
        for x1, y1, x2, y2 in filtered_horizontal_lines:
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
        
        # 绘制过滤后的竖线（黑色，1px）统一线宽
        for x1, y1, x2, y2 in filtered_vertical_lines:
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
        
        # 保存最终线段图像
        lines_only_path = os.path.join(output_dir, f"{image_name}_all_lines.png")
        cv2.imwrite(lines_only_path, lines_image)
        print(f"   ✅ 最终线段图像已保存: {lines_only_path}")
        print(f"      (横线: {len(filtered_horizontal_lines)}, 竖线: {len(filtered_vertical_lines)})")
        
        # 使用SAM everything模式处理 all_lines.png
        if SAM_AVAILABLE:
            print(f"\n🔍 使用SAM everything模式处理 {lines_only_path}...")
            try:
                sam_masks = process_image_with_sam_everything(
                    lines_only_path, 
                    image_path,  # 原图路径，用于绘制矩形框
                    output_dir,
                    image_name,
                    points_per_side=sam_points_per_side,
                    pred_iou_thresh=sam_pred_iou_thresh,
                    stability_score_thresh=sam_stability_thresh,
                    min_mask_area=sam_min_area,
                    bbox_alpha=bbox_alpha
                )
                print(f"   ✅ SAM处理完成，生成 {len(sam_masks)} 个masks")
                
                # 按面积过滤，去除异常大小的mask
                if len(sam_masks) > 0:
                    print(f"\n   步骤4: 按面积过滤mask...")
                    filtered_masks = filter_masks_by_area(sam_masks, area_tolerance=area_tolerance)
                    
                    if len(filtered_masks) > 0:
                        # 在原图上绘制终版检测框（蓝色，1px，半透明）
                        print(f"   步骤5: 绘制终版检测框...")
                        original_img = cv2.imread(image_path)
                        if original_img is not None:
                            # 创建overlay层用于半透明绘制
                            overlay = original_img.copy()
                            
                            for mask_info in filtered_masks:
                                bbox = mask_info['bbox']
                                x, y, w, h = bbox
                                # 绘制蓝色矩形框（1px）
                                cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 1)
                            
                            # 叠加半透明效果
                            final_vis_img = cv2.addWeighted(overlay, bbox_alpha, original_img, 1 - bbox_alpha, 0)
                            
                            # 保存终版检测框图像
                            final_output_path = os.path.join(output_dir, f"{image_name}_final_bboxes.png")
                            cv2.imwrite(final_output_path, final_vis_img)
                            print(f"   ✅ 终版检测框已保存: {final_output_path} (共{len(filtered_masks)}个，蓝色1px半透明α={bbox_alpha})")
                            
                            # 保存终版检测框JSON
                            final_json_path = os.path.join(output_dir, f"{image_name}_final_bboxes.json")
                            final_json_data = {
                                "image": image_name,
                                "total_masks": len(sam_masks),
                                "filtered_masks": len(filtered_masks),
                                "filter_method": "area_median",
                                "bboxes": []
                            }
                            
                            for i, mask_info in enumerate(filtered_masks):
                                bbox = mask_info['bbox']
                                final_json_data["bboxes"].append({
                                    "id": i,
                                    "x": float(bbox[0]),
                                    "y": float(bbox[1]),
                                    "width": float(bbox[2]),
                                    "height": float(bbox[3]),
                                    "area": int(mask_info['area'])
                                })
                            
                            with open(final_json_path, 'w', encoding='utf-8') as f:
                                json.dump(final_json_data, f, ensure_ascii=False, indent=2)
                            print(f"   ✅ 终版检测框JSON已保存: {final_json_path}")
                            
                            # 步骤6: 使用终版检测框进行字符检测
                            print(f"\n   步骤6: 字符检测（使用SAM box prompt）...")
                            try:
                                char_results = process_char_detection_with_sam(
                                    image_path,
                                    final_json_data["bboxes"],
                                    output_dir,
                                    image_name,
                                    bbox_alpha=bbox_alpha
                                )
                                
                                if len(char_results) > 0:
                                    print(f"   ✅ 字符检测完成，共检测到 {len(char_results)} 个字符")
                                    
                                    # 步骤7: 合并两种检测框到一张图像
                                    print(f"\n   步骤7: 生成合并检测框图像...")
                                    try:
                                        # 创建overlay层
                                        combined_overlay = original_img.copy()
                                        
                                        # 先绘制所有蓝色作文格框（外框）
                                        for mask_info in filtered_masks:
                                            bbox = mask_info['bbox']
                                            x, y, w, h = bbox
                                            cv2.rectangle(combined_overlay, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 1)
                                        
                                        # 再绘制红色字符框（内框，只有有字符的）
                                        for char_result in char_results:
                                            char_bbox = char_result['char_bbox']
                                            cx = char_bbox['x']
                                            cy = char_bbox['y']
                                            cw = char_bbox['width']
                                            ch = char_bbox['height']
                                            cv2.rectangle(combined_overlay, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 1)
                                        
                                        # 叠加半透明效果
                                        combined_vis_img = cv2.addWeighted(combined_overlay, bbox_alpha, original_img, 1 - bbox_alpha, 0)
                                        
                                        # 在图像顶部添加统计信息
                                        text = f"Blue Boxes: {len(filtered_masks)}  |  Red Boxes: {len(char_results)}"
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        font_scale = 1.0
                                        font_thickness = 2
                                        
                                        # 获取文本尺寸
                                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                                        
                                        # 绘制半透明背景条
                                        bg_height = text_height + baseline + 20
                                        bg_overlay = combined_vis_img.copy()
                                        cv2.rectangle(bg_overlay, (0, 0), (combined_vis_img.shape[1], bg_height), (0, 0, 0), -1)
                                        combined_vis_img = cv2.addWeighted(bg_overlay, 0.6, combined_vis_img, 0.4, 0)
                                        
                                        # 绘制白色文字
                                        text_x = 10
                                        text_y = text_height + 10
                                        cv2.putText(combined_vis_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                                        
                                        # 保存合并图像
                                        combined_path = os.path.join(output_dir, f"{image_name}_combined_detection.png")
                                        cv2.imwrite(combined_path, combined_vis_img)
                                        print(f"   ✅ 合并检测框已保存: {combined_path}")
                                        print(f"      📦 蓝色=作文格框({len(filtered_masks)}个) + 红色=字符框({len(char_results)}个)")
                                    except Exception as e:
                                        print(f"   ⚠️ 合并图像生成失败: {e}")
                                else:
                                    print(f"   ⚠️ 未检测到字符")
                            except Exception as e:
                                print(f"   ⚠️ 字符检测失败: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"   ⚠️ 无法读取原图: {image_path}")
                    else:
                        print(f"   ⚠️ 面积过滤后没有剩余mask")
                else:
                    print(f"   ⚠️ SAM未检测到任何mask")
                    
            except Exception as e:
                print(f"   ⚠️ SAM处理失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ⚠️ SAM未安装，跳过SAM处理")
        
        # 可视化（原有功能）
        vis_path = os.path.join(output_dir, f"{image_name}_deeplsd_visual.png")
        visualize_lines(image_path, horizontal_lines, vertical_lines, vis_path)
        
        # 保存文本结果（原有功能）
        txt_path = os.path.join(output_dir, f"{image_name}_deeplsd_lines.txt")
        save_results(txt_path, horizontal_lines, vertical_lines, image_name)
        
        # 保存JSON结果（原有功能）
        json_path = os.path.join(output_dir, f"{image_name}_deeplsd_lines.json")
        save_json_results(json_path, horizontal_lines, vertical_lines, image_name, 
                         (img_shape[1], img_shape[0]))
        
        print(f"✅ 处理完成: {image_name}\n")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # ========== 配置区域 ==========
    INPUT_FOLDER = "png"  # PNG文件夹
    OUTPUT_DIR = "deeplsd_results"  # 输出目录
    MODEL_PATH = None  # 模型路径，None则自动查找
    DEVICE = 'cuda'  # 'cuda' 或 'cpu'
    
    # DeepLSD模型参数
    GRAD_THRESH = 3  # 梯度阈值（3=正常，5-10=更严格，去除更多干扰线）
    MERGE_LINES = True  # 是否合并相近线段
    
    # 二次检测参数
    MIN_LENGTH_RATIO = 0.05  # 横线最小长度比例（相对图像对角线，0.05=5%）
    
    # 竖线端点过滤参数
    ENDPOINT_DISTANCE_THRESHOLD = 10  # 竖线端点到横线的距离阈值（像素，推荐5-15）
    
    # SAM参数（调整检测密度，防止过度分割）
    SAM_POINTS_PER_SIDE = 40  # 采样点密度（16=稀疏，32=适中，40=较密，64=密集）
    SAM_PRED_IOU_THRESH = 0.82  # 预测IOU阈值（越高越严格，推荐0.8-0.95）
    SAM_STABILITY_THRESH = 0.88  # 稳定性阈值（越高越严格，推荐0.85-0.95）
    SAM_MIN_AREA = 70  # 最小mask面积（像素，过滤碎片）
    
    # 面积过滤参数
    AREA_TOLERANCE = 0.5  # 面积容差范围（相对中位数，0.5=中位数的50%-150%）
    
    # 可视化参数
    BBOX_ALPHA = 0.6  # 矩形框透明度（0.0=完全透明，1.0=完全不透明，推荐0.5-0.8）
    # ========== 配置区域结束 ==========
    
    print("="*60)
    print("DeepLSD 横线竖线检测工具 (二次检测去干扰模式)")
    print("="*60)
    print(f"🔧 DeepLSD配置: grad_thresh={GRAD_THRESH}, min_length_ratio={MIN_LENGTH_RATIO*100}%")
    print(f"📋 策略: 横线用长度过滤，竖线用端点过滤（阈值={ENDPOINT_DISTANCE_THRESHOLD}px）")
    print(f"🎯 SAM配置: points={SAM_POINTS_PER_SIDE}, iou={SAM_PRED_IOU_THRESH}, stability={SAM_STABILITY_THRESH}, min_area={SAM_MIN_AREA}")
    print(f"📐 面积过滤: tolerance=±{AREA_TOLERANCE*100:.0f}% (保留中位数的{(1-AREA_TOLERANCE)*100:.0f}%-{(1+AREA_TOLERANCE)*100:.0f}%)")
    
    # 检查DeepLSD是否可用
    if not DEEPLSD_AVAILABLE:
        print("\n❌ DeepLSD未安装或导入失败")
        print("\n💡 请确保:")
        print("  1. DeepLSD 文件夹存在")
        print("  2. 已安装必要的依赖（PyTorch等）")
        print("  3. 已下载模型权重文件")
        return
    
    # 加载模型（使用配置参数）
    try:
        model, device = load_deeplsd_model(
            model_path=MODEL_PATH, 
            device=DEVICE,
            grad_thresh=GRAD_THRESH,
            merge_lines=MERGE_LINES
        )
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        return
    
    # 检查输入文件夹
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 输入文件夹不存在: {INPUT_FOLDER}")
        return
    
    # 获取所有PNG和TIF文件
    folder = Path(INPUT_FOLDER)
    image_files = []
    
    # 支持多种格式
    for ext in ['*.png', '*.PNG', '*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        image_files.extend(list(folder.glob(ext)))
    
    # 排序并去重
    image_files = sorted(list(set(image_files)))
    
    if len(image_files) == 0:
        print(f"⚠️ 未找到图像文件: {INPUT_FOLDER}/*.png 或 *.tif")
        return
    
    print(f"\n📁 找到 {len(image_files)} 个图像文件（png/tif格式）")
    print(f"📂 输出目录: {OUTPUT_DIR}\n")
    
    # 处理每张图像
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] 处理: {img_path.name}")
        process_single_image(
            str(img_path), 
            model, 
            device, 
            output_dir=OUTPUT_DIR,
            min_length_ratio=MIN_LENGTH_RATIO,
            endpoint_distance_threshold=ENDPOINT_DISTANCE_THRESHOLD,
            sam_points_per_side=SAM_POINTS_PER_SIDE,
            sam_pred_iou_thresh=SAM_PRED_IOU_THRESH,
            sam_stability_thresh=SAM_STABILITY_THRESH,
            sam_min_area=SAM_MIN_AREA,
            area_tolerance=AREA_TOLERANCE,
            bbox_alpha=BBOX_ALPHA
        )
    
    print(f"\n{'='*60}")
    print(f"✅ 批量处理完成！结果保存在: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

