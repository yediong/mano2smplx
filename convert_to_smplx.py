#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 Dyn-HaMR 的 MANO 输出转换为 SMPL-X 格式

作者：AI Assistant
日期：2025-11-23

关键说明:
1. SMPL-X 确实有 right_hand_pose 和 left_hand_pose 参数 (每个45维，15关节×3)
2. MANO 和 SMPL-X 的全局方向/平移是兼容的:
   - root_orient (MANO) → global_orient (SMPL-X)
   - trans (MANO) → transl (SMPL-X)
   两者都使用相同的坐标系统和axis-angle表示
3. 支持多种场景: 单手、双手、多轨迹
"""

import os
import sys
import argparse
import numpy as np

def analyze_hands_in_data(data):
    """
    分析数据中包含的手部信息
    
    返回: 
        list of dict: 每个轨迹的手部信息 [{'batch_idx': 0, 'is_right': True/False, 'num_frames': T}, ...]
    """
    if 'is_right' not in data:
        # 如果没有 is_right 字段，尝试从其他信息推断
        print("⚠️  警告: 数据中没有 'is_right' 字段，假设为右手")
        return [{'batch_idx': 0, 'is_right': True, 'num_frames': data['pose_body'].shape[1]}]
    
    is_right = data['is_right']
    
    if len(is_right.shape) == 1:
        # 形状 (T,) - 单个轨迹
        hands_info = [{
            'batch_idx': 0,
            'is_right': bool(is_right[0]),
            'num_frames': len(is_right)
        }]
    elif len(is_right.shape) == 2:
        # 形状 (B, T) - 多个轨迹
        hands_info = []
        for b in range(is_right.shape[0]):
            # 检查该轨迹在时间维度上is_right值是否一致
            unique_values = np.unique(is_right[b])
            if len(unique_values) > 1:
                print(f"⚠️  警告: 轨迹 {b} 的 is_right 值不一致: {unique_values}，使用第一帧的值")
            
            hands_info.append({
                'batch_idx': b,
                'is_right': bool(is_right[b, 0]),
                'num_frames': is_right.shape[1]
            })
    else:
        raise ValueError(f"不支持的 is_right 形状: {is_right.shape}")
    
    return hands_info


def convert_mano_to_smplx(input_npz_path, output_path=None, verbose=True):
    """
    将 MANO 格式的手部数据转换为 SMPL-X 格式
    
    参数:
        input_npz_path (str): 输入的 .npz 文件路径 (Dyn-HaMR 输出)
        output_path (str): 输出的 .npz 文件路径 (可选，默认在同目录下生成)
        verbose (bool): 是否打印详细信息
    
    返回:
        str or list: 生成的 SMPL-X 文件路径
    """
    
    # 加载 MANO 数据
    if verbose:
        print(f"📖 读取 MANO 数据: {input_npz_path}")
    
    mano_data = np.load(input_npz_path)
    
    if verbose:
        print("\n📋 输入文件包含的键值:")
        for key in mano_data.keys():
            print(f"  {key}: {mano_data[key].shape}")
    
    # 提取参数
    pose_body = mano_data['pose_body']  # 可能是 (B, T, 45), (T, 45), 或 (B, T, 15, 3)
    root_orient = mano_data['root_orient']  # (B, T, 3) 或 (T, 3)
    trans = mano_data['trans']  # (B, T, 3) 或 (T, 3)
    
    # 检查并处理 pose_body 的形状
    if len(pose_body.shape) == 4:
        # (B, T, 15, 3) -> (B, T, 45) - 展平手部关节维度
        print(f"  ℹ️  检测到 pose_body 形状为 {pose_body.shape}，展平为 (B, T, 45)")
        B, T = pose_body.shape[0], pose_body.shape[1]
        pose_body = pose_body.reshape(B, T, -1)  # 将 15x3 展平为 45
    elif len(pose_body.shape) == 3:
        # (T, 15, 3) -> (T, 45) -> (1, T, 45)
        print(f"  ℹ️  检测到 pose_body 形状为 {pose_body.shape}，展平为 (1, T, 45)")
        T = pose_body.shape[0]
        pose_body = pose_body.reshape(T, -1)  # 将 15x3 展平为 45
        pose_body = pose_body[np.newaxis, ...]  # 添加批次维度
        root_orient = root_orient[np.newaxis, ...]
        trans = trans[np.newaxis, ...]
        B = 1
    elif len(pose_body.shape) == 2:
        # (T, 45) -> (1, T, 45) - 已经是展平的
        pose_body = pose_body[np.newaxis, ...]
        root_orient = root_orient[np.newaxis, ...]
        trans = trans[np.newaxis, ...]
        B, T = 1, pose_body.shape[1]
    else:
        # (B, T, 45) - 已经是正确格式
        B, T = pose_body.shape[0], pose_body.shape[1]
    
    # 验证最终形状
    assert len(pose_body.shape) == 3, f"pose_body 形状错误: {pose_body.shape}"
    assert pose_body.shape[2] == 45, f"pose_body 最后一维应该是45，实际是 {pose_body.shape[2]}"
    
    # 处理 betas (形状参数)
    if 'betas' in mano_data:
        betas = mano_data['betas']
        if len(betas.shape) == 1:
            # (10,) -> (1, 10)
            betas = betas[np.newaxis, ...]
    else:
        print("⚠️  警告: 输入文件中没有 'betas'，使用默认零向量")
        betas = np.zeros((B, 10), dtype=np.float32)
    
    if verbose:
        print(f"\n📊 数据维度: B={B} (批次/轨迹数), T={T} (时间步/帧数)")
    
    # 分析手部信息
    hands_info = analyze_hands_in_data(mano_data)
    
    if verbose:
        print(f"\n🖐️  检测到的手部:")
        for i, info in enumerate(hands_info):
            hand_type = "右手" if info['is_right'] else "左手"
            print(f"  轨迹 {i} (batch {info['batch_idx']}): {hand_type}, {info['num_frames']} 帧")
    
    # 构建 SMPL-X 数据
    smplx_data_list = []
    
    for hand_info in hands_info:
        b = hand_info['batch_idx']
        is_right = hand_info['is_right']
        
        # SMPL-X 的 body_pose: (T, 63) - 21个身体关节
        # 因为 MANO 只有手部信息，所以用零填充
        smplx_body_pose = np.zeros((T, 63), dtype=np.float32)
        
        # 手部姿态: (T, 45)
        hand_pose = pose_body[b]
        
        # 全局方向: (T, 3)
        # MANO 的 root_orient 对应 SMPL-X 的 global_orient
        # 两者都使用 axis-angle 表示，坐标系统相同
        global_orient = root_orient[b]
        
        # 全局平移: (T, 3)
        # MANO 的 trans 对应 SMPL-X 的 transl
        # 两者都表示模型根节点在世界坐标系中的位置
        transl = trans[b]
        
        # 形状参数: (10,)
        betas_mean = betas[b]
        
        # 构建 SMPL-X 字典
        smplx_dict = {
            # 核心姿态参数
            'body_pose': smplx_body_pose,      # (T, 63) - 身体姿态 (全零)
            'global_orient': global_orient,     # (T, 3) - 全局方向 (从 root_orient)
            'transl': transl,                   # (T, 3) - 全局平移 (从 trans)
            'betas': betas_mean,                # (10,) - 形状参数
        }
        
        # 根据左右手添加手部姿态
        # SMPL-X 使用 right_hand_pose 和 left_hand_pose 参数
        if is_right:
            smplx_dict['right_hand_pose'] = hand_pose  # (T, 45)
            smplx_dict['left_hand_pose'] = np.zeros((T, 45), dtype=np.float32)
        else:
            smplx_dict['left_hand_pose'] = hand_pose   # (T, 45)
            smplx_dict['right_hand_pose'] = np.zeros((T, 45), dtype=np.float32)
        
        # 可选: 脸部表情参数 (SMPL-X 支持，但这里没有数据)
        # smplx_dict['expression'] = np.zeros((T, 10), dtype=np.float32)
        # smplx_dict['jaw_pose'] = np.zeros((T, 3), dtype=np.float32)
        
        # 添加相机参数（如果存在）
        if 'cam_R' in mano_data:
            cam_R = mano_data['cam_R']
            if len(cam_R.shape) > 2 and B > 1:
                smplx_dict['cam_R'] = cam_R[b]
            else:
                smplx_dict['cam_R'] = cam_R
        
        if 'cam_t' in mano_data:
            cam_t = mano_data['cam_t']
            if len(cam_t.shape) > 1 and B > 1:
                smplx_dict['cam_t'] = cam_t[b]
            else:
                smplx_dict['cam_t'] = cam_t
        
        if 'intrins' in mano_data:
            smplx_dict['intrins'] = mano_data['intrins']
        
        # 添加元数据
        smplx_dict['_metadata'] = {
            'source': 'Dyn-HaMR',
            'original_batch_idx': b,
            'is_right_hand': is_right,
            'num_frames': T,
            'note': 'body_pose is zero-filled (hand-only reconstruction)'
        }
        
        smplx_data_list.append(smplx_dict)
    
    # 保存 SMPL-X 格式数据
    if output_path is None:
        base_name = os.path.splitext(input_npz_path)[0]
        output_path = f"{base_name}_smplx.npz"
    
    output_paths = []
    
    if B == 1:
        # 单个轨迹，直接保存
        if verbose:
            print(f"\n💾 保存 SMPL-X 数据到: {output_path}")
            print("\n📋 输出文件包含的键值:")
            for key, value in smplx_data_list[0].items():
                if key != '_metadata':
                    print(f"  {key}: {value.shape}")
        
        # 移除 _metadata 用于保存
        save_dict = {k: v for k, v in smplx_data_list[0].items() if k != '_metadata'}
        np.savez(output_path, **save_dict)
        output_paths.append(output_path)
        
    else:
        # 多个轨迹，每个保存为单独文件
        for i, smplx_dict in enumerate(smplx_data_list):
            hand_type = "right" if hands_info[i]['is_right'] else "left"
            batch_output_path = output_path.replace('.npz', f'_batch{i}_{hand_type}.npz')
            
            if verbose:
                print(f"\n💾 保存轨迹 {i} ({hand_type}) 的 SMPL-X 数据到: {batch_output_path}")
                if i == 0:
                    print("\n📋 输出文件包含的键值:")
                    for key, value in smplx_dict.items():
                        if key != '_metadata':
                            print(f"  {key}: {value.shape}")
            
            save_dict = {k: v for k, v in smplx_dict.items() if k != '_metadata'}
            np.savez(batch_output_path, **save_dict)
            output_paths.append(batch_output_path)
    
    if verbose:
        print("\n" + "="*80)
        print("✅ 转换完成!")
        print("\n📝 重要说明:")
        print("  1. SMPL-X 的 right_hand_pose 和 left_hand_pose 参数已正确设置")
        print("  2. global_orient 和 transl 直接从 MANO 的 root_orient 和 trans 转换")
        print("     → 坐标系统和表示方法完全兼容，无需额外转换")
        print("  3. body_pose 填充为零（仅手部重建，身体保持 T-pose）")
        if B > 1:
            print(f"  4. 检测到 {B} 个轨迹，已分别保存到不同文件")
        print("="*80)
    
    return output_paths if B > 1 else output_paths[0]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将 Dyn-HaMR 的 MANO 输出转换为 SMPL-X 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换单个文件
  python convert_to_smplx.py /path/to/output.npz
  
  # 指定输出路径
  python convert_to_smplx.py /path/to/output.npz -o /path/to/smplx_output.npz
  
  # 转换整个目录中的所有 npz 文件
  python convert_to_smplx.py /path/to/output_dir/ --batch

重要说明:
  1. SMPL-X 确实支持 right_hand_pose 和 left_hand_pose 参数
  2. 脚本会自动识别每个轨迹是左手还是右手
  3. 支持同时包含多只手的情况（会生成多个文件）
  4. MANO 和 SMPL-X 的坐标系统兼容，无需转换
        """
    )
    
    parser.add_argument('input', type=str, 
                       help='输入的 MANO .npz 文件路径或包含多个 .npz 文件的目录')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出的 SMPL-X .npz 文件路径 (可选)')
    parser.add_argument('--batch', action='store_true',
                       help='批量转换目录中的所有 .npz 文件')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='静默模式，不打印详细信息')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # 检查输入
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入路径不存在: {args.input}")
        sys.exit(1)
    
    # 批量处理目录
    if args.batch or os.path.isdir(args.input):
        if not os.path.isdir(args.input):
            print(f"❌ 错误: --batch 模式需要输入一个目录")
            sys.exit(1)
        
        print(f"📂 批量转换目录: {args.input}")
        npz_files = [f for f in os.listdir(args.input) 
                     if f.endswith('.npz') and 'smplx' not in f.lower()]
        
        if len(npz_files) == 0:
            print("❌ 错误: 目录中没有找到 .npz 文件")
            sys.exit(1)
        
        print(f"找到 {len(npz_files)} 个文件\n")
        success_count = 0
        for npz_file in npz_files:
            input_path = os.path.join(args.input, npz_file)
            print(f"\n{'='*80}")
            print(f"处理: {npz_file}")
            print('='*80)
            try:
                convert_mano_to_smplx(input_path, verbose=verbose)
                success_count += 1
            except Exception as e:
                print(f"❌ 失败: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"✅ 批量转换完成: {success_count}/{len(npz_files)} 成功")
        print('='*80)
    else:
        # 单文件处理
        print('='*80)
        try:
            convert_mano_to_smplx(args.input, args.output, verbose=verbose)
        except Exception as e:
            print(f"\n❌ 转换失败: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
