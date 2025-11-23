#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试转换脚本，验证 MANO 到 SMPL-X 的转换是否正确
"""

import numpy as np
import os
import sys

def test_conversion():
    """测试转换功能"""
    
    print("="*80)
    print("🧪 测试 MANO → SMPL-X 转换")
    print("="*80)
    
    # 查找输出文件
    output_dir = "/Users/yth/project/human-motion/Dyn-HaMR/outputs/2025-11-05/web2-all-shot-0-0-500/smooth_fit"
    
    if not os.path.exists(output_dir):
        print(f"❌ 输出目录不存在: {output_dir}")
        return False
    
    # 查找结果文件
    npz_files = [f for f in os.listdir(output_dir) if f.endswith('_world_results.npz') and 'smplx' not in f]
    
    if not npz_files:
        print(f"❌ 在 {output_dir} 中没有找到结果文件")
        return False
    
    # 选择第一个文件进行测试
    test_file = os.path.join(output_dir, sorted(npz_files)[-1])  # 使用最后一个（通常是最终结果）
    
    print(f"\n📁 测试文件: {os.path.basename(test_file)}")
    print("-"*80)
    
    # 加载原始数据
    try:
        mano_data = np.load(test_file)
        print("\n✅ 成功加载 MANO 数据")
        print("\n原始数据字段:")
        for key in sorted(mano_data.keys()):
            print(f"  {key:20s}: {str(mano_data[key].shape):20s}")
        
        # 检查关键字段
        required_fields = ['pose_body', 'root_orient', 'trans']
        missing = [f for f in required_fields if f not in mano_data]
        if missing:
            print(f"\n❌ 缺少必需字段: {missing}")
            return False
        
        # 分析数据
        pose_body = mano_data['pose_body']
        is_right = mano_data.get('is_right', None)
        
        print(f"\n📊 数据分析:")
        print(f"  形状维度: {pose_body.shape}")
        
        if len(pose_body.shape) == 2:
            B, T = 1, pose_body.shape[0]
        else:
            B, T = pose_body.shape[0], pose_body.shape[1]
        
        print(f"  轨迹数 (B): {B}")
        print(f"  帧数 (T): {T}")
        print(f"  关节维度: {pose_body.shape[-1]} (应该是45)")
        
        # 分析手部类型
        if is_right is not None:
            print(f"\n🖐️  手部类型分析:")
            if len(is_right.shape) == 1:
                hand_type = "右手" if is_right[0] == 1 else "左手"
                print(f"  检测到: {hand_type}")
            else:
                for b in range(B):
                    hand_type = "右手" if is_right[b, 0] == 1 else "左手"
                    print(f"  轨迹 {b}: {hand_type}")
        
        # 执行转换
        print("\n" + "="*80)
        print("🔄 开始转换...")
        print("="*80 + "\n")
        
        sys.path.insert(0, '/Users/yth/project/human-motion')
        from convert_to_smplx import convert_mano_to_smplx
        
        output_files = convert_mano_to_smplx(test_file, verbose=True)
        
        # 验证转换结果
        print("\n" + "="*80)
        print("🔍 验证转换结果")
        print("="*80)
        
        if isinstance(output_files, str):
            output_files = [output_files]
        
        all_valid = True
        for output_file in output_files:
            if not os.path.exists(output_file):
                print(f"\n❌ 输出文件不存在: {output_file}")
                all_valid = False
                continue
            
            print(f"\n📄 检查: {os.path.basename(output_file)}")
            print("-"*80)
            
            smplx_data = np.load(output_file)
            
            # 检查必需的 SMPL-X 字段
            required_smplx = ['body_pose', 'global_orient', 'transl', 'betas', 
                            'right_hand_pose', 'left_hand_pose']
            
            print("SMPL-X 字段检查:")
            for field in required_smplx:
                if field in smplx_data:
                    print(f"  ✅ {field:20s}: {str(smplx_data[field].shape):20s}")
                else:
                    print(f"  ❌ {field:20s}: 缺失")
                    all_valid = False
            
            # 验证数据一致性
            print(f"\n数据一致性检查:")
            
            # 检查形状
            if 'global_orient' in smplx_data:
                converted_T = smplx_data['global_orient'].shape[0]
                print(f"  帧数匹配: {T} (原始) vs {converted_T} (转换后) - {'✅' if T == converted_T else '❌'}")
            
            # 检查 body_pose 是否为零
            if 'body_pose' in smplx_data:
                is_zero = np.allclose(smplx_data['body_pose'], 0)
                print(f"  body_pose 全零: {'✅' if is_zero else '❌ (应该全零)'}")
            
            # 检查手部姿态
            has_right = not np.allclose(smplx_data.get('right_hand_pose', 0), 0)
            has_left = not np.allclose(smplx_data.get('left_hand_pose', 0), 0)
            print(f"  右手数据: {'✅ 有' if has_right else '⭕ 无（全零）'}")
            print(f"  左手数据: {'✅ 有' if has_left else '⭕ 无（全零）'}")
            
            if not (has_right or has_left):
                print(f"  ⚠️  警告: 左右手数据都是零")
                all_valid = False
        
        if all_valid:
            print("\n" + "="*80)
            print("✅ 所有测试通过！")
            print("="*80)
            return True
        else:
            print("\n" + "="*80)
            print("⚠️  部分测试失败，请检查上述错误")
            print("="*80)
            return False
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_conversion()
    sys.exit(0 if success else 1)
