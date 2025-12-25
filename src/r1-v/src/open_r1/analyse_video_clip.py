import os
import torch
import numpy as np

def inspect_pt_file(pt_path: str):
    """读取并打印一个 .pt 文件中的特征信息"""
    if not os.path.exists(pt_path):
        print(f"文件不存在: {pt_path}")
        return

    print(f"正在读取: {pt_path}")
    obj = torch.load(pt_path, map_location="cpu")

    print("\n====== 基本信息 ======")
    print(f"类型(type): {type(obj)}")

    if isinstance(obj, torch.Tensor):
        print(f"张量形状(shape): {tuple(obj.shape)}")
        print(f"数据类型(dtype): {obj.dtype}")
        print(f"设备(device): {obj.device}")

        # 转成 numpy 方便查看具体数值
        arr = obj.numpy()
        
        import pdb; pdb.set_trace()

        print("\n====== 数值示例 ======")
        if arr.ndim == 2:
            T, D = arr.shape
            print(f"帧数 T = {T}, 特征维度 D = {D}")
            print("\n前 2 帧的前 5 个维度：")
            print(arr[:2, :5])
        else:
            print("张量维度不是 2D，全量打印前几项：")
            flat = arr.flatten()
            print(flat[:20])

        # 简单检查是否做了 L2 归一化
        print("\n====== L2 范数检查（每一帧） ======")
        if arr.ndim == 2:
            norms = np.linalg.norm(arr, axis=1)
            print("前 5 帧的 L2 范数：", norms[:5])
            print("范数均值: ", norms.mean())
        else:
            print("不是 2D 张量，跳过范数检查。")

    else:
        # 如果里面不是单一 tensor，而是 dict / list
        print("对象不是 torch.Tensor，内容概览：")
        print(obj)

def main():
    # 方式 1：直接在这里写死路径
    pt_path = "/data/wm/Video-R1/src/r1-v/src/open_r1/clip_embeddings/7b9efeae03237333_sub1_6.pt"

    # # 方式 2：运行时输入路径（推荐）
    # pt_path = input("请输入要检查的 .pt 文件完整路径：\n> ").strip()

    if not pt_path:
        print("未输入路径，退出。")
        return

    inspect_pt_file(pt_path)

if __name__ == "__main__":
    main()
