"""
download_mp_data_10k.py

万级数据获取脚本：从 Materials Project 获取 10000 个真实材料结构。
提取特征：【形成能】 (Formation Energy) 和 【带隙】 (Band Gap)
"""

import os
import csv
from mp_api.client import MPRester

# ==========================================
# 配置参数
# ==========================================
API_KEY = "slGl489IGxXfnc38Os2S5uJKHXDban5Q"  # ⚠️ 请务必替换为你的真实 API Key
OUTPUT_DIR = "real_mp_dataset"
MAX_SAMPLES = 10000  # 🌟 已扩大到 10000 个样本

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "id_prop.csv")

    print("开始连接 Materials Project 数据库...")
    
    with MPRester(API_KEY) as mpr:
        print("正在检索符合条件的稳定材料（可能需要几十秒，请稍候）...")
        # 检索原子数 <= 20 且热力学稳定的材料
        docs = mpr.summary.search(
            num_sites=(1, 20),
            is_stable=True,
            fields=["material_id", "structure", "formation_energy_per_atom", "band_gap"]
        )

    print(f"成功从云端检索到 {len(docs)} 个符合条件的材料！")
    
    # Python 切片会安全处理：即使云端符合条件的不到 10000 个，也会全部下载且不报错
    docs_to_download = docs[:MAX_SAMPLES]
    actual_count = len(docs_to_download)
    
    print(f"准备提取并保存前 {actual_count} 个样本...")
    print("注意：在本地生成万级 CIF 文件需要数分钟时间，请保持程序运行。")

    # 写入 CSV 并保存 CIF 文件
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        for i, doc in enumerate(docs_to_download):
            mp_id = str(doc.material_id)
            energy = doc.formation_energy_per_atom
            bandgap = doc.band_gap
            structure = doc.structure
            
            # 保存 CIF 文件
            cif_filename = os.path.join(OUTPUT_DIR, f"{mp_id}.cif")
            structure.to(filename=cif_filename)
            
            # 写入三列数据 -> [ID, 形成能, 带隙]
            writer.writerow([mp_id, energy, bandgap])
            
            # 优化打印频率：每处理 500 个打印一次进度，或者在最后一个打印
            if (i + 1) % 500 == 0 or (i + 1) == actual_count:
                print(f"已处理 {i + 1}/{actual_count} 个材料...")

    print(f"\n🎉 万级数据集下载及处理完成！")
    print(f"请检查 '{OUTPUT_DIR}' 文件夹，id_prop.csv 和所有 CIF 文件均已准备就绪。")

if __name__ == "__main__":
    main()