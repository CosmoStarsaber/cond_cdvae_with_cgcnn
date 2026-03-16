"""
download_mp_data_v2.py

更新版：获取真实材料数据，并提取【形成能】和【带隙】两个物理量
生成的三列 id_prop.csv 格式为: mp-id, formation_energy_per_atom, band_gap
"""

import os
import csv
from mp_api.client import MPRester

# ==========================================
# 配置参数
# ==========================================
API_KEY = "slGl489IGxXfnc38Os2S5uJKHXDban5Q"  # ⚠️ 请务必替换为你的真实 API Key
OUTPUT_DIR = "real_mp_dataset"
MAX_SAMPLES = 2000  # 保持 2000 个样本进行测试

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "id_prop.csv")

    print("开始连接 Materials Project 数据库...")
    
    with MPRester(API_KEY) as mpr:
        print("正在检索符合条件的稳定材料 (包含形成能和带隙)，请稍候...")
        # 🌟 关键修改：在 fields 中增加了 "band_gap" 字段
        docs = mpr.summary.search(
            num_sites=(1, 20),
            is_stable=True,
            fields=["material_id", "structure", "formation_energy_per_atom", "band_gap"]
        )

    print(f"成功检索到 {len(docs)} 个符合条件的材料！")
    
    docs_to_download = docs[:MAX_SAMPLES]
    print(f"准备更新前 {len(docs_to_download)} 个样本并保存...")

    # 写入 CSV 并保存 CIF 文件
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        for i, doc in enumerate(docs_to_download):
            mp_id = str(doc.material_id)
            energy = doc.formation_energy_per_atom
            bandgap = doc.band_gap  # 🌟 提取带隙数据
            structure = doc.structure
            
            # 保存 CIF 文件 (如果之前已经下载过，这里会直接覆盖，确保数据完美对齐)
            cif_filename = os.path.join(OUTPUT_DIR, f"{mp_id}.cif")
            structure.to(filename=cif_filename)
            
            # 🌟 关键修改：写入三列数据 -> [ID, 形成能, 带隙]
            writer.writerow([mp_id, energy, bandgap])
            
            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(docs_to_download)} 个材料...")

    print(f"\n🎉 数据更新完成！")
    print(f"请检查 '{OUTPUT_DIR}/id_prop.csv'，它现在应该包含 3 列数据了！")

if __name__ == "__main__":
    main()