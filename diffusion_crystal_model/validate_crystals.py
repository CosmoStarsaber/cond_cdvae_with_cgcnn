"""
validate_crystals.py

AI 生成晶体的全自动多维物理质检模块 (生产级完全体)
包含：Spglib 对称性分析、化合价电中性检查、原子碰撞硬截断过滤、M3GNet 结构弛豫。
"""

import os
import glob
import warnings
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

# 尝试导入松弛所需的库
try:
    import matgl
    from matgl.ext.ase import Relaxer
    import logging
    logging.getLogger("matgl").setLevel(logging.ERROR)
    HAS_MATGL = True
except ImportError:
    HAS_MATGL = False
    print("⚠️ 警告: 未检测到 matgl 或 ase 库。将跳过结构弛豫流程。")

def analyze_symmetry(structure, symprec=0.1):
    """提取精确的空间群对称性"""
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        dataset = sga.get_symmetry_dataset()
        if dataset is not None:
            return f"{dataset['international']} ({dataset['number']})"
        return "P1 (1)"
    except Exception:
        return "分析出错"

def check_charge_neutrality(structure):
    """检查生成的化学式是否可能满足电中性 (基于元素常规氧化态)"""
    try:
        # 尝试推测合理的化合价组合，如果返回列表不为空，说明存在电中性可能
        guesses = structure.composition.oxi_state_guesses()
        return len(guesses) > 0
    except Exception:
        return False

def check_unphysical_bonds(structure, min_dist=0.7):
    """
    检查是否存在物理上不可能的极短键长 (原子碰撞)
    min_dist 默认设为 0.7 埃，拦截绝对不可能的靠拢。
    """
    try:
        # 获取所有距离小于 min_dist 的邻居对 (包含 PBC)
        all_neighbors = structure.get_all_neighbors(r=min_dist)
        for neighbors in all_neighbors:
            if len(neighbors) > 0:
                return True  # 发现了原子碰撞
        return False
    except Exception:
        return True  # 如果计算图崩了，也视为不合理结构

def validate_and_relax(input_dir, output_dir, report_csv="validation_report.csv"):
    os.makedirs(output_dir, exist_ok=True)
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))
    
    if not cif_files:
        print(f"❌ 在 {input_dir} 中没有找到 CIF 文件！请确保 train.py 已经生成了样本。")
        return

    print(f"\n🔬 开始对 {len(cif_files)} 个 AI 生成晶体进行多维物理质检...\n" + "="*50)

    relaxer = None
    if HAS_MATGL:
        print("⏳ 正在加载 M3GNet 通用力场模型，请稍候...")
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        relaxer = Relaxer(potential=pot)
        print("✅ 力场加载完毕！\n")

    results = []

    for idx, cif_path in enumerate(cif_files):
        filename = os.path.basename(cif_path)
        print(f"[{idx+1}/{len(cif_files)}] 质检中: {filename}")
        
        try:
            struct = Structure.from_file(cif_path)
            init_vol = struct.volume
            
            # 🌟 多维度交叉验证
            init_sym = analyze_symmetry(struct, symprec=0.1)
            is_neutral = check_charge_neutrality(struct)
            has_collision = check_unphysical_bonds(struct, min_dist=0.7)
            
            report_data = {
                "Filename": filename,
                "Formula": struct.composition.reduced_formula,
                "Num_Atoms": len(struct),
                "Init_Symmetry": init_sym,
                "Charge_Neutral": is_neutral,
                "Atom_Collision": has_collision,
                "Init_Volume": round(init_vol, 2),
            }

            # 🛑 核心防爆机制：拦截碰撞废片
            if has_collision:
                report_data["Status"] = "❌ 原子碰撞拦截"
                print(f"   ⚠️ 警告：检测到原子极度靠近 (<0.7Å)，判定为非物理结构，放弃弛豫。")
                results.append(report_data)
                print("-" * 40)
                continue

            # --- 物理弛豫 (Relaxation) ---
            if relaxer:
                # 只有结构合理的晶体才会交由力场处理
                relax_results = relaxer.relax(struct, fmax=0.05)
                
                final_struct = relax_results['final_structure']
                final_energy = relax_results['trajectory'].energies[-1]
                final_vol = final_struct.volume
                final_sym = analyze_symmetry(final_struct, symprec=0.01)
                
                vol_change_percent = abs(final_vol - init_vol) / init_vol * 100

                report_data.update({
                    "Final_Symmetry": final_sym,
                    "Final_Volume": round(final_vol, 2),
                    "Volume_Change_%": round(vol_change_percent, 2),
                    "Energy_per_Atom_eV": round(final_energy / len(struct), 4),
                    "Status": "✅ 松弛成功"
                })

                relaxed_path = os.path.join(output_dir, filename.replace(".cif", "_relaxed.cif"))
                final_struct.to(filename=relaxed_path)
                
                print(f"   => 对称性: {init_sym} -> {final_sym}")
                print(f"   => 电中性: {'✅' if is_neutral else '❌'} | 能量: {report_data['Energy_per_Atom_eV']} eV/atom")
                print(f"   => 体积形变率: {report_data['Volume_Change_%']}%")
                
            else:
                report_data["Status"] = "⚠️ 未弛豫 (仅分析性质)"
                print(f"   => 对称性: {init_sym} | 电中性: {'✅' if is_neutral else '❌'}")

            results.append(report_data)

        except Exception as e:
            print(f"   ❌ 文件解析失败: {e}")
            results.append({"Filename": filename, "Status": "❌ 文件损坏或无法解析"})
            
        print("-" * 40)

    # 导出统计报表
    df = pd.DataFrame(results)
    df.to_csv(report_csv, index=False)
    
    # 打印最终总结
    valid_count = len(df[df["Status"] == "✅ 松弛成功"])
    print("\n" + "="*50)
    print(f"🎉 质检流水线执行完毕！")
    print(f"📊 报告统计: 共处理 {len(cif_files)} 个结构，成功松弛并存活 {valid_count} 个。")
    print(f"📄 详细数据大表已保存至: {report_csv}")
    print("="*50)

if __name__ == "__main__":
    # 指向我们在 train.py 中设定的生成目录
    INPUT_DIR = "generated_cifs"
    OUTPUT_DIR = "relaxed_cifs"
    
    validate_and_relax(INPUT_DIR, OUTPUT_DIR)