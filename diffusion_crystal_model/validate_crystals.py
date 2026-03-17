"""
validate_crystals.py

AI 生成晶体的全自动物理质检模块
1. Spglib 对称性分析 (Pymatgen SpacegroupAnalyzer)
2. M3GNet / ASE 通用机器学习力场结构弛豫 (Relaxation)
3. 生成质检报告 (CSV)
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
    print("⚠️ 警告: 未检测到 matgl 或 ase 库。只能进行对称性分析，将跳过结构弛豫。")
    print("💡 建议运行: pip install matgl ase")

def analyze_symmetry(structure, symprec=0.1):
    """
    使用 spglib 分析结构的对称性。
    对于未经松弛的 AI 生成结构，容差 (symprec) 适当放宽。
    """
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
        return f"{sg_symbol} ({sg_number})"
    except Exception:
        return "P1 (1) - 无明显对称性"

def validate_and_relax(input_dir, output_dir, report_csv="validation_report.csv"):
    os.makedirs(output_dir, exist_ok=True)
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))
    
    if not cif_files:
        print(f"❌ 在 {input_dir} 中没有找到 CIF 文件！")
        return

    print(f"\n🔬 开始对 {len(cif_files)} 个 AI 生成晶体进行物理质检...\n" + "="*50)

    # 如果有 matgl，预先加载 M3GNet 力场 (只加载一次以节省时间)
    relaxer = None
    if HAS_MATGL:
        print("⏳ 正在加载 M3GNet 通用力场模型，请稍候...")
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        relaxer = Relaxer(potential=pot)
        print("✅ 力场加载完毕！\n")

    results = []

    for idx, cif_path in enumerate(cif_files):
        filename = os.path.basename(cif_path)
        print(f"[{idx+1}/{len(cif_files)}] 正在检验: {filename}")
        
        try:
            struct = Structure.from_file(cif_path)
            init_vol = struct.volume
            init_sym = analyze_symmetry(struct, symprec=0.1) # 生成初态容差放宽
            
            report_data = {
                "Filename": filename,
                "Formula": struct.composition.reduced_formula,
                "Num_Atoms": len(struct),
                "Init_Symmetry": init_sym,
                "Init_Volume": round(init_vol, 2),
            }

            # --- 物理弛豫 (Relaxation) ---
            if relaxer:
                # fmax=0.05 是受力收敛标准 (eV/Å)
                relax_results = relaxer.relax(struct, fmax=0.05)
                
                final_struct = relax_results['final_structure']
                final_energy = relax_results['trajectory'].energies[-1]
                final_vol = final_struct.volume
                final_sym = analyze_symmetry(final_struct, symprec=0.01) # 松弛后要求严格对称
                
                # 计算体积形变率 (越小说明 AI 生成得越准)
                vol_change_percent = abs(final_vol - init_vol) / init_vol * 100

                report_data.update({
                    "Final_Symmetry": final_sym,
                    "Final_Volume": round(final_vol, 2),
                    "Volume_Change_%": round(vol_change_percent, 2),
                    "System_Energy_eV": round(final_energy, 4),
                    "Energy_per_Atom_eV": round(final_energy / len(struct), 4),
                    "Status": "✅ 松弛成功"
                })

                # 保存优化后的晶体
                relaxed_path = os.path.join(output_dir, filename.replace(".cif", "_relaxed.cif"))
                final_struct.to(filename=relaxed_path)
                print(f"   => 初始空间群: {init_sym} | 松弛后空间群: {final_sym}")
                print(f"   => 最终能量: {report_data['Energy_per_Atom_eV']} eV/atom | 体积变化: {report_data['Volume_Change_%']}%")
                
            else:
                report_data["Status"] = "⚠️ 未弛豫 (仅分析对称性)"
                print(f"   => 初始空间群: {init_sym}")

            results.append(report_data)

        except Exception as e:
            print(f"   ❌ 检验失败 (结构可能崩溃): {e}")
            results.append({"Filename": filename, "Status": "❌ 结构崩溃或不合理"})
            
        print("-" * 30)

    # 保存报告
    df = pd.DataFrame(results)
    df.to_csv(report_csv, index=False)
    print("\n" + "="*50)
    print(f"🎉 质检全部完成！")
    print(f"📄 详细报告已保存至: {report_csv}")
    if HAS_MATGL:
        print(f"💎 松弛后的稳态晶体已保存至: {output_dir}")

if __name__ == "__main__":
    # 输入：AI 扩散模型生成的原始 CIF 文件夹
    INPUT_DIR = "ai_diffusion_materials"
    
    # 输出：M3GNet 优化后的 CIF 文件夹
    OUTPUT_DIR = "ai_diffusion_relaxed"
    
    # 🌟 修复：调用正确的函数名 validate_and_relax
    validate_and_relax(INPUT_DIR, OUTPUT_DIR)