"""
运行所有实验：对比实验、消融实验、性能测试
"""

import os
import sys
import io

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.experiments.comparison import ClassifierComparison
from src.experiments.ablation import AblationStudy
from src.experiments.performance import PerformanceTest


def main():
    """运行所有实验"""
    
    print("\n" + "="*80)
    print("🔬 语音识别系统完整实验评估 | Complete Experimental Evaluation")
    print("="*80)
    
    # 数据目录
    train_dir = os.path.join(project_root, 'data', 'train')
    test_dir = os.path.join(project_root, 'data', 'test')
    
    # 检查数据目录
    if not os.path.exists(train_dir):
        print(f"\n❌ 错误: 训练数据目录不存在: {train_dir}")
        print("   请将训练数据放入 data/train/ 目录下，按以下结构组织:")
        print("   data/train/digit_0/")
        print("   data/train/digit_1/")
        print("   ...")
        return
    
    # 检查是否有测试集
    use_test_set = os.path.exists(test_dir)
    if not use_test_set:
        print(f"\n⚠️  警告: 测试数据目录不存在: {test_dir}")
        print("   将使用训练集的一部分进行测试\n")
        test_dir = None
    
    # 询问用户要运行哪些实验
    print("\n请选择要运行的实验:")
    print("1️⃣  对比实验 - Classifier Comparison")
    print("2️⃣  消融实验 - Ablation Study")
    print("3️⃣  性能测试 - Performance Test")
    print("4️⃣  运行所有实验 - Run All Experiments")
    print("0️⃣  退出 - Exit")
    
    choice = input("\n请输入选项 (1-4, 0=退出): ").strip()
    
    if choice == '0':
        print("\n👋 再见！")
        return
    
    # ========== 1. 对比实验 ==========
    if choice in ['1', '4']:
        print("\n" + "🔹"*40)
        print("开始运行对比实验...")
        print("🔹"*40 + "\n")
        
        comparison = ClassifierComparison(train_dir, test_dir)
        comparison.run_comparison(use_cross_validation=(test_dir is None))
        comparison.visualize_results(
            output_dir=os.path.join(project_root, 'data', 'results', 'comparison')
        )
    
    # ========== 2. 消融实验 ==========
    if choice in ['2', '4']:
        print("\n" + "🔹"*40)
        print("开始运行消融实验...")
        print("🔹"*40 + "\n")
        
        ablation = AblationStudy(train_dir, test_dir)
        ablation.run_ablation_study()
        ablation.visualize_results(
            output_dir=os.path.join(project_root, 'data', 'results', 'ablation')
        )
    
    # ========== 3. 性能测试 ==========
    if choice in ['3', '4']:
        print("\n" + "🔹"*40)
        print("开始运行性能测试...")
        print("🔹"*40 + "\n")
        
        performance = PerformanceTest(train_dir, test_dir)
        performance.run_performance_tests()
        performance.visualize_results(
            output_dir=os.path.join(project_root, 'data', 'results', 'performance')
        )
    
    # ========== 生成综合报告 ==========
    if choice == '4':
        generate_summary_report(project_root)
    
    print("\n" + "="*80)
    print("✅ 所有实验完成！")
    print("="*80)
    print(f"\n📊 实验结果已保存到: {os.path.join(project_root, 'data', 'results')}")
    print("\n您可以查看以下目录获取详细结果:")
    print(f"   - 对比实验: data/results/comparison/")
    print(f"   - 消融实验: data/results/ablation/")
    print(f"   - 性能测试: data/results/performance/")
    print()


def generate_summary_report(project_root: str):
    """生成综合实验报告"""
    
    print("\n" + "="*80)
    print("📝 生成综合实验报告")
    print("="*80 + "\n")
    
    results_dir = os.path.join(project_root, 'data', 'results')
    report_path = os.path.join(results_dir, 'EXPERIMENT_REPORT.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 语音识别系统实验报告 | Speech Recognition System Experiment Report\n\n")
        f.write("---\n\n")
        
        # 实验概述
        f.write("## 1. 实验概述 | Experiment Overview\n\n")
        f.write("本报告包含了语音识别系统的完整实验评估，包括：\n\n")
        f.write("- **对比实验** - 比较不同分类器算法的性能\n")
        f.write("- **消融实验** - 分析不同特征组合对识别准确率的影响\n")
        f.write("- **性能测试** - 评估系统的运行效率和资源占用\n\n")
        
        # 实验结果
        f.write("## 2. 实验结果 | Experimental Results\n\n")
        
        f.write("### 2.1 对比实验结果 | Classifier Comparison Results\n\n")
        f.write("**可视化结果:**\n")
        f.write("- 指标对比图: `comparison/metrics_comparison.png`\n")
        f.write("- 时间对比图: `comparison/time_comparison.png`\n")
        f.write("- 混淆矩阵: `comparison/confusion_matrix_*.png`\n\n")
        
        f.write("**主要发现:**\n")
        f.write("- 测试了6种分类器算法：模板匹配、朴素贝叶斯、Fisher判别、决策树、SVM、KNN\n")
        f.write("- 每种分类器在准确率、精确率、召回率、F1分数等指标上的表现\n")
        f.write("- 不同分类器的训练和预测时间对比\n\n")
        
        f.write("### 2.2 消融实验结果 | Ablation Study Results\n\n")
        f.write("**可视化结果:**\n")
        f.write("- 特征组合对比: `ablation/feature_comparison.png`\n")
        f.write("- 特征重要性分析: `ablation/feature_importance.png`\n\n")
        
        f.write("**主要发现:**\n")
        f.write("- 测试了7种特征组合\n")
        f.write("- 单特征：短时能量、过零率、平均幅度\n")
        f.write("- 双特征组合：能量+过零率、能量+幅度、过零率+幅度\n")
        f.write("- 全特征：能量+过零率+幅度\n")
        f.write("- 分析了每种特征对识别性能的贡献度\n\n")
        
        f.write("### 2.3 性能测试结果 | Performance Test Results\n\n")
        f.write("**可视化结果:**\n")
        f.write("- 训练时间对比: `performance/training_time.png`\n")
        f.write("- 预测时间对比: `performance/prediction_time.png`\n")
        f.write("- 内存占用对比: `performance/memory_usage.png`\n")
        f.write("- 综合性能对比: `performance/comprehensive_performance.png`\n\n")
        
        f.write("**主要发现:**\n")
        f.write("- 评估了各分类器的训练时间、预测时间、内存占用、吞吐量\n")
        f.write("- 分析了准确率与效率之间的权衡\n")
        f.write("- 识别出最适合实际应用的分类器\n\n")
        
        # 评价指标说明
        f.write("## 3. 评价指标说明 | Evaluation Metrics\n\n")
        
        f.write("### 3.1 准确率 (Accuracy)\n")
        f.write("- **定义**: 正确分类的样本数 / 总样本数\n")
        f.write("- **意义**: 衡量整体分类正确性\n\n")
        
        f.write("### 3.2 精确率 (Precision)\n")
        f.write("- **定义**: 真正例 / (真正例 + 假正例)\n")
        f.write("- **意义**: 预测为正的样本中真正为正的比例\n\n")
        
        f.write("### 3.3 召回率 (Recall)\n")
        f.write("- **定义**: 真正例 / (真正例 + 假负例)\n")
        f.write("- **意义**: 实际为正的样本中被正确识别的比例\n\n")
        
        f.write("### 3.4 F1分数 (F1-Score)\n")
        f.write("- **定义**: 2 * (精确率 * 召回率) / (精确率 + 召回率)\n")
        f.write("- **意义**: 精确率和召回率的调和平均，综合评价指标\n\n")
        
        f.write("### 3.5 混淆矩阵 (Confusion Matrix)\n")
        f.write("- **定义**: 展示真实标签与预测标签的对应关系\n")
        f.write("- **意义**: 直观显示每个类别的识别情况和混淆情况\n\n")
        
        # 结论
        f.write("## 4. 结论与建议 | Conclusions and Recommendations\n\n")
        
        f.write("### 4.1 分类器选择建议\n")
        f.write("- **追求最高准确率**: 选择F1分数最高的分类器\n")
        f.write("- **追求实时性**: 选择预测时间最短的分类器\n")
        f.write("- **资源受限场景**: 选择内存占用最小的分类器\n")
        f.write("- **平衡性能**: 综合考虑准确率、速度和资源占用\n\n")
        
        f.write("### 4.2 特征工程建议\n")
        f.write("- 根据消融实验结果，选择对识别性能贡献最大的特征组合\n")
        f.write("- 考虑特征提取的计算成本与性能提升的权衡\n")
        f.write("- 可以尝试添加更多时域或频域特征以提升性能\n\n")
        
        f.write("### 4.3 系统优化方向\n")
        f.write("- **数据增强**: 增加训练样本数量，提高模型泛化能力\n")
        f.write("- **特征优化**: 提取更多有效特征，如MFCC、频域特征等\n")
        f.write("- **算法改进**: 尝试深度学习方法（如CNN、RNN）\n")
        f.write("- **端到端优化**: 从特征提取到分类的全流程优化\n\n")
        
        # 附录
        f.write("## 5. 附录 | Appendix\n\n")
        f.write("### 5.1 实验环境\n")
        f.write("- Python版本: 3.x\n")
        f.write("- 主要依赖库: numpy, scikit-learn, matplotlib, scipy\n\n")
        
        f.write("### 5.2 数据集说明\n")
        f.write("- 数字范围: 0-9（共10类）\n")
        f.write("- 音频格式: WAV\n")
        f.write("- 数据组织: 按数字分类存储\n\n")
        
        f.write("---\n\n")
        f.write("*报告生成时间: {}*\n".format(__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    print(f"✅ 实验报告已生成: {report_path}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  实验被用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

