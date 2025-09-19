from scipy.stats import ttest_ind
# 假设你的方法结果（10次实验）和基线方法结果
your_method = [0.628, 0.651, 0.667, 0.653, 0.649]  # 你的BERTScore结果（10个值）
baseline    = [0.604, 0.58, 0.602, 0.627, 0.605]  # 基线的BERTScore结果（10个值）
# 做独立样本t检验
t_stat, p_value = ttest_ind(your_method, baseline)
print("P值:", p_value)