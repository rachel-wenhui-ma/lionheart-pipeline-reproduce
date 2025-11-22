# 文件清理评估

## 文件分类

### ✅ 核心文件（必须保留）

**主要代码：**
- `run_pipeline.py` - 主pipeline脚本
- `src/` 目录下所有文件 - 核心功能模块
  - `extract_insert_size.py` - 片段长度提取
  - `compute_coverage.py` - 覆盖度计算和校正
  - `compute_correlation.py` - 相关性计算
  - `assemble_features.py` - 特征组装
  - `utils.py` - 工具函数
  - `outlier_detection.py` - ZIPoisson异常值检测
  - `insert_size_correction.py` - Insert size校正

**文档：**
- `README.md` - 项目说明文档

**工具：**
- `download_reference.py` - 下载参考基因组（可能还有用）

**数据：**
- `data/` 目录 - 测试数据和masks
- `features_sample.csv` - 示例输出文件

---

### ❌ 阶段性Markdown文档（可以删除）

这些文档记录了开发过程，但内容已过时或已完成：

1. **`COMMIT_PLAN.md`** - Git提交计划
   - 状态：已完成所有提交
   - 建议：删除

2. **`COMMIT1_INSTRUCTIONS.md`** - 第一次提交说明
   - 状态：已完成
   - 建议：删除

3. **`TESTING_GUIDE.md`** - 测试指南
   - 状态：内容可能过时（测试文件可能已改变）
   - 建议：删除（如果需要可以重新写）

4. **`VALIDATION_SUMMARY.md`** - 验证总结
   - 状态：内容过时（基于旧版本pipeline）
   - 建议：删除

5. **`FILE_ORGANIZATION.md`** - 文件组织说明
   - 状态：内容可能过时
   - 建议：删除

6. **`SIMPLIFICATION_ASSESSMENT.md`** - 简化评估
   - 状态：内容已过时（我们已经实现了这些功能）
   - 建议：删除

7. **`IMPLEMENTATION_NOTES.md`** - 实现笔记
   - 状态：内容已过时（功能已实现）
   - 建议：删除

---

### ⚠️ 测试/验证文件（评估）

**可能还有用的测试文件：**
- `test_comprehensive.py` - 综合测试套件
  - 状态：可能还有用，但需要更新以匹配当前pipeline
  - 建议：保留但需要更新

- `validate_pipeline.py` - Pipeline验证
  - 状态：可能还有用，但需要更新
  - 建议：保留但需要更新

- `compare_with_lionheart.py` - 与LIONHEART比较
  - 状态：可能还有用，但需要更新
  - 建议：保留但需要更新

**可以删除的测试文件：**
- `test_pipeline_step2.py` - 阶段性测试
- `test_additional.py` - 额外测试
- `test_megabin.py` - Megabin测试
- `test_gc_correction.py` - GC校正测试
- `test_import.py` - 导入测试
- `check_features.py` - 特征检查
- `check_bin_indices.py` - Bin索引检查
- `check_bin_indices2.py` - Bin索引检查2
- `debug_gc_factors.py` - GC因子调试
- `compare_with_without_gc.py` - GC比较
- `fix_compute_coverage.py` - 修复脚本

---

### ❌ 其他文件（可以删除）

- `prepare_commits.py` - 准备提交的脚本
  - 状态：已完成所有提交
  - 建议：删除

- `test_features_output.csv` - 测试输出
  - 状态：临时文件
  - 建议：删除

---

## 清理建议

### 立即删除（已完成/过时）：
1. 所有阶段性Markdown文档（7个）
2. `prepare_commits.py`
3. 阶段性测试文件（`test_pipeline_step2.py`, `test_additional.py`, `test_megabin.py`, `test_gc_correction.py`, `test_import.py`）
4. 调试/检查文件（`check_*.py`, `debug_*.py`, `fix_*.py`, `compare_with_without_gc.py`）
5. `test_features_output.csv`

### 保留但需要更新：
1. `test_comprehensive.py` - 更新以匹配当前pipeline
2. `validate_pipeline.py` - 更新以匹配当前pipeline
3. `compare_with_lionheart.py` - 更新以匹配当前pipeline

### 保留：
1. 所有核心代码文件
2. `README.md`
3. `download_reference.py`
4. `data/` 目录
5. `features_sample.csv`（作为示例）

---

## 缺失功能检查

检查是否有之前做了但现在没做而又确实有必要做的步骤：

### ✅ 已实现的功能：
1. ZIPoisson异常值检测 - ✅ 已实现
2. GC校正 - ✅ 已实现
3. Insert size校正（noise, skewness, mean shift） - ✅ 已实现
4. Megabin归一化 - ✅ 已实现
5. 10种特征类型 - ✅ 已实现

### ⚠️ 可能需要但未实现的功能：
1. **与LIONHEART输出的直接比较** - 需要更新`compare_with_lionheart.py`
2. **完整的验证测试** - 需要更新`validate_pipeline.py`
3. **扩展到所有染色体** - 当前只处理chr21（这是有意的简化）
4. **扩展到所有898个cell types** - 当前只有3个测试cell types（这是有意的简化）

### ✅ 结论：
没有遗漏关键功能。所有核心功能都已实现。测试文件需要更新以匹配当前pipeline。

