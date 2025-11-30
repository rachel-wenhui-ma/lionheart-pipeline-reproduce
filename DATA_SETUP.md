# 数据文件设置说明

## 大文件处理

由于GitHub对文件大小的限制（单个文件建议<50MB，硬限制100MB），以下大文件**不会**提交到Git仓库：

### 被忽略的文件（.gitignore）

1. **LIONHEART资源文件** (~1GB)
   - `data/lionheart_resources/` - 整个目录
   - 包含22个parquet文件（每个~30-85MB）
   - 需要从Zenodo下载：https://zenodo.org/records/15747531/files/inference_resources_v003.tar.gz

2. **参考基因组文件**
   - `data/chr*.fa` - 染色体FASTA文件
   - `data/chr*.fa.fai` - 索引文件
   - 可以使用 `download_reference.py` 下载

3. **BAM文件**
   - `*.bam` 和 `*.bai` - 比对文件和索引
   - 示例BAM文件在项目外部目录

4. **输出文件**
   - `*.csv`, `*.npy`, `*.npz` - 特征提取输出文件

## 如何获取数据文件

### 方法1：使用setup脚本（推荐）

如果你已经有LIONHEART资源文件在 `/home/mara/lionheart_work/resources`：

```bash
python setup_lionheart_resources.py
```

这会自动复制所有必要的资源文件到 `data/lionheart_resources/`。

### 方法2：手动下载和设置

1. **下载LIONHEART资源**：
   ```bash
   wget https://zenodo.org/records/15747531/files/inference_resources_v003.tar.gz
   tar -xvzf inference_resources_v003.tar.gz
   # 将解压后的目录内容复制到 reproduce/data/lionheart_resources/
   ```

2. **下载参考基因组**：
   ```bash
   python download_reference.py --chromosome chr21  # 测试用
   # 或下载全部22条染色体：
   python download_reference.py --all_autosomes
   ```

3. **准备BAM文件**：
   - 使用你自己的hg38比对BAM文件
   - 或使用示例BAM文件（在项目外部）

## 目录结构

设置完成后，`data/` 目录应该包含：

```
data/
├── lionheart_resources/          # LIONHEART资源文件（~1GB，不提交到Git）
│   ├── bin_indices_by_chromosome/
│   │   ├── chr1.parquet
│   │   ├── chr2.parquet
│   │   └── ...
│   ├── gc_contents_bin_edges.npy
│   ├── insert_size_bin_edges.npy
│   ├── outliers/
│   │   ├── outlier_indices.npz
│   │   └── zero_coverage_indices.npz
│   ├── DNase.idx_to_cell_type.csv
│   └── ATAC.idx_to_cell_type.csv
├── chr1.fa, chr2.fa, ...        # 参考基因组（不提交到Git）
├── masks/                        # 测试用mask文件（小文件，已提交）
│   ├── Tcell.bed
│   ├── Monocyte.bed
│   └── Liver.bed
└── (BAM文件在外部目录)
```

## 验证设置

运行测试脚本验证所有文件是否正确设置：

```bash
python test_paths.py              # 测试路径配置
python test_new_resources_path.py # 测试资源文件加载
```

## 注意事项

- **不要**尝试将大文件提交到Git（会被拒绝或导致仓库过大）
- 所有大文件都在 `.gitignore` 中
- 在README中说明如何获取这些文件
- 团队成员需要单独下载资源文件


