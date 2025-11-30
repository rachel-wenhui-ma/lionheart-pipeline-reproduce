# LIONHEART资源文件设置说明

## 资源文件来源

LIONHEART的资源文件（包括parquet文件）**不是**在安装时自动下载的，也不是包含在lionheart工程目录中的。

这些文件需要**手动从Zenodo下载**：
- 下载地址: https://zenodo.org/records/15747531/files/inference_resources_v003.tar.gz
- 解压后会得到一个包含所有资源文件的目录

## 资源文件位置

我们已经将资源文件复制到我们的数据目录中，和example_bam、参考基因组放在一起：

**路径**: `reproduce/data/lionheart_resources/`

包含以下关键文件和目录：
- `bin_indices_by_chromosome/` - 包含所有chr*.parquet文件（~1GB，22个文件）
  - 每个parquet文件包含`idx`和`GC`两列
  - GC值是从reference genome (2bit文件)提取的，在100bp context中计算
- `gc_contents_bin_edges.npy` - GC correction的bin edges
- `insert_size_bin_edges.npy` - Insert size correction的bin edges
- `outliers/` - 包含blacklist文件
  - `outlier_indices.npz` - 极端outlier bins
  - `zero_coverage_indices.npz` - 零覆盖bins
- `DNase.idx_to_cell_type.csv` - DNase cell type索引映射
- `ATAC.idx_to_cell_type.csv` - ATAC cell type索引映射
- `rows_per_chrom_pre_exclusion.txt` - 每个染色体在exclusion前的行数

## 为什么不能直接用lionheart目录下的文件？

1. **LIONHEART工程目录不包含资源文件** - 这些文件需要单独从Zenodo下载
2. **独立性** - 我们的reproduction应该有自己的数据目录，不依赖外部路径
3. **可移植性** - 所有数据（BAM、参考基因组、资源文件）放在一起，便于管理和分享

## 如何使用

所有脚本现在使用统一的路径配置（`src/paths.py`）：

```python
from src.paths import get_resources_dir, get_reference_dir, get_bam_path

resources_dir = get_resources_dir()  # 自动找到我们的资源文件目录
reference_dir = get_reference_dir()  # 参考基因组目录
bam_path = get_bam_path()  # BAM文件路径
```

路径查找顺序：
1. 优先使用 `reproduce/data/lionheart_resources/`（我们的数据目录）
2. 如果不存在，尝试 `/home/mara/lionheart_work/resources`（原始路径，用于比较脚本）
3. 如果都不存在，使用 `lionheart_resources/`（备用路径）

## 复制资源文件

如果资源文件还没有复制，运行：

```bash
python setup_lionheart_resources.py
```

这会从 `/home/mara/lionheart_work/resources` 复制所有必要的资源文件到 `reproduce/data/lionheart_resources/`。

## 验证

运行测试脚本验证资源文件是否正确设置：

```bash
python test_new_resources_path.py
```

应该看到成功加载GC内容的输出。


