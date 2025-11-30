"""
统一的路径配置
所有脚本应该使用这个模块来获取资源文件路径
"""
import os
from pathlib import Path


def get_resources_dir():
    """
    获取LIONHEART资源文件目录
    优先使用inference_resources_v003（用户提供的完整资源目录）
    其次使用我们数据目录中的资源文件（和example_bam、参考基因组一起）
    如果不存在，则尝试使用原始路径
    """
    if os.path.exists("/mnt/d"):
        base_path = "/mnt/d/MADS/25 Fall/CSC 527/Final project"
        # 优先使用inference_resources_v003（用户提供的完整资源目录）
        resources_dir = Path(f"{base_path}/inference_resources_v003")
        if resources_dir.exists():
            return resources_dir
        # 其次使用我们数据目录中的资源文件
        resources_dir = Path(f"{base_path}/reproduce/data/lionheart_resources")
        if not resources_dir.exists():
            # 如果不存在，尝试使用原始路径（用于比较脚本）
            resources_dir = Path("/home/mara/lionheart_work/resources")
            if not resources_dir.exists():
                resources_dir = Path(f"{base_path}/lionheart_resources")
    else:
        resources_dir = Path("data/lionheart_resources")
        if not resources_dir.exists():
            resources_dir = Path("../lionheart_resources")
    
    return resources_dir


def get_base_path():
    """获取项目基础路径"""
    if os.path.exists("/mnt/d"):
        return "/mnt/d/MADS/25 Fall/CSC 527/Final project"
    else:
        return Path(__file__).parent.parent.parent


def get_bam_path():
    """获取示例BAM文件路径"""
    base_path = get_base_path()
    if os.path.exists("/mnt/d"):
        return f"{base_path}/example_bam_hg38/IC38.hg38.downsampled.aligned.sorted.markdup.bam"
    else:
        return "data/demo.bam"


def get_reference_dir():
    """获取参考基因组目录"""
    base_path = get_base_path()
    if os.path.exists("/mnt/d"):
        return Path(f"{base_path}/reproduce/data")
    else:
        return Path("data")


def get_lionheart_output_dir():
    """获取LIONHEART输出目录（用于比较）"""
    if os.path.exists("/mnt/d"):
        return Path("/home/mara/lionheart_work/output/dataset")
    else:
        return Path("/home/mara/lionheart_work/output/dataset")

