"""
Mamba训练配置管理器
提供配置模板生成、参数搜索和消融实验管理功能
"""

import os
import sys
import json
import yaml
import copy
import itertools
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum


class ConfigFormat(Enum):
    """配置格式枚举"""
    TXT = 'txt'
    JSON = 'json'
    YAML = 'yaml'


@dataclass
class HyperparameterRange:
    """超参数范围定义"""
    name: str
    values: List[Any]
    description: str = ""
    

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    base_config: Dict[str, Any]
    hyperparameter_ranges: List[HyperparameterRange] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            templates_dir: 模板目录路径
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / 'config_templates'
        self.templates_dir = Path(templates_dir)
        
        # 加载所有模板
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载所有配置模板"""
        templates = {}
        
        for template_file in self.templates_dir.glob('*.yaml'):
            try:
                with open(template_file, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # 处理继承关系
                if '_base_' in config:
                    base_file = self.templates_dir / config['_base_']
                    if base_file.exists():
                        with open(base_file, 'r') as f:
                            base_config = yaml.safe_load(f)
                        # 合并配置 (子配置覆盖父配置)
                        merged_config = {**base_config, **config}
                        merged_config.pop('_base_', None)
                        templates[template_file.stem] = merged_config
                    else:
                        print(f"警告: 基础配置文件不存在: {base_file}")
                        templates[template_file.stem] = config
                else:
                    templates[template_file.stem] = config
                    
            except Exception as e:
                print(f"加载模板失败 {template_file}: {e}")
                
        return templates
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """获取配置模板"""
        if template_name not in self.templates:
            raise ValueError(f"模板不存在: {template_name}")
        return copy.deepcopy(self.templates[template_name])
    
    def create_config(self, 
                     template_name: str,
                     output_path: str,
                     format: ConfigFormat = ConfigFormat.YAML,
                     **overrides) -> Path:
        """
        创建配置文件
        
        Args:
            template_name: 模板名称
            output_path: 输出路径
            format: 输出格式
            **overrides: 覆盖参数
            
        Returns:
            创建的配置文件路径
        """
        # 获取模板
        config = self.get_template(template_name)
        
        # 应用覆盖参数
        config.update(overrides)
        
        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存配置文件
        if format == ConfigFormat.TXT:
            self._save_txt_config(config, output_path)
        elif format == ConfigFormat.JSON:
            self._save_json_config(config, output_path)
        elif format == ConfigFormat.YAML:
            self._save_yaml_config(config, output_path)
        else:
            raise ValueError(f"不支持的格式: {format}")
            
        return output_path
    
    def _save_txt_config(self, config: Dict[str, Any], output_path: Path):
        """保存为TXT格式"""
        with open(output_path, 'w') as f:
            for key, value in sorted(config.items()):
                # 处理列表和字典
                if isinstance(value, list):
                    value_str = '[' + ', '.join(str(v) for v in value) + ']'
                elif isinstance(value, dict):
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)
                f.write(f"{key} = {value_str}\n")
    
    def _save_json_config(self, config: Dict[str, Any], output_path: Path):
        """保存为JSON格式"""
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _save_yaml_config(self, config: Dict[str, Any], output_path: Path):
        """保存为YAML格式"""
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def generate_hyperparameter_search(self,
                                      template_name: str,
                                      output_dir: str,
                                      hyperparameters: List[HyperparameterRange],
                                      format: ConfigFormat = ConfigFormat.YAML) -> List[Path]:
        """
        生成超参数搜索配置
        
        Args:
            template_name: 模板名称
            output_dir: 输出目录
            hyperparameters: 超参数范围列表
            format: 输出格式
            
        Returns:
            生成的配置文件路径列表
        """
        # 获取基础配置
        base_config = self.get_template(template_name)
        
        # 生成所有超参数组合
        param_names = [hp.name for hp in hyperparameters]
        param_values = [hp.values for hp in hyperparameters]
        
        config_files = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, combination in enumerate(itertools.product(*param_values)):
            # 创建配置副本
            config = copy.deepcopy(base_config)
            
            # 应用超参数
            for name, value in zip(param_names, combination):
                config[name] = value
            
            # 生成配置名称
            param_str = '_'.join(f"{name}_{value}" for name, value in zip(param_names, combination))
            config_name = f"{template_name}_{param_str}"
            config['experiment_name'] = config_name
            
            # 保存配置文件
            output_path = output_dir / f"{config_name}.{format.value}"
            self._save_config_by_format(config, output_path, format)
            
            config_files.append(output_path)
            
            print(f"生成配置 {i+1}: {output_path}")
            
        return config_files
    
    def _save_config_by_format(self, config: Dict[str, Any], output_path: Path, format: ConfigFormat):
        """根据格式保存配置"""
        if format == ConfigFormat.TXT:
            self._save_txt_config(config, output_path)
        elif format == ConfigFormat.JSON:
            self._save_json_config(config, output_path)
        elif format == ConfigFormat.YAML:
            self._save_yaml_config(config, output_path)
    
    def create_ablation_study(self,
                             template_name: str,
                             output_dir: str,
                             ablation_params: Dict[str, List[Any]],
                             format: ConfigFormat = ConfigFormat.YAML) -> List[Path]:
        """
        创建消融实验配置
        
        Args:
            template_name: 模板名称
            output_dir: 输出目录
            ablation_params: 消融参数 {参数名: [值列表]}
            format: 输出格式
            
        Returns:
            生成的配置文件路径列表
        """
        # 获取基础配置
        base_config = self.get_template(template_name)
        
        config_files = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每个消融参数创建配置
        for param_name, values in ablation_params.items():
            for i, value in enumerate(values):
                # 创建配置副本
                config = copy.deepcopy(base_config)
                
                # 应用消融参数
                config[param_name] = value
                
                # 生成配置名称
                config_name = f"{template_name}_ablation_{param_name}_{i}"
                config['experiment_name'] = config_name
                config['tags'] = config.get('tags', []) + ['ablation', f'ablation_{param_name}']
                
                # 保存配置文件
                output_path = output_dir / f"{config_name}.{format.value}"
                self._save_config_by_format(config, output_path, format)
                
                config_files.append(output_path)
                
                print(f"生成消融配置 {param_name}={value}: {output_path}")
                
        return config_files
    
    def validate_config(self, config_path: str) -> bool:
        """
        验证配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            是否有效
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                print(f"配置文件不存在: {config_path}")
                return False
            
            # 加载配置
            if config_path.suffix == '.txt':
                # 简单验证TXT格式
                with open(config_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # 基本验证
                            if not key or not value:
                                print(f"无效的行: {line}")
                                return False
            elif config_path.suffix in ['.json', '.yaml', '.yml']:
                # 验证JSON/YAML格式
                with open(config_path, 'r') as f:
                    if config_path.suffix == '.json':
                        config = json.load(f)
                    else:
                        config = yaml.safe_load(f)
                        
                # 检查必需字段
                required_fields = ['device', 'lr', 'N_eps', 'model_type']
                for field in required_fields:
                    if field not in config:
                        print(f"缺少必需字段: {field}")
                        return False
                        
            else:
                print(f"不支持的配置文件格式: {config_path.suffix}")
                return False
                
            return True
            
        except Exception as e:
            print(f"验证配置文件失败 {config_path}: {e}")
            return False
    
    def compare_configs(self, config_paths: List[str]) -> Dict[str, List[Any]]:
        """
        比较多个配置文件
        
        Args:
            config_paths: 配置文件路径列表
            
        Returns:
            差异字典 {参数名: [各配置的值]}
        """
        configs = []
        
        # 加载所有配置
        for config_path in config_paths:
            config_path = Path(config_path)
            if not config_path.exists():
                print(f"配置文件不存在: {config_path}")
                continue
                
            try:
                if config_path.suffix == '.txt':
                    config = self._load_txt_config(config_path)
                elif config_path.suffix == '.json':
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                elif config_path.suffix in ['.yaml', '.yml']:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    print(f"不支持的格式: {config_path.suffix}")
                    continue
                    
                configs.append((config_path.name, config))
                
            except Exception as e:
                print(f"加载配置文件失败 {config_path}: {e}")
                continue
        
        if len(configs) < 2:
            print("需要至少2个配置文件进行比较")
            return {}
        
        # 收集所有参数名
        all_params = set()
        for _, config in configs:
            all_params.update(config.keys())
        
        # 比较参数值
        differences = {}
        for param in sorted(all_params):
            values = []
            for config_name, config in configs:
                values.append(config.get(param, "未设置"))
            
            # 检查是否有差异
            # 将值转换为可哈希的类型
            hashable_values = []
            for v in values:
                if isinstance(v, list):
                    hashable_values.append(tuple(v))
                elif isinstance(v, dict):
                    hashable_values.append(json.dumps(v, sort_keys=True))
                else:
                    hashable_values.append(v)
            
            if len(set(hashable_values)) > 1:
                differences[param] = values
        
        return differences
    
    def _load_txt_config(self, config_path: Path) -> Dict[str, Any]:
        """加载TXT格式配置"""
        config = {}
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 尝试解析值
                    try:
                        # 尝试解析为列表
                        if value.startswith('[') and value.endswith(']'):
                            value = json.loads(value)
                        # 尝试解析为数字
                        elif value.replace('.', '').replace('-', '').isdigit():
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        # 尝试解析为布尔值
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                    except:
                        pass
                    
                    config[key] = value
                    
        return config
    
    def generate_training_script(self,
                               config_paths: List[str],
                               output_script: str,
                               gpu_ids: Optional[List[int]] = None) -> Path:
        """
        生成批量训练脚本
        
        Args:
            config_paths: 配置文件路径列表
            output_script: 输出脚本路径
            gpu_ids: GPU ID列表
            
        Returns:
            生成的脚本路径
        """
        import time
        output_script = Path(output_script)
        output_script.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成脚本内容
        script_content = "#!/bin/bash\n"
        script_content += "# 自动生成的Mamba批量训练脚本\n"
        script_content += "# 生成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
        
        script_content += "set -e  # 出错时退出\n\n"
        
        # 设置环境变量
        script_content += "# 设置Python路径\n"
        script_content += "export PYTHONPATH=$PYTHONPATH:$(pwd)\n\n"
        
        # 训练命令
        script_content += "echo '开始批量训练...'\n"
        script_content += "start_time=$(date +%s)\n\n"
        
        # 为每个配置生成训练命令
        for i, config_path in enumerate(config_paths):
            config_path = Path(config_path)
            if not config_path.exists():
                print(f"警告: 配置文件不存在: {config_path}")
                continue
            
            # 设置GPU
            gpu_cmd = ""
            if gpu_ids:
                gpu_id = gpu_ids[i % len(gpu_ids)]
                gpu_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} "
            
            # 训练命令
            script_content += f"echo '训练配置 {i+1}/{len(config_paths)}: {config_path.name}'\n"
            script_content += f"{gpu_cmd}python training/mamba_infra/train_mamba_branch.py \\\n"
            script_content += f"  --config {config_path} \\\n"
            script_content += f"  2>&1 | tee {config_path.stem}_train.log\n\n"
            
            script_content += f"echo '配置 {config_path.name} 训练完成'\n"
            script_content += "echo '----------------------------------------'\n\n"
        
        # 计算总时间
        script_content += "end_time=$(date +%s)\n"
        script_content += "total_time=$((end_time - start_time))\n"
        script_content += "echo \"批量训练完成! 总时间: ${total_time}秒\"\n"
        
        # 保存脚本
        with open(output_script, 'w') as f:
            f.write(script_content)
        
        # 设置执行权限
        os.chmod(output_script, 0o755)
        
        print(f"生成批量训练脚本: {output_script}")
        return output_script


# 示例使用
if __name__ == '__main__':
    import time
    
    # 创建配置管理器
    manager = ConfigManager()
    
    # 示例1: 创建单个配置
    print("示例1: 创建分支B配置")
    config_path = manager.create_config(
        template_name='branch_B_config',
        output_path='configs/branch_B_experiment.yaml',
        experiment_name='my_branch_B_experiment',
        lr=1.5e-4,
        N_eps=100
    )
    print(f"创建配置: {config_path}")
    
    # 示例2: 生成超参数搜索
    print("\n示例2: 生成超参数搜索配置")
    hyperparams = [
        HyperparameterRange('lr', [1e-4, 5e-4, 1e-3], '学习率'),
        HyperparameterRange('batch_size', [1, 2, 4], '批大小'),
        HyperparameterRange('weight_decay', [1e-5, 1e-4, 1e-3], '权重衰减')
    ]
    
    search_configs = manager.generate_hyperparameter_search(
        template_name='base_config',
        output_dir='configs/hyperparam_search',
        hyperparameters=hyperparams
    )
    print(f"生成 {len(search_configs)} 个超参数配置")
    
    # 示例3: 创建消融实验
    print("\n示例3: 创建消融实验配置")
    ablation_params = {
        'lr_decay': [True, False],
        'loss_type': ['mse', 'l1', 'huber'],
        'optimizer': ['adamw', 'adam', 'sgd']
    }
    
    ablation_configs = manager.create_ablation_study(
        template_name='branch_C_config',
        output_dir='configs/ablation_study',
        ablation_params=ablation_params
    )
    print(f"生成 {len(ablation_configs)} 个消融配置")
    
    # 示例4: 验证配置
    print("\n示例4: 验证配置")
    is_valid = manager.validate_config(config_path)
    print(f"配置 {config_path} 有效: {is_valid}")
    
    # 示例5: 生成批量训练脚本
    print("\n示例5: 生成批量训练脚本")
    all_configs = search_configs + ablation_configs
    if all_configs:
        script_path = manager.generate_training_script(
            config_paths=all_configs[:3],  # 只使用前3个作为示例
            output_script='scripts/batch_train.sh',
            gpu_ids=[0, 1]
        )
        print(f"生成批量训练脚本: {script_path}")