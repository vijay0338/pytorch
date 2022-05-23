from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Type

from torch.ao.quantization import QuantType


@dataclass
class StandaloneModuleNameConfigEntry:
    module_name: str
    # qconfig_dict for the prepare function called in the submodule,
    # None means use qconfig from parent qconfig_dict
    # TODO: replace this with QConfigMapping
    qconfig_dict: Dict[str, Any]
    prepare_custom_config: PrepareCustomConfig
    # TODO: replace this with BackendConfig
    backend_config_dict: Dict


@dataclass
class StandaloneModuleClassConfigEntry:
    module_class: Type
    # qconfig_dict for the prepare function called in the submodule,
    # None means use qconfig from parent qconfig_dict
    # TODO: replace this with QConfigMapping
    qconfig_dict: Dict[str, Any]
    prepare_custom_config: PrepareCustomConfig
    # TODO: replace this with BackendConfig
    backend_config_dict: Dict


class PrepareCustomConfig:
    """
    TODO: write this
    """

    def __init__(self):
        self.standalone_module_name_configs: List[StandaloneModuleNameConfigEntry] = []
        self.standalone_module_class_configs: List[StandaloneModuleClassConfigEntry] = []
        self.float_to_observed_mapping: Dict[QuantType, Dict[Type, Type]] = {}
        self.non_traceable_module_names: List[str] = []
        self.non_traceable_module_classes: List[Type] = []
        self.input_quantized_indexes: List[int] = []
        self.output_quantized_indexes: List[int] = []
        self.preserved_attributes: List[str] = []

    def set_standalone_module_name(
            self,
            module_name: str,
            qconfig_dict: Dict[str, Any],
            prepare_custom_config: PrepareCustomConfig,
            backend_config: Dict) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.standalone_module_name_configs.append(
            StandaloneModuleNameConfigEntry(
                module_name, qconfig_dict, prepare_custom_config, backend_config_dict))
        return self

    def set_standalone_module_class(
            self,
            module_class: str,
            qconfig_dict: Dict[str, Any],
            prepare_custom_config: PrepareCustomConfig,
            backend_config: Dict) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.standalone_module_class_configs.append(
            StandaloneModuleClassConfigEntry(
                module_class, qconfig_dict, prepare_custom_config, backend_config_dict))
        return self

    def set_float_to_observed_mapping(
            self,
            float_class: Type,
            observed_class: Type,
            quant_type: QuantType = QuantType.STATIC) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        if quant_type not in self.float_to_observed_mapping:
            self.float_to_observed_mapping[quant_type] = {}
        self.float_to_observed_mapping[quant_type][float_class] = observed_class
        return self

    def set_non_traceable_module_names(self, module_names: List[str]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.non_traceable_module_names = module_names
        return self

    def set_non_traceable_module_class(self, module_classes: List[Type]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.non_traceable_module_classes = module_classes
        return self

    def set_input_quantized_index(self, indexes: List[int]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.input_quantized_indexes = indexes
        return self
        
    def set_output_quantized_index(self, indexes: List[int]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.output_quantized_indexes = indexes
        return self

    def set_preserved_attribute(self, attributes: List[str]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.preserved_attributes = attributes
        return self


class ConvertCustomConfig:
    """
    TODO: write this
    """

    def __init__(self):
        self.observed_to_quantized_mapping: Dict[QuantType, Dict[Type, Type]] = {}

    def set_observed_to_quantized_mapping(
            self,
            observed_class: Type,
            quantized_class: Type,
            quant_type: QuantType = QuantType.STATIC) -> ConvertsCcustomConfig:
        """
        TODO: write this
        """
        if quant_type not in self.observed_to_quantized_mapping:
            self.observed_to_quantized_mapping[quant_type] = {}
        self.observed_to_quantized_mapping[quant_type][observed_class] = quantized_class
        return self
