#!/usr/bin/env python

"""Tests for `comfy_archi` package."""

import pytest
from unittest.mock import MagicMock
from src.comfy_archi.nodes import SelectPosPrompt, SelectNegPrompt, ArchFunc


@pytest.fixture
def sample_class():
    """Fixture to create an Example node instance."""
    return SelectPosPrompt()


@pytest.fixture
def mock_clip():
    clip = MagicMock()
    clip.tokenize.return_value = ["token1", "token2"]
    clip.encode_from_tokens.return_value = ([1, 2, 3], {"pooled_output": [4, 5, 6]})
    return clip


# Fixture 用于创建测试对象
@pytest.fixture
def your_class_instance():
    return SelectPosPrompt()


class TestValidateInput:
    @staticmethod
    def test_validate_input_success():
        # 测试正常输入
        value = "option1"
        data = {"key": {"option1": "value1", "option2": "value2"}}
        key = "key"

        # 调用方法，验证不抛出异常
        try:
            ArchFunc._validate_input(value, data, key)
        except ValueError:
            pytest.fail("ValueError was raised unexpectedly")


class TestCombinePrompt:
    def setup_method(self):
        self.obj = SelectPosPrompt()

        self.obj._load_toml = MagicMock()
        self.obj._validate_input = MagicMock()
        self.obj._clip_condition = MagicMock()

        self.data = {
            "general": {"general1": "value1"},
            "styles": {"style1": "value2"},
            "types": {"type1": "value3"},
            "materials": {"material1": "value4"},
        }

        self.obj._load_toml.return_value = self.data
        self.obj._clip_condition.return_value = "Processed clip"

    def test_combine_prompt(self):
        clip = "clip_data"
        positive_prompt = "Initial prompt"
        building_general = "general1"
        building_styles = "style1"
        building_types = "type1"
        building_materials = "material1"

        # 定义预期的文本内容
        self.expected_text = (
            f"{positive_prompt}\n"
            f"{self.data['general'][building_general]}\n"
            f"{self.data['styles'][building_styles]}\n"
            f"{self.data['types'][building_types]}\n"
            f"{self.data['materials'][building_materials]}"
        )

        result = self.obj.combine_prompt(clip, positive_prompt, building_general, building_styles, building_types, building_materials)

        # 检查 _load_toml 是否被调用
        self.obj._load_toml.assert_called_once_with(self.obj.POSITIVE_PROMPT_FILE)

        # 检查 _validate_input 是否被正确调用
        self.obj._validate_input.assert_any_call(building_general, self.data, "general")
        self.obj._validate_input.assert_any_call(building_styles, self.data, "styles")
        self.obj._validate_input.assert_any_call(building_types, self.data, "types")
        self.obj._validate_input.assert_any_call(building_materials, self.data, "materials")

        # 检查 _clip_condition 是否被调用
        self.obj._clip_condition.assert_called_once_with(clip, self.expected_text)

        # 检查返回值
        assert result == "Processed clip"


def test_example_node_initialization(sample_class):
    """Test that the node can be instantiated."""
    assert isinstance(sample_class, SelectPosPrompt)


def test_example_node(sample_class):
    """Test the example node."""
    assert sample_class is not None  # 确保 example_node 不为 None
    # 进行其他测试逻辑


def test_return_types():
    """Test the node's metadata."""
    assert SelectPosPrompt.RETURN_TYPES == ("CONDITIONING",)
    assert SelectPosPrompt.FUNCTION == "combine_prompt"
    assert SelectPosPrompt.CATEGORY == "Archi24/Clip"
    assert SelectNegPrompt.RETURN_TYPES == ("CONDITIONING",)


# # 测试有效输入
# def test_combine_prompt_valid(your_class_instance):
#     clip = mock_clip
#     positive_prompt = "Positive Prompt"
#     building_general = "option1"
#     building_styles = "style1"
#     building_types = "type1"
#     building_materials = "material1"

#     result = your_class_instance.combine_prompt(
#         clip=clip,
#         positive_prompt=positive_prompt,
#         building_general=building_general,
#         building_styles=building_styles,
#         building_types=building_types,
#         building_materials=building_materials,
#     )

#     expected_result = (
#         [
#             [
#                 [1, 2, 3],  # 模拟的 cond
#                 {"pooled_output": [4, 5, 6]},  # 模拟的 pooled
#             ]
#         ],
#     )

#     assert result == expected_result
