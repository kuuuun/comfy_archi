#!/usr/bin/env python

"""Tests for `comfy_archi` package."""

import pytest
from unittest.mock import MagicMock
from src.comfy_archi.nodes import SelectPosPrompt, SelectNegPrompt


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
