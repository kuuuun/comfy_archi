#!/usr/bin/env python

"""Tests for `comfy_archi` package."""

import pytest
from unittest.mock import MagicMock
from src.comfy_archi.nodes import SellectPosPrompt, SellectNegPrompt


@pytest.fixture
def example_node():
    """Fixture to create an Example node instance."""
    return SellectPosPrompt()


@pytest.fixture
def mock_clip():
    clip = MagicMock()
    clip.tokenize.return_value = ["token1", "token2"]
    clip.encode_from_tokens.return_value = ([1, 2, 3], {"pooled_output": [4, 5, 6]})
    return clip


# Fixture 用于创建测试对象
@pytest.fixture
def your_class_instance():
    return SellectPosPrompt()


# 测试有效输入
def test_combine_prompt_valid(your_class_instance):
    clip = mock_clip
    positive_prompt = "Positive Prompt"
    building_general = "option1"
    building_styles = "style1"
    building_types = "type1"
    building_materials = "material1"

    result = your_class_instance.combine_prompt(
        clip=clip,
        positive_prompt=positive_prompt,
        building_general=building_general,
        building_styles=building_styles,
        building_types=building_types,
        building_materials=building_materials,
    )

    expected_result = (
        [
            [
                [1, 2, 3],  # 模拟的 cond
                {"pooled_output": [4, 5, 6]},  # 模拟的 pooled
            ]
        ],
    )

    assert result == expected_result


def test_example_node_initialization(example_node):
    """Test that the node can be instantiated."""
    assert isinstance(example_node, SellectPosPrompt)


def test_example_node(example_node):
    """Test the example node."""
    assert example_node is not None  # 确保 example_node 不为 None
    # 进行其他测试逻辑


def test_return_types():
    """Test the node's metadata."""
    assert SellectPosPrompt.RETURN_TYPES == ("CONDITIONING",)
    assert SellectPosPrompt.FUNCTION == "combine_prompt"
    assert SellectPosPrompt.CATEGORY == "Archi24/TestNodes"
    assert SellectNegPrompt.RETURN_TYPES == ("CONDITIONING",)
