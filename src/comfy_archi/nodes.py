import toml
from pathlib import Path


class SellectPosPrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        file_name = Path(__file__).resolve().parent / "positive_prompt.toml"
        with file_name.open("r", encoding="utf-8") as file:
            data = toml.load(file)
        # num = len(data)
        elements_general = list(data["general"])
        elements_styles = list(data["styles"])
        elements_types = list(data["types"])
        elements_materials = list(data["materials"])

        return {
            "required": {
                "clip": ("CLIP", {"forceInput": True, "tooltip": "A CLIP model used for encoding the text."}),
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "beautiful scence,",
                    },
                ),
                "building_general": (elements_general,),
                "building_styles": (elements_styles,),
                "building_types": (elements_types,),
                "building_materials": (elements_materials,),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "combine_prompt"

    # OUTPUT_NODE = False

    CATEGORY = "Archi24/TestNodes"

    def combine_prompt(self, clip, positive_prompt, building_general, building_styles, building_types, building_materials):
        file_name = Path(__file__).resolve().parent / "positive_prompt.toml"
        with file_name.open("r", encoding="utf-8") as file:
            data = toml.load(file)
        # 验证用户输入的值是否存在于 data 中
        if building_general not in data["general"]:
            raise ValueError(f"Invalid value for building_general: '{building_general}'. Available options: {list(data['general'].keys())}")
        if building_styles not in data["styles"]:
            raise ValueError(f"Invalid value for building_styles: '{building_styles}'. Available options: {list(data['styles'].keys())}")
        if building_types not in data["types"]:
            raise ValueError(f"Invalid value for building_types: '{building_types}'. Available options: {list(data['types'].keys())}")
        if building_materials not in data["materials"]:
            raise ValueError(
                f"Invalid value for building_materials: '{building_materials}'. Available options: {list(data['materials'].keys())}"
            )

        result = f"{positive_prompt}\n{data['general'][building_general]}\n{data['styles'][building_styles]}\n{data['types'][building_types]}\n{data['materials'][building_materials]}"
        # return (result,)
        tokens = clip.tokenize(result)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


class SellectNegPrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        file_name = Path(__file__).resolve().parent / "negative_prompt.toml"
        with file_name.open("r", encoding="utf-8") as file:
            data = toml.load(file)
        # num = len(data)
        elements_general1 = list(data["general1"])
        elements_general2 = list(data["general2"])

        return {
            "required": {
                "clip": ("CLIP", {"forceInput": True, "tooltip": "A CLIP model used for encoding the text."}),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "nsfw, ",
                    },
                ),
                "general1": (elements_general1,),
                "general2": (elements_general2,),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "combine_prompt"

    # OUTPUT_NODE = False
    # OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Archi24/TestNodes"

    def combine_prompt(self, clip, negative_prompt, general1, general2):
        file_name = Path(__file__).resolve().parent / "negative_prompt.toml"
        with file_name.open("r", encoding="utf-8") as file:
            data = toml.load(file)
        # 验证用户输入的值是否存在于 data 中
        if general1 not in data["general1"]:
            raise ValueError(f"Invalid value for general1: '{general1}'. Available options: {list(data['general1'].keys())}")
        if general2 not in data["general2"]:
            raise ValueError(f"Invalid value for general2: '{general2}'. Available options: {list(data['general2'].keys())}")

        result = f"{negative_prompt}\n{data['general1'][general1]}\n{data['general2'][general2]}"
        tokens = clip.tokenize(result)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SellectPosPrompt": SellectPosPrompt,
    "SellectNegPrompt": SellectNegPrompt,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SellectPosPrompt": "Sellect Positive Prompt",
    "SellectNegPrompt": "Sellect Negative Prompt",
}
