import toml
from pathlib import Path


class ArchFunc:
    """
    基础功能
    """

    def __init__(self):
        pass

    @staticmethod
    def _load_toml(file_name):
        """
        读取 toml 文件
        """
        try:
            with file_name.open("r", encoding="utf-8") as file:
                return toml.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_name}")
        except toml.TomlDecodeError:
            raise ValueError(f"Invalid TOML format in file: {file_name}")

    @staticmethod
    def _validate_input(value, data, key) -> None:
        """
        检验输入
        """
        if value not in data[key]:
            raise ValueError(f"Invalid value for {key}: '{value}'. Available options: {list(data[key].keys())}")

    @staticmethod
    def _clip_condition(clip, text):
        """
        Clip 转 conditioning
        """
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


class SelectPosPrompt(ArchFunc):
    POSITIVE_PROMPT_FILE = Path(__file__).resolve().parent / "positive_prompt.toml"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
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

    CATEGORY = "Archi24/Clip"

    def combine_prompt(self, clip, positive_prompt, building_general, building_styles, building_types, building_materials):
        data = self._load_toml(self.POSITIVE_PROMPT_FILE)

        self._validate_input(building_general, data, "general")
        self._validate_input(building_styles, data, "styles")
        self._validate_input(building_types, data, "types")
        self._validate_input(building_materials, data, "materials")

        text = f"{positive_prompt}\n{data['general'][building_general]}\n{data['styles'][building_styles]}\n{data['types'][building_types]}\n{data['materials'][building_materials]}"

        return self._clip_condition(clip, text)


class SelectNegPrompt(ArchFunc):
    NEGATIVE_PROMPT_FILE = Path(__file__).resolve().parent / "negative_prompt.toml"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
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

    CATEGORY = "Archi24/Clip"

    def combine_prompt(self, clip, negative_prompt, general1, general2):
        data = self._load_toml(self.NEGATIVE_PROMPT_FILE)
        # 验证用户输入的值是否存在于 data 中
        self._validate_input(general1, data, "general1")
        self._validate_input(general2, data, "general2")

        text = f"{negative_prompt}\n{data['general1'][general1]}\n{data['general2'][general2]}"

        return self._clip_condition(clip, text)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SelectPosPrompt": SelectPosPrompt,
    "SelectNegPrompt": SelectNegPrompt,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectPosPrompt": "Select Positive Prompt",
    "SelectNegPrompt": "Select Negative Prompt",
}
