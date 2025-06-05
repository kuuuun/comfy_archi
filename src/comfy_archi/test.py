import toml
import pprint
from pathlib import Path

file_name = Path(__file__).resolve().parent / "prompt.toml"
with file_name.open("r", encoding="utf-8") as file:
    data = toml.load(file)
# num = len(data)
class_list = list(data)
elements_general = list(data[class_list[0]])
elements_styles = list(data[class_list[1]])
elements_types = list(data[class_list[2]])
elements_materials = list(data[class_list[3]])

pprint.pp(elements_general)
for v in elements_general:
    print(v)
pprint.pp(elements_types)
pprint.pp(elements_styles)
pprint.pp(elements_materials)
pprint.pp(data)
