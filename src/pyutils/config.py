"""
This module provides a structured and dynamic approach to loading configuration files.
The term dynamic refers that it is not required to know the sections ahead of time.
Moreover, since configurations are classes and not instances of a class if so,
it is reassured that they will be loaded only one time when the module is imported.
If this module imported again it is cached in sys.modules.
By leveraging metaclasses, it eliminates the need for manually defining configuration classes,
making the code more scalable and maintainable.
"""

import configparser
import json
from pathlib import Path

base_dir = Path(__file__).parent.resolve()
config_dir = base_dir / "configurations"

__all__ = ["ConfigMeta"]


class SectionType(type):
    """
    Description

    A metaclass responsible for creating section classes that store configuration key-value pairs.

    Attributes

    section_name (str):
        The name of the configuration section.

    section_attrs (dict):
        Additional attributes corresponding to key-value pairs from the configuration file.
    """

    def __new__(mcls, name, bases, cls_attrs, section_name, section_attrs):
        cls_attrs["__doc__"] = f"Configurations for the {section_name} section"
        cls_attrs["section_name"] = section_name
        for key, value in section_attrs.items():
            cls_attrs[key] = value

        return super().__new__(mcls, name, bases, cls_attrs)


class ConfigMeta(type):
    """
    Description

    A metaclass responsible for dynamically creating configuration classes
    that load settings from an .ini or .json file. It loads configurations dynamically as it is not required to
    know the sections ahead of time.

    Attributes

    config_path (Path): The full path to the configuration file.

    Sections: Each section in the configuration file is dynamically assigned as a class attribute,
              with its keys mapped as attributes of a dynamically generated section class.
    """

    def __new__(mcls, name, bases, cls_attrs, config_filename="prod.ini", config_directory=config_dir):
        """
        config_filename: str
            The name of the configuration file for the environment ('prod', 'dev')
        config_dir: str
            The directory where the configuration file is located
        """
        config_path = Path(config_directory).resolve().joinpath(config_filename)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        cls_attrs["config_path"] = config_path
        cls_attrs["__doc__"] = f"Configurations for the {config_path.stem} environment"
        config = configparser.ConfigParser()
        if config_path.suffix == ".json":
            config_dict = json.load(open(config_path))
            config.read_dict(config_dict)
        else:
            config.read(config_path)

        global_section_attrs = config["Globals"] if "Globals" in config else {}
        for section_name in config.sections():
            class_name = section_name.capitalize()
            cls_attr_name = section_name.casefold()
            section_attrs = config[section_name]

            # Needed so the get method of each Section has access to global_sections
            def make_getter(section_attrs, global_section_attrs):
                """
                Define a get function that get with resolution order local configs - global configs -default value
                So, each section which is a class will look for a key firstly in its attribute, if the key does
                not exist in the section(class attributes) then it will look for the key in the Global section

                Args:
                    section_attrs: Local configurations(Configurations under the section)
                    global_section_attrs: Global configurations(Found under the section Global)

                Returns:
                    Closure with access to local and global configurations.
                """

                # The get method of each section which is stored as class attribute has access
                # both to global and local configurations
                def get(attr, default=None, cast=None):
                    val = section_attrs.get(attr, None)
                    if val is None:
                        val = global_section_attrs.get(attr, default)
                    if cast and val is not None:
                        try:
                            return cast(val)
                        except:
                            return default
                    return val

                return get

            Section = SectionType(
                class_name,
                (object,),
                {"get": make_getter(section_attrs, global_section_attrs)},
                section_name=section_name,
                section_attrs=section_attrs,
            )

            cls_attrs[cls_attr_name] = Section

        return super().__new__(mcls, name, bases, cls_attrs)
