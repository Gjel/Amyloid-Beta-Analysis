from typing import Optional

import paquo.projects
from paquo.images import QuPathProjectImageEntry


def get_image_name_from_path(path: str):
    file_name = path.split('/')[-1]
    image_name = file_name[:-8]
    return image_name

class Project:

    PROJECT = None
    PROJECT_PATH = "F:/QuPath Abeta images 100+/Qupath data/project.qpproj"

    @classmethod
    def get_project(cls):
        if not cls.PROJECT:
            cls.PROJECT = paquo.projects.QuPathProject(cls.PROJECT_PATH)
        return cls.PROJECT

    @classmethod
    def get_image(cls, name: str) -> Optional[QuPathProjectImageEntry]:
        project = cls.get_project()
        try:
            return next((entry for entry in project.images if entry.image_name == name))
        except StopIteration:
            return None

    @classmethod
    def get_image_from_path(cls, path: str) -> Optional[QuPathProjectImageEntry]:
        name = get_image_name_from_path(path)
        return cls.get_image(name)


