

from dataclasses import dataclass, asdict, field


EMPTY_DICT = {}
class Base:
    """base class for all of our models"""

    def __init_subclass__(cls):
        """always define dataclasses"""
        dataclass(cls)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def fromkeys(cls):
        return dict.fromkeys(cls.__dataclass_fields__)

    def get_dict(self):
        return asdict(self)

    @classmethod
    def get_series(cls):
        import pandas

        return pandas.Series(cls.fromkeys())

    @classmethod
    def get_dtype(cls):
        data = dict.fromkeys(cls.__dataclass_fields__)
        for k in data:
            data[k] = "O"
        return data
    
    @classmethod
    def properties(cls):
        """in the json schema parlance, return the properties or dataclass keys."""

        return list(cls.__dataclass_fields__)
    

