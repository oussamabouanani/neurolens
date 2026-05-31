from pathlib import Path
from string import Formatter


def validate_text_template(templates: str | list[str]) -> list[str]:
    if isinstance(templates, str):
        _templates = [templates]
    else:
        _templates = templates

    for next_template in _templates:
        validate_simple_unique_ordered_fields(next_template, "")

    return _templates


def validate_simple_unique_ordered_fields(str, *field_names) -> None:

    expected_fields = [(name, "", None) for name in field_names]

    try:
        fields = [
            (field_name, format_spec, conversion)
            for _, field_name, format_spec, conversion in Formatter().parse(str)
            if field_name is not None
        ]
    except ValueError as e:
        raise ValueError(f"Text str {str!r} is not a valid parsable string") from e

    if fields != expected_fields:
        raise ValueError(
            f"Text str {str!r} must contain exactly the following fields once in this order: {field_names}"
        )


def validate_suffix(*suffixes: str, value: str) -> None:

    if not suffixes:
        raise ValueError("at least one suffix is required")

    normalized_suffixes = tuple(suffix if suffix.startswith(".") else f".{suffix}" for suffix in suffixes)
    if not value.endswith(normalized_suffixes):
        expected = " or ".join(repr(suffix) for suffix in normalized_suffixes)
        raise ValueError(f"filename must end with {expected}")


def validate_plain_path_component(value: str) -> None:

    if not value:
        raise ValueError("must not be empty")

    path = Path(value)

    if path.name != value or value in {".", ".."}:
        raise ValueError("must be a plain name, not a path")
