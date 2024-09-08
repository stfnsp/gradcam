from typing import Optional

import typer
from typing_extensions import Annotated

from gradcam.attention_map import get_attention_map
from gradcam.utils import save_attention_map


def main(
    image_path: Annotated[str, typer.Argument()],
    result_path: Annotated[str, typer.Argument()],
    caption_text: Annotated[str, typer.Argument()],
    model_name: Annotated[str, typer.Argument()],
    pretrain_tag: Annotated[Optional[str], typer.Argument()] = None,
):
    attention_map = get_attention_map(
        image_path, caption_text, model_name, pretrain_tag
    )
    save_attention_map(attention_map, image_path, result_path)


if __name__ == "__main__":
    typer.run(main)
