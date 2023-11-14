# one-command launcher to start training
import typer
from typing_extensions import Annotated

app = typer.Typer(help="FMEngine.")


@app.command()
def profile():
    raise NotImplementedError("Not implemented yet")

@app.command()
def train(
    ds_config_path: Annotated[str, typer.Option(help="Path to Deepspeed config file")],
    dataset_path: Annotated[str, typer.Option(help="Path to .jsonl dataset file")],
    model: Annotated[str, typer.Option(help="ckpt path to the model")],
    output_dir: Annotated[str, typer.Option(help="where to store output files")],
    auto_tuning: Annotated[
        bool,
        typer.Option(
            help="if fmengine should auto-tune training parameters to maximize performance"
        ),
    ],
):
    """
    Train a model on a given dataset.
    """
    print("Training...")


if __name__ == "__main__":
    app()
