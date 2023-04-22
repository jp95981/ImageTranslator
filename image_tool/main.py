import click
import gui as GUI


def _headless(_root_file_path: str) -> None:
    print(
        "Headless mode will only attempt to traverse the file directory and align+save the images."
    )


def _gui(root_file_path: str) -> None:
    print("Loading GUI...")
    GUI.main(root_file_path)


@click.command()
@click.option("--headless", default=False, help="Headless mode on(True) or off(False)")
@click.option(
    "--root-file-path",
    default=".",
    help="Set the location where the data is stored, the directory",
)
def main(headless: bool, root_file_path: str) -> None:
    """
    Program to do some image stuff
    """
    print(f"Starting program with following settings: ")
    headless_message = "on" if headless else "off"
    print(f"\t- headless mode: {headless_message}")
    print(f"\t- Root file location: {root_file_path}")

    if headless:
        _headless(root_file_path)
    else:
        _gui(root_file_path)


if __name__ == "__main__":
    main()
