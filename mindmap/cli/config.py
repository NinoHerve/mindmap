import argparse

from mindmap.core.utils import create_config, create_tmp_config, add_to_config

def main():
    parser = argparse.ArgumentParser(
        prog = "mindmap-config",
        description = "Manage MindMap configuration files."
    )

    # Create subparsers for the idfferent commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ----------------
    # Command: create
    # ----------------

    parser_create = subparsers.add_parser("create", help="Generate config form a root directory")
    parser_create.add_argument("root", type=str, help="Path to the dataset root folder")
    parser_create.add_argument("--fname", default="config.yml", help="Output filename (default: config.yml)")

    # ------------------
    # Command: template
    # ------------------

    parser_temp = subparsers.add_parser("template", help="Create an empty configuration template")
    parser_temp.add_argument("--fname", default="config_template.yml", help="Template filename")
    
    # -------------
    # Command: add
    # -------------

    parser_add = subparsers.add_parser("add", help="Add or update a property in the config")
    parser_add.add_argument("config_file", type=str, help="Path to the YAML config file")
    parser_add.add_argument("key", type=str, help="The key to add (e.g. DIR_NEW or STD_param)")
    parser_add.add_argument("value", type=str, help="The value associated with the key.")

    args = parser.parse_args()

    # Routing logic
    if args.command == "create":
        create_config(args.root, fname=args.fname)

    elif args.command == "template":
        create_tmp_config(fname=args.fname)

    elif args.command == "add":
        add_to_config(args.config_file, args.key, args.value)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()