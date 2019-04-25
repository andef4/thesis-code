import json
from pathlib import Path
import base64
import sys


def save_notebook(base_path, filepath):
    path = None
    nb = json.load(open(filepath))
    filename = filepath.name.rsplit('.', 1)[0]
    i = 0

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue

        for output in cell['outputs']:
            for key, value in output.get('data', {}).items():
                if key == 'image/png':
                    if path is None:
                        path = Path(base_path / 'images')
                        if not path.exists():
                            path.mkdir()
                        path = path / filename
                        if not path.exists():
                            path.mkdir()

                    image_data = base64.b64decode(value)
                    with open(path / f'{i}.png', 'wb') as f:
                        f.write(image_data)

                    i += 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'python {sys.argv[0]} <directory>')
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print('Directory does not exist')
        sys.exit(1)
    if not path.is_dir():
        print('Given path is not a directory')
        sys.exit(1)

    for filename in path.glob('**/*.ipynb'):
        if 'checkpoint' in filename.name:
            continue
        save_notebook(path, filename)
